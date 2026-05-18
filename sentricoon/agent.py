"""The main agent loop — wires every subsystem together.

  OBSERVE → PLAN → EXECUTE → VERIFY → (DIAGNOSE → REPAIR) → ESCALATE → MEMORY

Per-step retry counter is owned by the loop. Repairer enforces the cap.
Escalation triggers (budget / time / strategy-exhausted / loop / abort)
are checked once per iteration. Every event is logged to the episodic
JSONL log so the run is replayable from disk.

Contract: `Agent.run()` NEVER raises — all failures resolve to a
TaskOutcome with `status="escalated"` and a populated FailureReport.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .confirmation import (
    AlwaysDenyProvider,
    ConfirmationProvider,
    ConfirmationRequest,
)
from .diagnoser import Diagnoser, Diagnosis
from .escalation import (
    EscalationReason,
    FailureReport,
    TaskBudget,
    build_report,
    record_strategy,
    register_snapshot,
    should_escalate,
    strategy_fingerprint,
    write_report,
)
from .executor import Executor
from .memory import EpisodicLog, Memory
from .planner import PlanError, Planner, plan_risk_warnings, weak_predicate_warnings
from .repair import Repairer, RepairDecision
from .safety.guard import GuardError
from .safety.policies import REQUIRES_CONFIRMATION
from .snapshot import Snapshotter
from .state import Step, StepRecord, TaskState, step_to_dict
from .tools.base import ToolResult
from .verifier import Verification, Verifier


@dataclass
class TaskOutcome:
    run_id: str
    status: str   # "done" | "escalated"
    state: TaskState
    failure_report: FailureReport | None = None
    report_paths: tuple[Path, Path] | None = None  # (markdown, json) if written


class Agent:
    def __init__(
        self,
        *,
        run_id: str,
        planner: Planner,
        executor: Executor,
        verifier: Verifier,
        diagnoser: Diagnoser,
        repairer: Repairer,
        episodic: EpisodicLog,
        snapshotter: Snapshotter | None = None,
        confirmation_provider: ConfirmationProvider | None = None,
        memory: Memory | None = None,
        budget: TaskBudget | None = None,
        reports_dir: Path | None = None,
    ) -> None:
        self.run_id = run_id
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.diagnoser = diagnoser
        self.repairer = repairer
        self.episodic = episodic
        self.snapshotter = snapshotter
        # Default: deny confirmations. If the agent encounters a high-risk
        # step with no provider configured, it cannot proceed — that's the
        # safe default. Real usage passes CLIConfirmationProvider() or similar.
        self.confirmation_provider = confirmation_provider or AlwaysDenyProvider()
        self.memory = memory or Memory()
        self.budget = budget or TaskBudget()
        self.reports_dir = reports_dir

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, task: str, goal_class: str | None = None) -> TaskOutcome:
        state = TaskState(task=task, goal_class=goal_class)
        self.episodic.append_task_start(self.run_id, task, goal_class)

        # --- initial planning ---
        try:
            initial_plan = self.planner.create_plan(task, goal_class)
        except PlanError as e:
            return self._escalate(
                state, EscalationReason.UNRECOVERABLE_ERROR,
                summary=f"initial plan failed: {e}",
            )

        self._adopt_plan(state, initial_plan)

        # Per-step attempts counter. Cleared on every replan because
        # the plan indices are fresh after a replan.
        step_attempts: defaultdict[int, int] = defaultdict(int)

        # --- main loop ---
        while state.current_step < len(state.plan):
            # Escalation check happens BEFORE each step so budget/time
            # exhaustion can terminate the run cleanly.
            esc = should_escalate(
                state, self.budget,
                quarantine_source=self.memory.is_strategy_quarantined,
            )
            if esc is not None:
                return self._escalate(state, esc)

            step = state.plan[state.current_step]

            # Pre-step snapshot (best-effort; never blocks the loop)
            self._maybe_snapshot(state, step)

            # Confirmation gate for high-risk actions
            token = self._maybe_confirm(state, step)
            if token is False:  # explicit denial
                return self._escalate(
                    state, EscalationReason.ABORT_REQUESTED,
                    summary="user denied confirmation for a high-risk action",
                )

            # Execute + verify (returns the populated record + raw artifacts
            # needed by diagnoser if the step failed)
            record, result, verification = self._execute_and_verify(step, token)
            step_attempts[state.current_step] += 1

            decision: RepairDecision | None = None
            diagnosis: Diagnosis | None = None
            if not verification.success:
                diagnosis = self.diagnoser.diagnose(
                    step, result, verification, task=state.task,
                )
                decision = self.repairer.repair(
                    step, diagnosis, step_attempts[state.current_step],
                    task=state.task,
                )
                record.diagnosis = diagnosis.to_dict()
                record.repair_strategy = decision.strategy

            # Log the fully-populated record exactly once.
            state.history.append(record)
            self.episodic.append_step(self.run_id, record)

            if verification.success:
                state.current_step += 1
                continue

            # Apply repair decision
            assert decision is not None and diagnosis is not None
            outcome = self._apply_repair(state, step, decision, diagnosis, step_attempts)
            if outcome is not None:
                return outcome

        # All steps completed
        self.episodic.append_task_end(self.run_id, "done")
        return TaskOutcome(run_id=self.run_id, status="done", state=state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _adopt_plan(self, state: TaskState, plan: list[Step]) -> None:
        state.plan = plan
        state.current_step = 0
        fp = strategy_fingerprint(plan)
        record_strategy(state, plan)
        self.episodic.append_plan(
            self.run_id, [step_to_dict(s) for s in plan], strategy_fingerprint=fp,
        )
        self.episodic.append_strategy(self.run_id, fp, [s.action for s in plan])

        # Surface plan-quality concerns: weak predicates AND risky patterns
        # (sudo, runas, etc.). Warnings only — the run still proceeds. The
        # user sees these in the audit log AND in the final terminal output
        # via TaskOutcome.state.plan_warnings.
        for w in weak_predicate_warnings(plan):
            state.plan_warnings.append(w)
            self.episodic.append_plan_warning(self.run_id, w)
        for w in plan_risk_warnings(plan):
            state.plan_warnings.append(w)
            self.episodic.append_plan_warning(self.run_id, w)

    def _maybe_snapshot(self, state: TaskState, step: Step) -> None:
        if self.snapshotter is None:
            return
        snap = self.snapshotter.make_snapshot(self.run_id)
        if step.action in ("file.write", "file.delete"):
            path = step.args.get("path")
            if isinstance(path, str):
                snap.entries.append(self.snapshotter.capture_file(path))
        elif step.action == "process.kill":
            pid = step.args.get("pid")
            if isinstance(pid, int):
                snap.entries.append(self.snapshotter.capture_process(pid))
        # Only persist if we captured something
        if not snap.entries:
            return
        try:
            path = self.snapshotter.write(snap)
        except OSError:
            return  # best-effort; failing to snapshot must not block the run
        register_snapshot(state, path)
        self.episodic.append_snapshot(self.run_id, "pre-step", str(path))

    def _maybe_confirm(self, state: TaskState, step: Step) -> str | bool | None:
        """Returns the granted token (str), False on denial, None if no
        confirmation was required."""
        if step.action not in REQUIRES_CONFIRMATION:
            return None
        resp = self.confirmation_provider.request(
            ConfirmationRequest(step=step, rationale=step.expected_state),
        )
        self.episodic.append_confirmation(
            self.run_id, step.action, resp.token or "", resp.granted,
        )
        if not resp.granted:
            return False
        # Grant token to the executor's guard for the *specific* action
        self.executor.guard.grant_confirmation(resp.token, step.action)
        return resp.token

    def _execute_and_verify(
        self, step: Step, token: str | bool | None,
    ) -> tuple[StepRecord, ToolResult, Verification]:
        confirmation_token = token if isinstance(token, str) else None
        started = time.time()
        try:
            result = self.executor.execute(step, confirmation_token=confirmation_token)
        except GuardError as e:
            result = ToolResult(success=False, error=f"GuardError: {e}")
        except Exception as e:
            result = ToolResult(success=False, error=f"executor raised {type(e).__name__}: {e}")
        ended = time.time()

        verification = self.verifier.verify(step, result)

        record = StepRecord(
            step=step,
            started_at=started,
            ended_at=ended,
            success=verification.success,
            output=result.output,
            verification={
                "success": verification.success,
                "reason": verification.reason,
                "needs_retry": verification.needs_retry,
                "layer": verification.layer,
            },
            error=result.error,
        )
        return record, result, verification

    def _apply_repair(
        self,
        state: TaskState,
        step: Step,
        decision: RepairDecision,
        diagnosis: Diagnosis,
        step_attempts: defaultdict[int, int],
    ) -> TaskOutcome | None:
        if decision.strategy == "retry":
            return None  # loop again on same step
        if decision.strategy == "fix_input" and decision.step is not None:
            state.plan[state.current_step] = decision.step
            # Re-run risk warnings on the rewritten step. fix_input is the
            # exact path that previously sneaked sudo past the plan-time
            # check — surface any newly-introduced risk so the user sees
            # it before approving at the confirmation prompt.
            for w in plan_risk_warnings([decision.step]):
                state.plan_warnings.append(w)
                self.episodic.append_plan_warning(self.run_id, w)
            return None
        if decision.strategy == "replan":
            try:
                new_plan = self.planner.replan(
                    state.task, step, diagnosis.to_dict(),
                    goal_class=state.goal_class,
                )
            except PlanError as e:
                return self._escalate(
                    state, EscalationReason.UNRECOVERABLE_ERROR,
                    summary=f"replan failed: {e}",
                )
            self._adopt_plan(state, new_plan)
            step_attempts.clear()
            return None
        # abort, fix_input-without-step, or any unknown strategy
        return self._escalate(
            state, EscalationReason.ABORT_REQUESTED,
            summary=f"repair aborted: {decision.reason or decision.strategy}",
        )

    def _escalate(
        self,
        state: TaskState,
        reason: EscalationReason,
        *,
        summary: str | None = None,
    ) -> TaskOutcome:
        report = build_report(state, reason, summary=summary)
        self.episodic.append_escalation(
            self.run_id, reason.value, report.attempts, len(report.strategies_tried),
        )
        self.episodic.append_task_end(self.run_id, "escalated", summary=summary)
        paths: tuple[Path, Path] | None = None
        if self.reports_dir is not None:
            try:
                paths = write_report(report, self.reports_dir)
            except OSError:
                paths = None  # never block the outcome on a disk error
        return TaskOutcome(
            run_id=self.run_id,
            status="escalated",
            state=state,
            failure_report=report,
            report_paths=paths,
        )
