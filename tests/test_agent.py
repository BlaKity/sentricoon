"""End-to-end agent-loop tests.

Each test wires the real classes together (executor + safety + tool router
+ verifier + episodic log + snapshotter) with mocked LLM backends for the
planner / diagnoser / repair roles. Tools are real for system.info and
file.{read,write,delete}; mock for anything we want to script.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sentricoon.agent import Agent, TaskOutcome
from sentricoon.confirmation import AlwaysDenyProvider, AutoApproveProvider
from sentricoon.diagnoser import Diagnoser
from sentricoon.escalation import EscalationReason, TaskBudget
from sentricoon.executor import Executor
from sentricoon.llm import LLMRouter, MockBackend, RouterConfig
from sentricoon.memory import EpisodicLog
from sentricoon.planner import Planner
from sentricoon.repair import Repairer
from sentricoon.router import ToolRouter
from sentricoon.safety.guard import SafetyGuard
from sentricoon.snapshot import Snapshotter
from sentricoon.tools.base import Tool, ToolResult
from sentricoon.tools.file_ops import FileReadTool, FileWriteTool
from sentricoon.tools.system import SystemInfoTool
from sentricoon.verifier import HardVerifier, Verifier


# ============================================================================
# Test fixture: scriptable cloud + local backends, plus a real tool set.
# ============================================================================


def _plan_json(steps: list[dict]) -> str:
    """Wrap a list of step dicts as a valid planner response.

    Default expected_state is intentionally long + concrete so it doesn't
    trip the weak-predicate heuristic; tests that want a weak predicate
    pass it explicitly.
    """
    full_steps = []
    for s in steps:
        full_steps.append({
            "action": s["action"],
            "args": s.get("args", {}),
            "expected_state": s.get(
                "expected_state",
                "the step's stated postcondition holds with returncode == 0",
            ),
            "risk_level": s.get("risk_level", "low"),
        })
    return json.dumps({"steps": full_steps})


def _diagnosis_json(strategy: str = "abort", **kw) -> str:
    return json.dumps({
        "failure_type": kw.get("failure_type", "tool_error"),
        "root_cause": kw.get("root_cause", "test diagnosis"),
        "fix_strategy": strategy,
    })


class _AlwaysFailTool(Tool):
    name = "system.info"  # piggyback on an allowlisted name
    risk_level = "low"
    is_read_only = True

    def __init__(self, n_failures: int = 999, then_succeed: bool = False) -> None:
        self.calls = 0
        self.n_failures = n_failures
        self.then_succeed = then_succeed

    def run(self, args):
        self.calls += 1
        if self.calls > self.n_failures and self.then_succeed:
            return ToolResult(success=True, output={"platform": "linux", "system": "test"})
        return ToolResult(success=False, error="permission denied")


class _Harness:
    """Builds a fully-wired Agent with the requested mock scripts.

    For tests with a single plan + diagnoses, use plan_scripts + diag_scripts —
    the harness concatenates them. For tests where the cloud LLM is called
    in a non-trivial order (create_plan → diagnose → replan), pass
    cloud_script directly with the exact sequence.
    """

    def __init__(
        self,
        *,
        plan_scripts: list[str] | None = None,
        diag_scripts: list[str] | None = None,
        cloud_script: list[str] | None = None,
        repair_scripts: list[str] | None = None,
        tool: Tool | None = None,
        snapshot_dir: Path | None = None,
        reports_dir: Path | None = None,
        confirmation=None,
        budget: TaskBudget | None = None,
    ) -> None:
        self.tmp = tempfile.mkdtemp()
        if cloud_script is None:
            cloud_script = list(plan_scripts or []) + list(diag_scripts or [])
        self.cloud_script = list(cloud_script)
        cloud = MockBackend(script=self.cloud_script, name="cloud", model="cloud-1")
        local = MockBackend(script=list(repair_scripts or []), name="local", model="local-1")
        self.router_llm = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
        self.router_llm.register(cloud)
        self.router_llm.register(local)
        self.cloud = cloud
        self.local = local

        # Real tool layer
        self.tool_router = ToolRouter()
        if tool is None:
            tool = SystemInfoTool()
        self.tool_router.register(tool)
        self.tool = tool

        self.guard = SafetyGuard()
        self.executor = Executor(self.tool_router, self.guard)

        self.episodic_path = Path(self.tmp) / "run.jsonl"
        self.episodic = EpisodicLog(self.episodic_path)

        snapshotter = Snapshotter(snapshot_dir) if snapshot_dir else None

        self.agent = Agent(
            run_id="run-test",
            planner=Planner(self.router_llm),
            executor=self.executor,
            verifier=Verifier.hard_only(),
            diagnoser=Diagnoser(self.router_llm),
            repairer=Repairer(self.router_llm),
            episodic=self.episodic,
            snapshotter=snapshotter,
            confirmation_provider=confirmation,
            budget=budget,
            reports_dir=reports_dir,
        )


# ============================================================================
# Happy path
# ============================================================================


class TestHappyPath(unittest.TestCase):
    def test_single_step_plan_succeeds(self):
        h = _Harness(plan_scripts=[
            _plan_json([{"action": "system.info", "expected_state": "platform present"}]),
        ])
        outcome = h.agent.run("inspect system", goal_class="audio_no_sound")
        self.assertEqual(outcome.status, "done")
        self.assertEqual(outcome.run_id, "run-test")
        self.assertEqual(len(outcome.state.history), 1)
        self.assertTrue(outcome.state.history[0].success)

    def test_two_step_plan_succeeds(self):
        h = _Harness(plan_scripts=[
            _plan_json([
                {"action": "system.info", "expected_state": "platform present"},
                {"action": "system.info", "expected_state": "platform still present"},
            ]),
        ])
        outcome = h.agent.run("inspect twice")
        self.assertEqual(outcome.status, "done")
        self.assertEqual(len(outcome.state.history), 2)

    def test_episodic_log_records_full_lifecycle(self):
        h = _Harness(plan_scripts=[
            _plan_json([{"action": "system.info"}]),  # strong default predicate
        ])
        h.agent.run("x")
        types = [e["type"] for e in h.episodic.read_all()]
        # task_start, plan, strategy, step, task_end — no plan_warnings
        # because the default expected_state is strong.
        self.assertEqual(types, ["task_start", "plan", "strategy", "step", "task_end"])


# ============================================================================
# Retry path — diagnoser says retry, second attempt succeeds
# ============================================================================


class TestRetryPath(unittest.TestCase):
    def test_retry_succeeds_on_second_attempt(self):
        # First call fails, second succeeds
        flaky = _AlwaysFailTool(n_failures=1, then_succeed=True)
        h = _Harness(
            plan_scripts=[_plan_json([{"action": "system.info", "expected_state": "x"}])],
            diag_scripts=[_diagnosis_json("retry")],
            tool=flaky,
        )
        outcome = h.agent.run("x")
        self.assertEqual(outcome.status, "done")
        self.assertEqual(flaky.calls, 2)


# ============================================================================
# Abort path
# ============================================================================


class TestAbortPath(unittest.TestCase):
    def test_diagnosis_abort_escalates(self):
        h = _Harness(
            plan_scripts=[_plan_json([{"action": "system.info", "expected_state": "x"}])],
            diag_scripts=[_diagnosis_json("abort", root_cause="unrecoverable")],
            tool=_AlwaysFailTool(),
        )
        outcome = h.agent.run("x")
        self.assertEqual(outcome.status, "escalated")
        self.assertEqual(outcome.failure_report.reason, EscalationReason.ABORT_REQUESTED)
        self.assertIn("unrecoverable", outcome.failure_report.summary)


# ============================================================================
# Replan path
# ============================================================================


class TestReplanPath(unittest.TestCase):
    def test_replan_produces_new_plan(self):
        # The cloud is called in this order:
        #   1) planner.create_plan        → plan1
        #   2) diagnoser.diagnose          → "replan"
        #   3) planner.replan              → plan2
        h = _Harness(
            cloud_script=[
                _plan_json([{"action": "system.info", "expected_state": "x"}]),
                _diagnosis_json("replan"),
                _plan_json([{"action": "system.info", "expected_state": "x"}]),
            ],
            tool=_AlwaysFailTool(n_failures=1, then_succeed=True),
        )
        outcome = h.agent.run("x")
        self.assertEqual(outcome.status, "done")
        # Episodic log shows TWO plans emitted
        plans = [e for e in h.episodic.read_all() if e["type"] == "plan"]
        self.assertEqual(len(plans), 2)
        strategies = [e for e in h.episodic.read_all() if e["type"] == "strategy"]
        self.assertEqual(len(strategies), 2)


# ============================================================================
# Budget exhaustion via retry budget on a step
# ============================================================================


class TestRetryBudget(unittest.TestCase):
    def test_three_retries_then_abort(self):
        # All attempts fail; diagnoser keeps saying retry; budget kicks in
        h = _Harness(
            plan_scripts=[_plan_json([{"action": "system.info", "expected_state": "x"}])],
            # Three diagnosis responses for three failed attempts
            diag_scripts=[_diagnosis_json("retry")] * 3,
            tool=_AlwaysFailTool(),
        )
        outcome = h.agent.run("x")
        self.assertEqual(outcome.status, "escalated")
        # The 3rd repair call returns abort because attempts_so_far==3 hits max
        self.assertEqual(h.tool.calls, 3)


# ============================================================================
# Initial plan failure
# ============================================================================


class TestInitialPlanFailure(unittest.TestCase):
    def test_invalid_plan_response_escalates_cleanly(self):
        h = _Harness(plan_scripts=["definitely not json"])
        outcome = h.agent.run("x")
        self.assertEqual(outcome.status, "escalated")
        self.assertEqual(outcome.failure_report.reason, EscalationReason.UNRECOVERABLE_ERROR)
        self.assertIn("plan failed", outcome.failure_report.summary)


# ============================================================================
# High-risk confirmation flow
# ============================================================================


class _NoOpHighRisk(Tool):
    """Fake high-risk shell.run tool. hard_only verifier passes it via
    returncode=0 — keeping confirmation tests focused on the gate, not
    the verifier layers."""

    name = "shell.run"
    risk_level = "high"
    is_read_only = False

    def __init__(self):
        self.calls = []

    def run(self, args):
        self.calls.append(args)
        return ToolResult(success=True, output={"stdout": "", "stderr": ""},
                          extra={"returncode": 0})


_HIGH_RISK_STEP = {
    "action": "shell.run", "args": {"argv": ["net", "start", "Audiosrv"]},
    "expected_state": "service Audiosrv RUNNING", "risk_level": "high",
}


class TestConfirmationFlow(unittest.TestCase):
    def test_denied_confirmation_escalates(self):
        h = _Harness(
            plan_scripts=[_plan_json([_HIGH_RISK_STEP])],
            tool=_NoOpHighRisk(),
            confirmation=AlwaysDenyProvider(),
        )
        outcome = h.agent.run("restart audio")
        self.assertEqual(outcome.status, "escalated")
        self.assertEqual(outcome.failure_report.reason, EscalationReason.ABORT_REQUESTED)
        self.assertEqual(len(h.tool.calls), 0)  # tool never invoked

    def test_approved_confirmation_runs_step(self):
        h = _Harness(
            plan_scripts=[_plan_json([_HIGH_RISK_STEP])],
            tool=_NoOpHighRisk(),
            confirmation=AutoApproveProvider(),
        )
        outcome = h.agent.run("restart audio")
        self.assertEqual(outcome.status, "done")
        self.assertEqual(len(h.tool.calls), 1)

    def test_default_provider_denies(self):
        """No provider configured = AlwaysDeny. High-risk step fails by default."""
        h = _Harness(
            plan_scripts=[_plan_json([_HIGH_RISK_STEP])],
            tool=_NoOpHighRisk(),
            confirmation=None,  # falls back to AlwaysDeny inside Agent
        )
        outcome = h.agent.run("restart audio")
        self.assertEqual(outcome.status, "escalated")
        self.assertEqual(len(h.tool.calls), 0)


# ============================================================================
# Snapshots
# ============================================================================


class TestSnapshotIntegration(unittest.TestCase):
    def test_pre_step_snapshot_for_file_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.txt"
            target.write_text("pre-content")

            snap_dir = Path(tmp) / "snaps"
            h = _Harness(
                plan_scripts=[_plan_json([{
                    "action": "file.write",
                    "args": {"path": str(target), "content": "post-content"},
                    "expected_state": f"{target} contains post-content",
                    "risk_level": "medium",
                }])],
                tool=FileWriteTool(),
                snapshot_dir=snap_dir,
            )
            outcome = h.agent.run("update config")
            self.assertEqual(outcome.status, "done")
            # A snapshot was registered on the state
            self.assertEqual(len(outcome.state.snapshots), 1)
            # The snapshot file actually exists on disk
            snap_path = Path(outcome.state.snapshots[0])
            self.assertTrue(snap_path.exists())
            # And captured the PRE state, not the post state
            payload = json.loads(snap_path.read_text())
            entry = payload["entries"][0]
            self.assertEqual(entry["payload"]["content"], "pre-content")
            # The file is now post-content (the agent's write succeeded)
            self.assertEqual(target.read_text(), "post-content")


# ============================================================================
# Failure report written to disk
# ============================================================================


class TestFailureReportWritten(unittest.TestCase):
    def test_report_files_created_on_escalation(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp) / "failures"
            h = _Harness(
                plan_scripts=["totally bogus"],
                reports_dir=reports_dir,
            )
            outcome = h.agent.run("x", goal_class="audio_no_sound")
            self.assertEqual(outcome.status, "escalated")
            self.assertIsNotNone(outcome.report_paths)
            md, js = outcome.report_paths
            self.assertTrue(md.exists())
            self.assertTrue(js.exists())
            parsed = json.loads(js.read_text())
            self.assertEqual(parsed["reason"], "unrecoverable_error")


# ============================================================================
# Contract: Agent.run never raises
# ============================================================================


class TestAgentNeverRaises(unittest.TestCase):
    def test_planner_blowup_does_not_raise(self):
        # Planner script empty → backend exhausted → PlanError → caught
        h = _Harness(plan_scripts=[])
        outcome = h.agent.run("x")
        self.assertEqual(outcome.status, "escalated")


class TestPlanWarnings(unittest.TestCase):
    def test_weak_predicates_logged_to_episodic_and_state(self):
        # Plan with a weak predicate — the run still succeeds, but warnings
        # are surfaced on TaskState and in the episodic log.
        h = _Harness(plan_scripts=[
            _plan_json([{
                "action": "system.info",
                "expected_state": "OS information is retrieved",  # weak
            }]),
        ])
        outcome = h.agent.run("x")
        self.assertEqual(outcome.status, "done")
        # TaskState has the warning
        self.assertEqual(len(outcome.state.plan_warnings), 1)
        self.assertIn("retrieved", outcome.state.plan_warnings[0])
        # Episodic log too
        warnings = [e for e in h.episodic.read_all() if e["type"] == "plan_warning"]
        self.assertEqual(len(warnings), 1)

    def test_strong_predicates_produce_no_warnings(self):
        h = _Harness(plan_scripts=[
            _plan_json([{
                "action": "system.info",
                "expected_state": "system.info output has key 'system' equal to 'Linux'",
            }]),
        ])
        outcome = h.agent.run("x")
        self.assertEqual(outcome.state.plan_warnings, [])

    def test_sudo_in_plan_surfaces_risk_warning(self):
        h = _Harness(plan_scripts=[
            _plan_json([{
                "action": "shell.run",
                "args": {"argv": ["sudo", "find", "/tmp", "-size", "+50M"]},
            }]),
        ])
        # Plan parses, but agent will block on confirmation for shell.run.
        # We don't care if the run succeeds — we care that the warning landed.
        h.agent.run("x")
        self.assertTrue(
            any("privilege escalation" in w for w in h.episodic.read_events("plan_warning")[0]["payload"]["message"]
                if isinstance(w, str)) or
            any("privilege escalation" in e["payload"]["message"]
                for e in h.episodic.read_events("plan_warning")),
        )


if __name__ == "__main__":
    unittest.main()
