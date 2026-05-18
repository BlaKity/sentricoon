"""Repair engine — maps a Diagnosis to a RepairDecision.

Strategies:
  - retry:     re-execute the same step (transient failures only)
  - fix_input: call LLMRole.INPUT_REPAIR (local backend) to rewrite the args
  - replan:    return up to the agent loop, which invokes Planner.replan
  - abort:     give up on this step; agent loop kicks to escalation

Per-step retry budget is enforced HERE: if `attempts_so_far` has hit
`max_attempts_per_step`, the Repairer forces abort regardless of what
the diagnoser recommended. That stops the diagnoser → repair → retry
loop from spinning indefinitely on a step that just won't recover.

fix_input MUST NOT change the action — that's replan's job. We validate
the action stays the same when applying the rewritten args.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .diagnoser import Diagnosis
from .llm.role import LLMRole
from .llm.router import LLMRouter
from .llm.types import LLMRequest, Message
from .state import Step


@dataclass
class RepairDecision:
    strategy: str  # "retry" | "fix_input" | "replan" | "abort"
    step: Step | None = None  # populated when strategy == "fix_input"
    reason: str | None = None


_INPUT_REPAIR_SYSTEM_PROMPT = """You rewrite the ARGUMENTS for a failed agent step.

You did NOT plan this step. You are fixing the wrong inputs while keeping
the same action and the same intent.

Rules:
- DO NOT change the action.
- DO NOT change expected_state.
- Only fix the args.
- RESPECT USER CONSTRAINTS from the original task. Phrases like
  "do not use sudo", "do not modify X", "read-only" are HARD constraints.
  If the only fix would violate one, return args identical to the failing
  step (no productive fix possible — caller will abort).

Respond ONLY with a JSON object:
{"args": {...}}"""


class Repairer:
    def __init__(
        self,
        router: LLMRouter | None = None,
        *,
        max_attempts_per_step: int = 3,
    ) -> None:
        self.router = router
        self.max_attempts_per_step = max_attempts_per_step

    def repair(
        self,
        step: Step,
        diagnosis: Diagnosis,
        attempts_so_far: int,
        *,
        task: str | None = None,
    ) -> RepairDecision:
        if attempts_so_far >= self.max_attempts_per_step:
            return RepairDecision(
                strategy="abort",
                reason=f"step retry budget exhausted ({self.max_attempts_per_step})",
            )

        strategy = diagnosis.fix_strategy

        if strategy == "abort":
            return RepairDecision(strategy="abort", reason=diagnosis.root_cause)

        if strategy == "retry":
            return RepairDecision(strategy="retry", reason=diagnosis.root_cause)

        if strategy == "replan":
            return RepairDecision(strategy="replan", reason=diagnosis.root_cause)

        if strategy == "fix_input":
            return self._fix_input(step, diagnosis, task=task)

        return RepairDecision(
            strategy="abort",
            reason=f"unknown fix_strategy: {strategy!r}",
        )

    # ----- input repair via local LLM -----

    def _fix_input(self, step: Step, diagnosis: Diagnosis, *, task: str | None = None) -> RepairDecision:
        if self.router is None:
            return RepairDecision(
                strategy="abort",
                reason="fix_input requested but Repairer has no LLM router",
            )

        task_block = (
            f"Original user task (HARD constraints — must not violate):\n  {task}\n\n"
            if task else ""
        )
        user_prompt = (
            f"{task_block}"
            f"Failed step:\n"
            f"  action: {step.action}\n"
            f"  args (current): {json.dumps(step.args, default=str)}\n"
            f"  expected_state: {step.expected_state}\n\n"
            f"Diagnosis:\n"
            f"  failure_type: {diagnosis.failure_type}\n"
            f"  root_cause: {diagnosis.root_cause}\n"
            f"  evidence: {diagnosis.evidence}\n\n"
            "Return ONLY the corrected args."
        )

        req = LLMRequest(
            messages=[
                Message(role="system", content=_INPUT_REPAIR_SYSTEM_PROMPT),
                Message(role="user", content=user_prompt),
            ],
            temperature=0.1,
            json_mode=True,
        )

        try:
            resp = self.router.complete(LLMRole.INPUT_REPAIR, req)
        except Exception as e:
            return RepairDecision(
                strategy="abort",
                reason=f"input repair backend unreachable: {e}",
            )

        try:
            parsed = json.loads(resp.content)
        except (json.JSONDecodeError, TypeError):
            return RepairDecision(
                strategy="abort",
                reason=f"input repair returned non-JSON: {resp.content[:200]!r}",
            )

        if not isinstance(parsed, dict):
            return RepairDecision(
                strategy="abort",
                reason="input repair returned non-object JSON",
            )

        new_args = parsed.get("args")
        if not isinstance(new_args, dict):
            return RepairDecision(
                strategy="abort",
                reason="input repair response missing or invalid 'args' object",
            )

        # Sanity check: refuse to silently change the action.
        if "action" in parsed and parsed["action"] != step.action:
            return RepairDecision(
                strategy="abort",
                reason=(
                    f"input repair tried to change action from {step.action!r} to "
                    f"{parsed['action']!r} — that's replan's job, not fix_input"
                ),
            )

        new_step = Step(
            action=step.action,
            args=new_args,
            expected_state=step.expected_state,
            risk_level=step.risk_level,
        )
        return RepairDecision(
            strategy="fix_input",
            step=new_step,
            reason=diagnosis.root_cause,
        )
