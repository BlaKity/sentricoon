"""Diagnoser — explains why a step failed and recommends a fix strategy.

Routes via LLMRole.DIAGNOSER (cloud at stage 1, different model family at
stage 3 per [[plan-llm-use]]). System prompt explicitly tells the model
"you did NOT plan this step" so the diagnoser audits rather than re-plans.
This is the correlated-failures mitigation from the review.

Return type is a typed `Diagnosis` dataclass. The agent stores
`asdict(diagnosis)` in `StepRecord.diagnosis` so it round-trips through
the episodic JSONL log as plain JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .llm.role import LLMRole
from .llm.router import LLMRouter
from .llm.types import LLMRequest, Message
from .state import Step
from .tools.base import ToolResult
from .verifier import Verification

FAILURE_TYPES = {"tool_error", "bad_input", "env_changed", "unknown"}
FIX_STRATEGIES = {"retry", "fix_input", "replan", "abort"}


@dataclass
class Diagnosis:
    failure_type: str  # one of FAILURE_TYPES
    root_cause: str
    fix_strategy: str  # one of FIX_STRATEGIES
    ruled_out: str | None = None
    evidence: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM_PROMPT = """You analyze why a single agent step failed.

You did NOT plan this step. You are auditing another AI's work. Your job
is to classify the failure honestly and recommend a fix strategy.

Respond ONLY with a JSON object:
{
  "failure_type": "tool_error" | "bad_input" | "env_changed" | "unknown",
  "root_cause": "<one short sentence>",
  "fix_strategy": "retry" | "fix_input" | "replan" | "abort",
  "ruled_out": "<optional: what causes are eliminated, or null>",
  "evidence": "<optional: which observation supports your diagnosis, or null>"
}

Fix strategy guidance:
  - "retry":     transient failure (timeout, resource busy, brief network) — re-run as-is
  - "fix_input": step intent is right but args are wrong (e.g., wrong path, wrong service name)
  - "replan":    fundamental approach is wrong (e.g., service doesn't exist on this OS)
  - "abort":     destructive risk, repeated unrecoverable failure, or evidence is contradictory

USER CONSTRAINT HANDLING (critical):
The user's original task may contain explicit constraints — phrases like
"do not use sudo", "do not modify X", "do not change the default device",
"don't delete anything", "read-only", etc. YOU MUST RESPECT THESE.

If the only fix would violate a user constraint, recommend fix_strategy=
"abort" and put the constraint violation in root_cause. Permission-denied
errors with a "no sudo" constraint mean partial results are the answer,
NOT a retry with elevation.

Be conservative. If you can't tell, return failure_type=unknown and
fix_strategy=abort."""


class Diagnoser:
    _MAX_OUTPUT_CHARS = 1500

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    def diagnose(
        self,
        step: Step,
        result: ToolResult,
        verification: Verification,
        *,
        task: str | None = None,
    ) -> Diagnosis:
        output_str = json.dumps(result.output, default=str)[: self._MAX_OUTPUT_CHARS]
        task_block = (
            f"Original user task (respect any constraints stated here):\n  {task}\n\n"
            if task else ""
        )
        user_prompt = (
            f"{task_block}"
            "Failed step:\n"
            f"  action: {step.action}\n"
            f"  args: {json.dumps(step.args, default=str)}\n"
            f"  expected_state: {step.expected_state}\n"
            f"  risk_level: {step.risk_level}\n\n"
            "Tool result:\n"
            f"  success: {result.success}\n"
            f"  output: {output_str}\n"
            f"  error: {result.error}\n\n"
            "Verification verdict:\n"
            f"  success: {verification.success}\n"
            f"  reason: {verification.reason}\n"
            f"  layer: {verification.layer}\n"
            f"  needs_retry: {verification.needs_retry}\n\n"
            "Diagnose. Return JSON only."
        )

        req = LLMRequest(
            messages=[
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=user_prompt),
            ],
            temperature=0.1,
            json_mode=True,
        )

        try:
            resp = self.router.complete(LLMRole.DIAGNOSER, req)
        except Exception as e:
            return Diagnosis(
                failure_type="unknown",
                root_cause=f"diagnoser unreachable: {e}",
                fix_strategy="abort",
            )

        try:
            parsed = json.loads(resp.content)
        except (json.JSONDecodeError, TypeError):
            return Diagnosis(
                failure_type="unknown",
                root_cause=f"diagnoser returned non-JSON: {resp.content[:200]!r}",
                fix_strategy="abort",
            )

        if not isinstance(parsed, dict):
            return Diagnosis(
                failure_type="unknown",
                root_cause="diagnoser returned non-object JSON",
                fix_strategy="abort",
            )

        # Coerce enums conservatively — unknown values become "unknown"/"abort"
        failure_type = parsed.get("failure_type")
        if failure_type not in FAILURE_TYPES:
            failure_type = "unknown"

        fix_strategy = parsed.get("fix_strategy")
        if fix_strategy not in FIX_STRATEGIES:
            fix_strategy = "abort"

        return Diagnosis(
            failure_type=failure_type,
            root_cause=str(parsed.get("root_cause") or "no root cause given"),
            fix_strategy=fix_strategy,
            ruled_out=_maybe_str(parsed.get("ruled_out")),
            evidence=_maybe_str(parsed.get("evidence")),
        )


def _maybe_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
