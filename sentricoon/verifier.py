"""Verifier — three-layer stack: HARD → OBSERVATION → SEMANTIC.

Each layer can return a Verification (definitive) or None (inconclusive,
escalate to the next layer). Cheapest layer runs first.

  - HardVerifier:      examine ToolResult only. No additional I/O.
  - ObservationVerifier: re-read system state via the executor.
  - SemanticVerifier:  ask the LLM (LLMRole.VERIFIER → local backend).

Each layer tags its verdict with `layer=` for forensics: when the agent
fails, knowing which layer signed off (or didn't) matters for debugging.

See [[project-goals-and-structure]] § "Verification types" and
[[plan-local-cloud-hybrid]] for the local-verifier routing rationale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .llm.role import LLMRole
from .llm.types import LLMRequest, Message
from .router import RouterError
from .state import Step
from .tools.base import ToolResult

if TYPE_CHECKING:
    from .executor import Executor
    from .llm.router import LLMRouter


@dataclass
class Verification:
    success: bool
    reason: str
    needs_retry: bool = False
    layer: str = ""


class HardVerifier:
    """Layer 1 — examines ToolResult only. No additional I/O.

    Returns None when the action has no hard rule (e.g., process.kill —
    can't know without re-reading the process list; defers to observation).
    """

    # Hint words that suggest the failure may be transient and worth a retry.
    _TRANSIENT_HINTS = ("timeout", "temporarily", "in use", "resource busy", "try again")

    def verify(self, step: Step, result: ToolResult) -> Verification | None:
        # Dry-run short-circuit: a simulated result has no real side effect
        # to verify. We pass it, but the reason field documents that no
        # real verification was performed — surfaced in the audit log so
        # readers can tell a dry-run pass from a real verified pass.
        if result.extra and result.extra.get("simulated"):
            return Verification(
                success=True,
                reason="dry-run simulated; no real verification performed",
                layer="hard",
            )

        if not result.success:
            err_lower = (result.error or "").lower()
            transient = any(h in err_lower for h in self._TRANSIENT_HINTS)
            return Verification(
                success=False,
                reason=result.error or "tool reported failure",
                needs_retry=transient,
                layer="hard",
            )

        action = step.action

        if action == "shell.run":
            rc = (result.extra or {}).get("returncode")
            if rc == 0:
                return Verification(success=True, reason="exit code 0", layer="hard")
            return Verification(
                success=False,
                reason=f"non-zero exit code: {rc}",
                needs_retry=False,
                layer="hard",
            )

        if action in {"system.info", "process.list", "file.read"}:
            return Verification(success=True, reason="read-only tool succeeded", layer="hard")

        if action == "file.write":
            return Verification(success=True, reason="write reported success", layer="hard")

        if action == "file.delete":
            return Verification(success=True, reason="delete reported success", layer="hard")

        if action == "process.kill":
            # Tool success means signal was sent, not that process is gone.
            # Defer to observation.
            return None

        # Unknown action — let observation/semantic try.
        return None


class ObservationVerifier:
    """Layer 2 — re-reads system state via the executor.

    Runs read-only actions (file.read, process.list) through the same
    executor the agent uses. The safety guard lets these through without
    confirmation tokens because they don't appear in REQUIRES_CONFIRMATION.
    """

    def __init__(self, executor: "Executor") -> None:
        self.executor = executor

    def verify(self, step: Step, result: ToolResult) -> Verification | None:
        action = step.action

        if action == "file.write":
            return self._verify_file_write(step)
        if action == "file.delete":
            return self._verify_file_delete(step)
        if action == "process.kill":
            return self._verify_process_kill(step)

        return None

    def _verify_file_write(self, step: Step) -> Verification | None:
        path = step.args.get("path")
        content = step.args.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            return None
        try:
            p = Path(path)
            if not p.exists():
                return Verification(
                    success=False,
                    reason=f"file not present after write: {path}",
                    needs_retry=False,
                    layer="observation",
                )
            actual = p.read_text(encoding=step.args.get("encoding", "utf-8"))
            if actual == content:
                return Verification(
                    success=True,
                    reason=f"file content matches ({len(content)} bytes)",
                    layer="observation",
                )
            return Verification(
                success=False,
                reason=f"file present but content differs (got {len(actual)} bytes, expected {len(content)})",
                needs_retry=False,
                layer="observation",
            )
        except OSError as e:
            return Verification(
                success=False,
                reason=f"observation read failed: {e}",
                needs_retry=True,
                layer="observation",
            )

    def _verify_file_delete(self, step: Step) -> Verification | None:
        path = step.args.get("path")
        if not isinstance(path, str):
            return None
        if Path(path).exists():
            return Verification(
                success=False,
                reason=f"file still present after delete: {path}",
                needs_retry=False,
                layer="observation",
            )
        return Verification(success=True, reason="file no longer present", layer="observation")

    def _verify_process_kill(self, step: Step) -> Verification | None:
        pid = step.args.get("pid")
        if not isinstance(pid, int):
            return None
        try:
            listing = self.executor.execute(Step(action="process.list", args={}))
        except (RouterError, Exception):
            return None  # can't observe — defer to semantic
        if not listing.success:
            return None
        procs = listing.output if isinstance(listing.output, list) else []
        still_running = any(isinstance(p, dict) and p.get("pid") == pid for p in procs)
        if still_running:
            return Verification(
                success=False,
                reason=f"pid {pid} still present in process list",
                needs_retry=True,
                layer="observation",
            )
        return Verification(
            success=True,
            reason=f"pid {pid} no longer in process list",
            layer="observation",
        )


class SemanticVerifier:
    """Layer 3 — asks the LLM whether the step achieved its expected_state.

    Routes via LLMRole.VERIFIER which goes to the LOCAL backend per
    [[plan-local-cloud-hybrid]]. This is the high-frequency LLM role —
    keeping it local is the bulk of the cost savings.
    """

    SYSTEM_PROMPT = (
        "You verify whether a single agent step achieved its stated intent. "
        "You did NOT plan this step — you are auditing another AI's work. "
        "Respond ONLY with a JSON object: "
        '{"success": true|false, "reason": "<short explanation>"}. '
        "Be conservative — if evidence is ambiguous, answer success=false."
    )

    _MAX_OUTPUT_CHARS = 1500

    def __init__(self, router: "LLMRouter") -> None:
        self.router = router

    def verify(self, step: Step, result: ToolResult) -> Verification:
        if not step.expected_state:
            return Verification(
                success=False,
                reason="semantic verifier: step has no expected_state to verify against",
                needs_retry=False,
                layer="semantic",
            )

        output_str = json.dumps(result.output, default=str)[: self._MAX_OUTPUT_CHARS]
        user_prompt = (
            f"Action: {step.action}\n"
            f"Args: {json.dumps(step.args, default=str)}\n"
            f"Expected state: {step.expected_state}\n"
            f"Tool success: {result.success}\n"
            f"Tool output: {output_str}\n"
            f"Tool error: {result.error}\n\n"
            "Did this step achieve the expected state? Return JSON only."
        )

        req = LLMRequest(
            messages=[
                Message(role="system", content=self.SYSTEM_PROMPT),
                Message(role="user", content=user_prompt),
            ],
            temperature=0.0,
            json_mode=True,
        )

        try:
            resp = self.router.complete(LLMRole.VERIFIER, req)
        except Exception as e:
            return Verification(
                success=False,
                reason=f"semantic verifier unreachable: {e}",
                needs_retry=True,
                layer="semantic",
            )

        try:
            parsed = json.loads(resp.content)
        except (json.JSONDecodeError, TypeError):
            return Verification(
                success=False,
                reason=f"semantic verifier returned non-JSON: {resp.content[:200]!r}",
                needs_retry=False,
                layer="semantic",
            )

        if not isinstance(parsed, dict):
            return Verification(
                success=False,
                reason="semantic verifier returned non-object JSON",
                needs_retry=False,
                layer="semantic",
            )

        success = bool(parsed.get("success", False))
        reason = str(parsed.get("reason") or "no reason given")
        return Verification(success=success, reason=reason, needs_retry=False, layer="semantic")


class Verifier:
    """Orchestrates the three layers cheapest-first."""

    def __init__(
        self,
        hard: HardVerifier,
        observation: ObservationVerifier | None = None,
        semantic: SemanticVerifier | None = None,
    ) -> None:
        self.hard = hard
        self.observation = observation
        self.semantic = semantic

    def verify(self, step: Step, result: ToolResult) -> Verification:
        v = self.hard.verify(step, result)
        if v is not None:
            return v

        if self.observation is not None:
            v = self.observation.verify(step, result)
            if v is not None:
                return v

        if self.semantic is not None:
            return self.semantic.verify(step, result)

        return Verification(
            success=False,
            reason="no verifier layer produced a result (no observation/semantic configured)",
            needs_retry=False,
            layer="none",
        )

    @classmethod
    def hard_only(cls) -> "Verifier":
        """Hard-only verifier — useful for tests and offline development."""
        return cls(hard=HardVerifier())
