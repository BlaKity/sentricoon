"""Executor — runs a step by passing it through SafetyGuard then ToolRouter.

Sits between the agent loop and the tools. The executor never reasons —
it validates, dispatches, and returns the ToolResult.

Dry-run mode (Step 9):
  - `Executor(..., dry_run=True)` short-circuits write-effect tools.
  - Read-only tools (`is_read_only=True`) run normally — they have no
    side effects to suppress, and their real output is exactly what
    a dry-run preview needs.
  - Write-effect tools call `tool.dry_run_describe(args)` and the result
    is wrapped in a `ToolResult` with `extra["simulated"]=True`. The
    HardVerifier recognizes that flag and passes the step without
    running real checks (which would all fail because nothing happened).
  - In dry-run we use `validate_action_only` — the allowlist is still
    enforced, but confirmation tokens are NOT consumed. Dry-run's whole
    point is "show me what would happen without burning my approvals."
"""

from __future__ import annotations

from .router import ToolRouter
from .safety.guard import SafetyGuard
from .state import Step
from .tools.base import ToolResult


class Executor:
    def __init__(
        self,
        router: ToolRouter,
        guard: SafetyGuard,
        *,
        dry_run: bool = False,
    ) -> None:
        self.router = router
        self.guard = guard
        self.dry_run = dry_run

    def execute(self, step: Step, *, confirmation_token: str | None = None) -> ToolResult:
        if self.dry_run:
            return self._dry_run(step)

        self.guard.validate(step, confirmation_token=confirmation_token)
        tool = self.router.get(step.action)
        return tool.run(step.args)

    def _dry_run(self, step: Step) -> ToolResult:
        # Allowlist still enforced — we don't simulate unauthorized actions.
        # But no confirmation token consumed: dry-run shouldn't burn approvals.
        self.guard.validate_action_only(step)
        tool = self.router.get(step.action)

        if tool.is_read_only:
            # Read-only tools have no side effects; let them run for real.
            # Their actual output is the most faithful preview.
            return tool.run(step.args)

        # Write-effect tool: ask it to describe what it would do.
        describe = tool.dry_run_describe(step.args)
        return ToolResult(
            success=True,
            output={"would_do": describe},
            extra={"simulated": True, "action": step.action},
        )
