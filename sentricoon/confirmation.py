"""Confirmation UX — surfaces high-risk actions to the human before execution.

The agent loop, when it sees a step with `action in REQUIRES_CONFIRMATION`,
calls a `ConfirmationProvider.request(...)`. The provider blocks until the
user responds and returns a ConfirmationResponse. If granted, the loop
passes the returned token to `Executor.execute(step, confirmation_token=...)`.

Tokens are single-use AND action-bound (see safety/guard.py): one token
authorizes one specific action exactly once. The model cannot chain a
single approval into multiple high-risk operations.

Providers:
  - CLIConfirmationProvider — stdin/stdout, blocks for y/n input
  - AutoApproveProvider     — for tests / explicit --yes mode
  - AlwaysDenyProvider      — for tests / dry-run safety
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from .state import Step


def new_confirmation_token() -> str:
    """Cryptographically random token. Single-use; consumed by SafetyGuard."""
    return secrets.token_urlsafe(16)


@dataclass
class ConfirmationRequest:
    step: Step
    rationale: str | None = None  # agent's explanation of why this action


@dataclass
class ConfirmationResponse:
    granted: bool
    token: str | None = None      # populated iff granted
    reason: str | None = None     # optional human reason (for denial logging)


@runtime_checkable
class ConfirmationProvider(Protocol):
    def request(self, request: ConfirmationRequest) -> ConfirmationResponse: ...


# ============================================================================
# CLI provider — the real human-in-the-loop implementation
# ============================================================================


# Tokens that, when they appear as argv[0] of a shell.run step, trigger
# an extra-visible warning banner in the confirmation prompt. Defense in
# depth: even if the planner ignored "do not use sudo" and the risk
# warning was missed, the user sees a clear banner before approving.
_PRIVILEGE_ESCALATION_TOKENS: frozenset[str] = frozenset({
    "sudo", "su", "doas", "runas", "gsudo",
})


def _privilege_escalation_token(step) -> str | None:
    """Return the first-argv-token that's a privilege-escalation marker, or None."""
    if step.action != "shell.run":
        return None
    argv = step.args.get("argv")
    if not isinstance(argv, list) or not argv:
        return None
    first = str(argv[0]).lower()
    if first in _PRIVILEGE_ESCALATION_TOKENS:
        return str(argv[0])
    return None


def _format_prompt(request: ConfirmationRequest) -> str:
    """Render a confirmation prompt to a multi-line string.

    Pure function — kept separate from I/O so it's directly testable.
    """
    step = request.step
    lines = [
        "",
        "=" * 70,
        "ACTION REQUIRES CONFIRMATION",
        "=" * 70,
    ]
    elevation_token = _privilege_escalation_token(step)
    if elevation_token is not None:
        lines.append(f"  [!] PRIVILEGE ESCALATION DETECTED: {elevation_token!r}")
        lines.append("      This step will run with elevated privileges.")
        lines.append("      Deny if you did NOT authorize sudo / admin in your task.")
        lines.append("")
    lines.append(f"  Action:    {step.action}  (risk: {step.risk_level})")
    if step.args:
        lines.append("  Args:")
        for k, v in step.args.items():
            lines.append(f"    {k} = {v!r}")
    else:
        lines.append("  Args:      (none)")
    if step.expected_state:
        lines.append(f"  Expected:  {step.expected_state}")
    if request.rationale:
        lines.append(f"  Rationale: {request.rationale}")
    lines.append("")
    lines.append("Approve? [y/N]: ")
    return "\n".join(lines)


class CLIConfirmationProvider:
    """Prompts the user via stdin and prints to stdout.

    `input_fn` and `output_fn` are injected for testability — production
    code passes `input` and `print`.
    """

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        token_factory: Callable[[], str] = new_confirmation_token,
    ) -> None:
        self._input = input_fn
        self._output = output_fn
        self._token_factory = token_factory

    def request(self, request: ConfirmationRequest) -> ConfirmationResponse:
        prompt = _format_prompt(request)
        # Print the formatted prompt without the trailing "Approve?" line,
        # then call input() with the question — that way the user input
        # ends up on its own line in the transcript.
        body, question = prompt.rsplit("\n", 1)
        self._output(body)
        try:
            answer = self._input(question).strip().lower()
        except EOFError:
            return ConfirmationResponse(granted=False, reason="EOF on stdin — treated as denial")
        except KeyboardInterrupt:
            return ConfirmationResponse(granted=False, reason="interrupted by user")

        if answer in {"y", "yes"}:
            return ConfirmationResponse(granted=True, token=self._token_factory())
        if answer in {"", "n", "no"}:
            return ConfirmationResponse(granted=False, reason="user said no")
        return ConfirmationResponse(
            granted=False,
            reason=f"unrecognized answer {answer!r} — treated as denial",
        )


# ============================================================================
# Test / dev providers
# ============================================================================


class AutoApproveProvider:
    """Always grants. Use for tests or an explicit `--yes` runtime mode.

    DO NOT make this the default in production — it disables the safety UX.
    """

    def __init__(self, *, token_factory: Callable[[], str] = new_confirmation_token) -> None:
        self._token_factory = token_factory

    def request(self, request: ConfirmationRequest) -> ConfirmationResponse:
        return ConfirmationResponse(granted=True, token=self._token_factory())


class AlwaysDenyProvider:
    """Always denies. Useful in dry-run mode to assert no high-risk action
    can be executed even if the agent loop forgets to check."""

    def request(self, request: ConfirmationRequest) -> ConfirmationResponse:
        return ConfirmationResponse(granted=False, reason="denied by AlwaysDenyProvider")
