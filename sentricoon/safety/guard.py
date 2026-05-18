"""SafetyGuard — the single gate every step passes through before execution.

Tokens are SINGLE-USE and ACTION-BOUND:
  - Granting a token records which action it authorizes.
  - validate() consumes the token (it's removed from the store after one use).
  - A token granted for action X cannot authorize action Y.

This stops the agent (or a compromised LLM) from chaining a single
approval into multiple high-risk operations. Each high-risk step must
get its own granted token from the ConfirmationProvider.

Validation order:
  1. action is a non-empty string
  2. action is in ACTION_ALLOWLIST
  3. if action requires confirmation: token must be granted, match the
     action, and not yet consumed
"""

from __future__ import annotations

from ..state import Step
from .policies import ACTION_ALLOWLIST, REQUIRES_CONFIRMATION


class GuardError(Exception):
    """Raised when a step is blocked by the safety gate."""


class SafetyGuard:
    def __init__(self) -> None:
        # token -> action it authorizes
        self._tokens: dict[str, str] = {}

    def grant_confirmation(self, token: str, action: str) -> None:
        """Record a confirmation token bound to a specific action.

        The token is good for exactly one execution of that action.
        Re-granting the same token rebinds it.
        """
        if not token or not isinstance(token, str):
            raise GuardError("confirmation token must be a non-empty string")
        if not action or not isinstance(action, str):
            raise GuardError("action must be a non-empty string")
        self._tokens[token] = action

    def has_pending(self, token: str) -> bool:
        """Test/inspection helper. Does NOT consume."""
        return token in self._tokens

    def validate_action_only(self, step: Step) -> None:
        """Validate allowlist without consuming a confirmation token.

        Used by the executor in dry-run mode: we still refuse to simulate
        actions outside the allowlist, but we don't burn the user's
        approval to preview what they would have approved.
        """
        if not step.action or not isinstance(step.action, str):
            raise GuardError("step.action must be a non-empty string")
        if step.action not in ACTION_ALLOWLIST:
            raise GuardError(f"action not in allowlist: {step.action!r}")

    def validate(self, step: Step, *, confirmation_token: str | None = None) -> None:
        if not step.action or not isinstance(step.action, str):
            raise GuardError("step.action must be a non-empty string")

        if step.action not in ACTION_ALLOWLIST:
            raise GuardError(f"action not in allowlist: {step.action!r}")

        if step.action in REQUIRES_CONFIRMATION:
            if not confirmation_token:
                raise GuardError(
                    f"action {step.action!r} requires a granted confirmation token"
                )
            bound_action = self._tokens.get(confirmation_token)
            if bound_action is None:
                raise GuardError(
                    f"unknown or already-used confirmation token for action {step.action!r}"
                )
            if bound_action != step.action:
                # Do NOT consume the token on action mismatch — preserve the
                # user's legitimate approval for the action it was granted for.
                # Otherwise a buggy/malicious caller could burn approvals by
                # trying the wrong action first.
                raise GuardError(
                    f"confirmation token authorizes action {bound_action!r}, "
                    f"not {step.action!r}"
                )
            # Clean match — consume now.
            del self._tokens[confirmation_token]
