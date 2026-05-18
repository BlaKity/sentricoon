"""Canonical action allowlist + risk classification.

The allowlist is the safety primitive — anything outside it is rejected.
Substring deny-list patterns are intentionally NOT used as a gate; they're
trivially bypassable (whitespace, env vars, base64). See
[[project-goals-and-structure]] § "Non-negotiables".
"""

from __future__ import annotations

from typing import Final

ACTION_ALLOWLIST: Final[frozenset[str]] = frozenset({
    "shell.run",
    "file.read",
    "file.write",
    "file.delete",
    "process.list",
    "process.kill",
    "system.info",
    "app.open",
    "browser.open",
})

# Actions that require an explicit user confirmation token before execution.
# Confirmation is enforced by SafetyGuard.validate; tools must never bypass it.
REQUIRES_CONFIRMATION: Final[frozenset[str]] = frozenset({
    "shell.run",
    "file.delete",
    "process.kill",
})
