"""Tool router — maps an action name to a registered Tool instance.

The router is the second gate (after SafetyGuard). It rejects any action
not in the allowlist OR not registered. Both checks matter: the allowlist
says "could exist," the registry says "is wired in this build."
"""

from __future__ import annotations

from .safety.policies import ACTION_ALLOWLIST
from .tools.base import Tool


class RouterError(Exception):
    """Raised when an action cannot be routed to a tool."""


class ToolRouter:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name not in ACTION_ALLOWLIST:
            raise RouterError(
                f"refusing to register tool {tool.name!r}: not in ACTION_ALLOWLIST"
            )
        self._tools[tool.name] = tool

    def get(self, action: str) -> Tool:
        if action not in ACTION_ALLOWLIST:
            raise RouterError(f"action not in allowlist: {action!r}")
        tool = self._tools.get(action)
        if tool is None:
            raise RouterError(f"no tool registered for action: {action!r}")
        return tool

    def registered_actions(self) -> list[str]:
        return sorted(self._tools.keys())
