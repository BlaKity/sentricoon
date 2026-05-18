"""Base class for all tools.

Tools are deterministic. They take a dict of args and return a dict of
results. No reasoning, no LLM calls, no surprises. The agent's intelligence
lives in the planner/diagnoser, NOT in tools.

Two extra attributes support dry-run mode (Step 9):
  - is_read_only: bool — True means the tool causes no side effects and
    is safe to run unchanged even in dry-run mode.
  - dry_run_describe(args) -> dict — produces a faithful preview of what
    a write-effect tool *would* do. Only called by Executor when in
    dry-run mode and the tool is not read-only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: str | None = None
    extra: dict[str, Any] | None = None


class Tool(ABC):
    name: str = ""
    risk_level: str = "low"
    is_read_only: bool = False  # write-effect by default — opt-in to read-only

    @abstractmethod
    def run(self, args: dict[str, Any]) -> ToolResult: ...

    def dry_run_describe(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return a dict describing what this tool *would* do given args.

        Default impl is generic — `{action, args}`. Tools that can produce
        a more useful preview (file sizes, existence checks, etc.) should
        override. Only called by Executor in dry-run mode for non-read-only
        tools.
        """
        return {"action": self.name, "args": dict(args)}
