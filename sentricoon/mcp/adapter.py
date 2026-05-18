"""MCPToolAdapter — wraps an MCP tool as a native Tool subclass.

Action name format: `mcp.<server>.<tool>`. The server prefix prevents
collisions between different MCP servers exposing tools with the same
name (e.g., two servers both publishing `search`).

Safety story:
  - The adapter is just a Tool. It goes through SafetyGuard and
    ToolRouter like any other tool.
  - The user must add the action name to ACTION_ALLOWLIST in
    safety/policies.py EXPLICITLY before registering. Untrusted by default.
  - risk_level and requires_confirmation are set by the *user*, not read
    from the MCP server's self-description (which is advisory and could
    be misleading).
"""

from __future__ import annotations

from typing import Any

from ..tools.base import Tool, ToolResult
from .client import MCPCallResult, MCPClient


def mcp_action_name(server: str, tool: str) -> str:
    """Canonical action name for an MCP tool.

    Validates that server and tool are non-empty and contain no dots,
    so the resulting name parses unambiguously.
    """
    if not server or not isinstance(server, str) or "." in server:
        raise ValueError(f"invalid MCP server name: {server!r} (must be non-empty, no dots)")
    if not tool or not isinstance(tool, str) or "." in tool:
        raise ValueError(f"invalid MCP tool name: {tool!r} (must be non-empty, no dots)")
    return f"mcp.{server}.{tool}"


class MCPToolAdapter(Tool):
    """Adapts a single MCP-server-side tool as a Tool subclass."""

    def __init__(
        self,
        client: MCPClient,
        *,
        server: str | None = None,
        tool: str,
        risk_level: str = "low",
        is_read_only: bool = False,
    ) -> None:
        self.client = client
        self.tool_name = tool
        self.server = server or client.server_name
        self.name = mcp_action_name(self.server, tool)
        self.risk_level = risk_level
        # Conservative default: MCP tools are write-effect unless the
        # integrator declares otherwise. The server's self-description is
        # not authoritative — the human registering the tool decides.
        self.is_read_only = is_read_only

    def dry_run_describe(self, args):
        return {"action": self.name, "server": self.server, "tool": self.tool_name, "args": dict(args or {})}

    def run(self, args: dict[str, Any]) -> ToolResult:
        try:
            result = self.client.call_tool(self.tool_name, args or {})
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"mcp call raised: {type(e).__name__}: {e}",
                extra={"server": self.server, "tool": self.tool_name},
            )

        return _convert(result, server=self.server, tool=self.tool_name)


def _convert(result: MCPCallResult, *, server: str, tool: str) -> ToolResult:
    """Convert an MCPCallResult to our native ToolResult."""
    text_parts: list[str] = []
    non_text_count = 0
    for c in result.content:
        if c.type == "text":
            text_parts.append(c.text)
        else:
            non_text_count += 1
            text_parts.append(f"[{c.type}: {c.data!r}]")
    joined = "\n".join(text_parts)

    extra = {"server": server, "tool": tool, "non_text_parts": non_text_count}

    if result.is_error:
        return ToolResult(
            success=False,
            error=joined or "mcp tool returned isError=true with no content",
            extra=extra,
        )

    return ToolResult(success=True, output=joined, extra=extra)
