"""MCP (Model Context Protocol) adapter — see [[plan-mcp-integration]].

Public surface:
  - MCPClient                — Protocol every MCP transport implements
  - MCPToolInfo, MCPCallResult, MCPContent — wire-level types
  - MockMCPClient            — test/dev client
  - MCPToolAdapter           — wraps an MCP tool as a native Tool subclass

The official `mcp` Python SDK is intentionally NOT a hard dependency.
When you wire a real MCP server, install `mcp` and write a thin wrapper
that adapts its types to our `MCPClient` Protocol. This keeps the rest
of the agent free of SDK-version coupling.
"""

from .adapter import MCPToolAdapter, mcp_action_name
from .client import MCPCallResult, MCPClient, MCPContent, MCPToolInfo
from .mock import MockMCPClient

__all__ = [
    "MCPCallResult",
    "MCPClient",
    "MCPContent",
    "MCPToolAdapter",
    "MCPToolInfo",
    "MockMCPClient",
    "mcp_action_name",
]
