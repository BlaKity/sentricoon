"""MCP wire-level types + client Protocol.

Mirrors the shape of the MCP spec without binding to the official SDK.
A future SDK-backed implementation just satisfies the Protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class MCPToolInfo:
    """One entry from MCP `tools/list`."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPContent:
    """One content item returned by `tools/call`.

    MCP supports text, image, embedded resource. For the OS-Technician
    use case we only consume text — image/binary results are stringified
    via repr in the adapter.
    """

    type: str  # "text" | "image" | "resource"
    text: str = ""
    data: Any = None  # for non-text payloads


@dataclass
class MCPCallResult:
    """The result of an MCP `tools/call`."""

    content: list[MCPContent] = field(default_factory=list)
    is_error: bool = False


@runtime_checkable
class MCPClient(Protocol):
    """Minimum interface the adapter uses from an MCP client.

    A real implementation (e.g., wrapping the official `mcp` Python SDK)
    just needs `list_tools()` and `call_tool()`.
    """

    server_name: str

    def list_tools(self) -> list[MCPToolInfo]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPCallResult: ...
