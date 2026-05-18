"""MockMCPClient — deterministic MCP client for tests and offline dev."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .client import MCPCallResult, MCPContent, MCPToolInfo


class MockMCPClient:
    """Configurable MCP client for tests.

    Two modes for tool calls:
      - script:  pre-loaded list of MCPCallResults, returned in order
      - responder: callable (tool_name, args) -> MCPCallResult
    """

    def __init__(
        self,
        *,
        server_name: str = "mock",
        tools: list[MCPToolInfo] | None = None,
        script: list[MCPCallResult] | None = None,
        responder: Callable[[str, dict[str, Any]], MCPCallResult] | None = None,
    ) -> None:
        if script is None and responder is None:
            raise ValueError("MockMCPClient needs script= or responder=")
        self.server_name = server_name
        self._tools = list(tools or [])
        self._script = list(script) if script is not None else None
        self._responder = responder
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_tools(self) -> list[MCPToolInfo]:
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPCallResult:
        self.calls.append((name, dict(arguments)))
        if self._script is not None:
            if not self._script:
                raise RuntimeError(f"MockMCPClient {self.server_name!r} script exhausted")
            return self._script.pop(0)
        assert self._responder is not None
        return self._responder(name, arguments)


def text_result(*parts: str, is_error: bool = False) -> MCPCallResult:
    """Convenience constructor for tests."""
    return MCPCallResult(
        content=[MCPContent(type="text", text=p) for p in parts],
        is_error=is_error,
    )
