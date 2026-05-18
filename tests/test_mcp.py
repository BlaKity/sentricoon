from __future__ import annotations

import unittest
from unittest.mock import patch

from sentricoon.mcp import (
    MCPCallResult,
    MCPClient,
    MCPContent,
    MCPToolAdapter,
    MCPToolInfo,
    MockMCPClient,
    mcp_action_name,
)
from sentricoon.mcp.mock import text_result
from sentricoon.router import RouterError, ToolRouter
from sentricoon.safety.guard import SafetyGuard
from sentricoon.safety.policies import ACTION_ALLOWLIST
from sentricoon.state import Step
from sentricoon.tools.base import Tool


# ============================================================================
# Action name canonicalization
# ============================================================================


class TestMCPActionName(unittest.TestCase):
    def test_normal_name(self):
        self.assertEqual(mcp_action_name("brave", "search"), "mcp.brave.search")

    def test_rejects_empty_server(self):
        with self.assertRaises(ValueError):
            mcp_action_name("", "search")

    def test_rejects_empty_tool(self):
        with self.assertRaises(ValueError):
            mcp_action_name("brave", "")

    def test_rejects_dotted_server(self):
        with self.assertRaises(ValueError):
            mcp_action_name("brave.cloud", "search")

    def test_rejects_dotted_tool(self):
        with self.assertRaises(ValueError):
            mcp_action_name("brave", "search.web")


# ============================================================================
# MockMCPClient
# ============================================================================


class TestMockMCPClient(unittest.TestCase):
    def test_satisfies_protocol(self):
        client = MockMCPClient(script=[text_result("hi")])
        self.assertIsInstance(client, MCPClient)

    def test_list_tools_returns_configured(self):
        tools = [MCPToolInfo(name="search", description="d")]
        client = MockMCPClient(tools=tools, script=[])
        self.assertEqual(client.list_tools(), tools)

    def test_script_returns_in_order(self):
        client = MockMCPClient(script=[text_result("a"), text_result("b")])
        self.assertEqual(client.call_tool("x", {}).content[0].text, "a")
        self.assertEqual(client.call_tool("x", {}).content[0].text, "b")

    def test_script_exhaustion_raises(self):
        client = MockMCPClient(script=[text_result("only")])
        client.call_tool("x", {})
        with self.assertRaises(RuntimeError):
            client.call_tool("x", {})

    def test_responder_called_with_args(self):
        seen = {}

        def responder(name, args):
            seen["name"] = name
            seen["args"] = args
            return text_result("ok")

        client = MockMCPClient(responder=responder)
        client.call_tool("search", {"q": "speakers"})
        self.assertEqual(seen, {"name": "search", "args": {"q": "speakers"}})

    def test_calls_recorded(self):
        client = MockMCPClient(script=[text_result("a"), text_result("b")])
        client.call_tool("search", {"q": "one"})
        client.call_tool("search", {"q": "two"})
        self.assertEqual(client.calls, [("search", {"q": "one"}), ("search", {"q": "two"})])

    def test_requires_script_or_responder(self):
        with self.assertRaises(ValueError):
            MockMCPClient()  # type: ignore[call-arg]


# ============================================================================
# MCPToolAdapter
# ============================================================================


class TestMCPToolAdapter(unittest.TestCase):
    def _adapter(self, **mock_kwargs):
        client = MockMCPClient(**mock_kwargs)
        return MCPToolAdapter(client, server="brave", tool="search"), client

    def test_is_a_tool(self):
        adapter, _ = self._adapter(script=[text_result("x")])
        self.assertIsInstance(adapter, Tool)

    def test_action_name(self):
        adapter, _ = self._adapter(script=[text_result("x")])
        self.assertEqual(adapter.name, "mcp.brave.search")

    def test_inherits_server_name_when_not_passed(self):
        client = MockMCPClient(server_name="github", script=[text_result("x")])
        adapter = MCPToolAdapter(client, tool="search_issues")
        self.assertEqual(adapter.name, "mcp.github.search_issues")

    def test_run_passes_args_to_client(self):
        adapter, client = self._adapter(script=[text_result("results")])
        adapter.run({"q": "speakers"})
        self.assertEqual(client.calls, [("search", {"q": "speakers"})])

    def test_success_text_result(self):
        adapter, _ = self._adapter(script=[text_result("hit 1", "hit 2")])
        r = adapter.run({"q": "x"})
        self.assertTrue(r.success)
        self.assertEqual(r.output, "hit 1\nhit 2")
        self.assertEqual(r.extra["server"], "brave")
        self.assertEqual(r.extra["tool"], "search")

    def test_error_result(self):
        adapter, _ = self._adapter(script=[text_result("API quota exhausted", is_error=True)])
        r = adapter.run({"q": "x"})
        self.assertFalse(r.success)
        self.assertIn("quota", r.error)

    def test_error_result_with_no_content_gets_default_message(self):
        result = MCPCallResult(content=[], is_error=True)
        adapter, _ = self._adapter(script=[result])
        r = adapter.run({})
        self.assertFalse(r.success)
        self.assertIn("isError=true with no content", r.error)

    def test_non_text_content_stringified(self):
        result = MCPCallResult(
            content=[
                MCPContent(type="text", text="here is the image:"),
                MCPContent(type="image", data=b"\x89PNG"),
            ]
        )
        adapter, _ = self._adapter(script=[result])
        r = adapter.run({})
        self.assertTrue(r.success)
        self.assertIn("here is the image:", r.output)
        self.assertIn("[image:", r.output)
        self.assertEqual(r.extra["non_text_parts"], 1)

    def test_client_exception_becomes_failed_tool_result(self):
        def boom(name, args):
            raise ConnectionError("server gone")

        client = MockMCPClient(responder=boom)
        adapter = MCPToolAdapter(client, server="brave", tool="search")
        r = adapter.run({})
        self.assertFalse(r.success)
        self.assertIn("ConnectionError", r.error)
        self.assertIn("server gone", r.error)

    def test_none_args_handled(self):
        adapter, client = self._adapter(script=[text_result("x")])
        adapter.run(None)  # type: ignore[arg-type]
        # Should pass {} to the client, not None
        self.assertEqual(client.calls, [("search", {})])

    def test_risk_level_propagates(self):
        client = MockMCPClient(script=[text_result("x")])
        adapter = MCPToolAdapter(client, server="brave", tool="search", risk_level="medium")
        self.assertEqual(adapter.risk_level, "medium")


# ============================================================================
# Integration with router + safety (requires allowlist patch)
# ============================================================================


class TestMCPRouterIntegration(unittest.TestCase):
    """Verifies that an MCP adapter behaves like any other Tool inside the
    existing ToolRouter + SafetyGuard machinery. Action name must be added
    to ACTION_ALLOWLIST explicitly — we patch it here for the test."""

    def test_router_rejects_unlisted_mcp_action(self):
        # Without patching the allowlist, registration must fail.
        client = MockMCPClient(script=[text_result("x")])
        adapter = MCPToolAdapter(client, server="brave", tool="search")
        router = ToolRouter()
        with self.assertRaises(RouterError):
            router.register(adapter)

    def test_router_accepts_when_action_is_allowlisted(self):
        patched = frozenset(ACTION_ALLOWLIST | {"mcp.brave.search"})
        with patch("sentricoon.router.ACTION_ALLOWLIST", patched), \
             patch("sentricoon.safety.guard.ACTION_ALLOWLIST", patched):
            client = MockMCPClient(script=[text_result("relevant docs")])
            adapter = MCPToolAdapter(client, server="brave", tool="search")
            router = ToolRouter()
            router.register(adapter)
            self.assertIn("mcp.brave.search", router.registered_actions())

    def test_safety_guard_allows_mcp_action_without_confirmation(self):
        patched = frozenset(ACTION_ALLOWLIST | {"mcp.brave.search"})
        with patch("sentricoon.safety.guard.ACTION_ALLOWLIST", patched):
            # mcp.brave.search is NOT in REQUIRES_CONFIRMATION → no token needed
            SafetyGuard().validate(Step(action="mcp.brave.search", args={"q": "x"}))


if __name__ == "__main__":
    unittest.main()
