from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sentricoon.executor import Executor
from sentricoon.mcp import MCPToolAdapter, MockMCPClient
from sentricoon.mcp.mock import text_result
from sentricoon.router import ToolRouter
from sentricoon.safety.guard import GuardError, SafetyGuard
from sentricoon.state import Step
from sentricoon.tools.base import Tool, ToolResult
from sentricoon.tools.file_ops import FileDeleteTool, FileReadTool, FileWriteTool
from sentricoon.tools.process import ProcessKillTool, ProcessListTool
from sentricoon.tools.shell import ShellTool
from sentricoon.tools.system import SystemInfoTool
from sentricoon.verifier import HardVerifier, Verifier


# ============================================================================
# Tool flags + dry_run_describe
# ============================================================================


class TestToolClassification(unittest.TestCase):
    def test_read_only_tools(self):
        for cls in (SystemInfoTool, ProcessListTool, FileReadTool):
            self.assertTrue(cls.is_read_only, msg=cls.__name__)

    def test_write_effect_tools(self):
        for cls in (FileWriteTool, FileDeleteTool, ShellTool, ProcessKillTool):
            self.assertFalse(cls.is_read_only, msg=cls.__name__)

    def test_base_default_is_not_read_only(self):
        # Safety default: a Tool subclass that forgets to declare its flag
        # is treated as write-effect — fail closed, not open.
        self.assertFalse(Tool.is_read_only)


class TestDryRunDescribe(unittest.TestCase):
    def test_file_write_describes_path_and_bytes(self):
        d = FileWriteTool().dry_run_describe({"path": "/tmp/x.txt", "content": "hello world"})
        self.assertEqual(d["action"], "file.write")
        self.assertEqual(d["path"], "/tmp/x.txt")
        self.assertEqual(d["bytes"], 11)

    def test_file_write_would_overwrite_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "exists.txt"
            existing.write_text("prior content")
            d = FileWriteTool().dry_run_describe({"path": str(existing), "content": "new"})
            self.assertTrue(d["would_overwrite"])

            absent = str(Path(tmp) / "missing.txt")
            d = FileWriteTool().dry_run_describe({"path": absent, "content": "new"})
            self.assertFalse(d["would_overwrite"])

    def test_file_delete_describes_existence(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "delete-me.txt"
            existing.write_text("z")
            d = FileDeleteTool().dry_run_describe({"path": str(existing)})
            self.assertEqual(d["action"], "file.delete")
            self.assertTrue(d["exists_now"])

            absent = str(Path(tmp) / "ghost.txt")
            d = FileDeleteTool().dry_run_describe({"path": absent})
            self.assertFalse(d["exists_now"])

    def test_shell_describes_argv(self):
        d = ShellTool().dry_run_describe({"argv": ["net", "start", "Audiosrv"]})
        self.assertEqual(d["action"], "shell.run")
        self.assertEqual(d["argv"], ["net", "start", "Audiosrv"])

    def test_process_kill_describes_pid(self):
        d = ProcessKillTool().dry_run_describe({"pid": 1234})
        self.assertEqual(d["would_kill_pid"], 1234)

    def test_base_default_describe(self):
        class _Stub(Tool):
            name = "stub.x"

            def run(self, args):
                return ToolResult(success=True)

        d = _Stub().dry_run_describe({"k": "v"})
        self.assertEqual(d, {"action": "stub.x", "args": {"k": "v"}})


class TestMCPAdapterDryRun(unittest.TestCase):
    def test_default_is_write_effect(self):
        adapter = MCPToolAdapter(
            MockMCPClient(server_name="brave", script=[text_result("x")]),
            tool="search",
        )
        self.assertFalse(adapter.is_read_only)

    def test_explicit_read_only(self):
        adapter = MCPToolAdapter(
            MockMCPClient(server_name="brave", script=[text_result("x")]),
            tool="search", is_read_only=True,
        )
        self.assertTrue(adapter.is_read_only)

    def test_describe_includes_server_and_tool(self):
        adapter = MCPToolAdapter(
            MockMCPClient(server_name="brave", script=[]), tool="search",
        )
        d = adapter.dry_run_describe({"q": "speakers"})
        self.assertEqual(d["server"], "brave")
        self.assertEqual(d["tool"], "search")
        self.assertEqual(d["args"], {"q": "speakers"})


# ============================================================================
# SafetyGuard.validate_action_only
# ============================================================================


class TestValidateActionOnly(unittest.TestCase):
    def test_passes_for_allowlisted_action(self):
        SafetyGuard().validate_action_only(Step(action="system.info"))
        SafetyGuard().validate_action_only(Step(action="shell.run"))  # no token needed!

    def test_rejects_non_allowlisted(self):
        with self.assertRaises(GuardError):
            SafetyGuard().validate_action_only(Step(action="rogue.action"))

    def test_rejects_empty_action(self):
        with self.assertRaises(GuardError):
            SafetyGuard().validate_action_only(Step(action=""))


# ============================================================================
# Executor in dry-run mode
# ============================================================================


def _make_executor(*, dry_run: bool) -> tuple[Executor, FileWriteTool, SystemInfoTool, ShellTool]:
    fw, si, sh = FileWriteTool(), SystemInfoTool(), ShellTool()
    router = ToolRouter()
    for t in (fw, si, sh):
        router.register(t)
    return Executor(router, SafetyGuard(), dry_run=dry_run), fw, si, sh


class TestExecutorDryRun(unittest.TestCase):
    def test_simulates_write_action_without_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "would-be-written.txt")
            ex, _, _, _ = _make_executor(dry_run=True)
            result = ex.execute(Step(action="file.write", args={"path": path, "content": "x"},
                                     expected_state="file exists"))
            self.assertTrue(result.success)
            self.assertEqual(result.extra["simulated"], True)
            self.assertEqual(result.extra["action"], "file.write")
            self.assertIn("would_do", result.output)
            self.assertEqual(result.output["would_do"]["path"], path)
            # The crucial assertion: nothing actually happened
            self.assertFalse(Path(path).exists())

    def test_runs_read_only_action_for_real(self):
        ex, _, _, _ = _make_executor(dry_run=True)
        result = ex.execute(Step(action="system.info", expected_state="platform present"))
        self.assertTrue(result.success)
        # Read-only tools produce REAL output in dry-run, not simulated wrappers
        self.assertNotIn("would_do", result.output if isinstance(result.output, dict) else {})
        self.assertIsNone((result.extra or {}).get("simulated"))
        self.assertIn("platform", result.output)

    def test_bypasses_confirmation_for_high_risk(self):
        """High-risk action in dry-run mode must NOT require a confirmation token."""
        ex, _, _, _ = _make_executor(dry_run=True)
        # shell.run is in REQUIRES_CONFIRMATION; without dry_run this would raise
        result = ex.execute(
            Step(action="shell.run", args={"argv": ["echo", "hi"]}, expected_state="hi printed"),
            confirmation_token=None,
        )
        self.assertTrue(result.success)
        self.assertTrue(result.extra["simulated"])

    def test_does_not_consume_confirmation_tokens(self):
        """Even if a token is passed in dry-run, it must not be burned."""
        ex, _, _, _ = _make_executor(dry_run=True)
        ex.guard.grant_confirmation("preserve-me", "shell.run")
        ex.execute(Step(action="shell.run", args={"argv": ["echo", "x"]},
                        expected_state="x"), confirmation_token="preserve-me")
        # Token is still pending — dry-run didn't consume it
        self.assertTrue(ex.guard.has_pending("preserve-me"))

    def test_still_enforces_allowlist(self):
        ex, _, _, _ = _make_executor(dry_run=True)
        with self.assertRaises(GuardError):
            ex.execute(Step(action="rogue.action", expected_state="x"))

    def test_non_dry_run_unchanged(self):
        """Sanity: with dry_run=False the executor behaves as before."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "written.txt")
            ex, _, _, _ = _make_executor(dry_run=False)
            ex.execute(Step(action="file.write", args={"path": path, "content": "real"},
                            expected_state="exists"))
            self.assertTrue(Path(path).exists())
            self.assertEqual(Path(path).read_text(), "real")


# ============================================================================
# Verifier behavior with simulated results
# ============================================================================


class TestHardVerifierSimulated(unittest.TestCase):
    def test_passes_simulated_result(self):
        result = ToolResult(success=True, output={"would_do": {"x": 1}},
                            extra={"simulated": True, "action": "file.write"})
        v = HardVerifier().verify(Step(action="file.write"), result)
        self.assertIsNotNone(v)
        self.assertTrue(v.success)
        self.assertIn("dry-run simulated", v.reason)
        self.assertEqual(v.layer, "hard")

    def test_short_circuits_per_action_logic(self):
        """A simulated process.kill normally would return None (defers to
        observation); the simulated short-circuit must take precedence."""
        result = ToolResult(success=True, output={"would_do": {}},
                            extra={"simulated": True, "action": "process.kill"})
        v = HardVerifier().verify(Step(action="process.kill", args={"pid": 1}), result)
        self.assertIsNotNone(v)
        self.assertTrue(v.success)

    def test_non_simulated_unchanged(self):
        """Real results take the normal hard-check path."""
        result = ToolResult(success=True, extra={"returncode": 0})
        v = HardVerifier().verify(Step(action="shell.run"), result)
        self.assertTrue(v.success)
        self.assertEqual(v.reason, "exit code 0")  # NOT the dry-run message

    def test_dry_run_pass_visible_in_audit(self):
        """The reason field is what gets logged — readers must be able to
        tell a dry-run pass from a real verified pass."""
        result = ToolResult(success=True, output={"would_do": {}},
                            extra={"simulated": True, "action": "shell.run"})
        v = HardVerifier().verify(Step(action="shell.run"), result)
        self.assertIn("dry-run", v.reason.lower())
        self.assertIn("no real verification", v.reason.lower())


class TestVerifierOrchestrationSimulated(unittest.TestCase):
    def test_simulated_resolves_at_hard_layer(self):
        """Full Verifier chain: simulated should resolve at Layer 1 and never
        invoke observation or semantic layers (which would burn cycles/LLM)."""
        result = ToolResult(success=True, output={"would_do": {}},
                            extra={"simulated": True, "action": "file.write"})
        # Use hard_only — proves observation/semantic aren't needed
        v = Verifier.hard_only().verify(Step(action="file.write"), result)
        self.assertTrue(v.success)
        self.assertEqual(v.layer, "hard")


# ============================================================================
# End-to-end dry-run flow
# ============================================================================


class TestDryRunEndToEnd(unittest.TestCase):
    def test_high_risk_dry_run_passes_with_no_side_effect_and_no_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "important.txt"
            existing.write_text("don't delete me")

            ex, _, _, _ = _make_executor(dry_run=True)
            ex.router.register(FileDeleteTool())  # add the high-risk tool

            result = ex.execute(
                Step(action="file.delete", args={"path": str(existing)},
                     expected_state="file gone"),
            )
            self.assertTrue(result.success)
            self.assertTrue(result.extra["simulated"])
            self.assertTrue(result.output["would_do"]["exists_now"])
            # File should STILL exist
            self.assertTrue(existing.exists())
            self.assertEqual(existing.read_text(), "don't delete me")

            # Verifier passes the simulated result
            v = Verifier.hard_only().verify(
                Step(action="file.delete", args={"path": str(existing)}),
                result,
            )
            self.assertTrue(v.success)


if __name__ == "__main__":
    unittest.main()
