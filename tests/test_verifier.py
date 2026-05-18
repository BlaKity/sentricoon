from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sentricoon.executor import Executor
from sentricoon.llm import LLMRouter, MockBackend, RouterConfig
from sentricoon.router import ToolRouter
from sentricoon.safety.guard import SafetyGuard
from sentricoon.state import Step
from sentricoon.tools.base import Tool, ToolResult
from sentricoon.tools.file_ops import FileReadTool, FileWriteTool
from sentricoon.verifier import (
    HardVerifier,
    ObservationVerifier,
    SemanticVerifier,
    Verification,
    Verifier,
)


# ============================================================================
# HardVerifier
# ============================================================================


class TestHardVerifier(unittest.TestCase):
    def test_shell_run_returncode_zero(self):
        v = HardVerifier().verify(
            Step(action="shell.run"),
            ToolResult(success=True, extra={"returncode": 0}),
        )
        self.assertTrue(v.success)
        self.assertEqual(v.layer, "hard")

    def test_shell_run_nonzero(self):
        v = HardVerifier().verify(
            Step(action="shell.run"),
            ToolResult(success=True, extra={"returncode": 1}),
        )
        self.assertFalse(v.success)
        self.assertFalse(v.needs_retry)
        self.assertIn("non-zero", v.reason)

    def test_read_only_actions_succeed(self):
        for action in ("system.info", "process.list", "file.read"):
            v = HardVerifier().verify(Step(action=action), ToolResult(success=True))
            self.assertTrue(v.success, msg=action)

    def test_file_write_reported_success_passes_hard(self):
        v = HardVerifier().verify(
            Step(action="file.write", args={"path": "/tmp/x", "content": "y"}),
            ToolResult(success=True, output={"path": "/tmp/x"}),
        )
        self.assertTrue(v.success)

    def test_process_kill_defers_to_observation(self):
        v = HardVerifier().verify(
            Step(action="process.kill", args={"pid": 1234}),
            ToolResult(success=True, output={"killed": 1234}),
        )
        self.assertIsNone(v)

    def test_unknown_action_returns_none(self):
        v = HardVerifier().verify(Step(action="totally.unknown"), ToolResult(success=True))
        self.assertIsNone(v)

    def test_tool_failure_propagates(self):
        v = HardVerifier().verify(
            Step(action="file.write"),
            ToolResult(success=False, error="permission denied"),
        )
        self.assertFalse(v.success)
        self.assertFalse(v.needs_retry)
        self.assertIn("permission", v.reason)

    def test_transient_error_marks_needs_retry(self):
        for err in ("Timeout after 30s", "resource busy", "Try again later"):
            v = HardVerifier().verify(
                Step(action="shell.run"),
                ToolResult(success=False, error=err),
            )
            self.assertTrue(v.needs_retry, msg=err)

    def test_non_transient_error_no_retry(self):
        v = HardVerifier().verify(
            Step(action="shell.run"),
            ToolResult(success=False, error="command not found"),
        )
        self.assertFalse(v.needs_retry)


# ============================================================================
# ObservationVerifier — uses real file system + a real executor with real tools
# ============================================================================


def _make_executor() -> Executor:
    """Build an executor with read-only tools wired (no confirmation needed)."""
    from sentricoon.tools.process import ProcessListTool

    router = ToolRouter()
    router.register(FileReadTool())
    router.register(ProcessListTool())
    return Executor(router, SafetyGuard())


class _FakeProcessListTool(Tool):
    """Returns a controlled process list so we can verify kill outcomes deterministically."""

    name = "process.list"
    risk_level = "low"

    def __init__(self, procs: list[dict]) -> None:
        self.procs = procs

    def run(self, args):
        return ToolResult(success=True, output=self.procs)


def _make_executor_with_proc_list(procs: list[dict]) -> Executor:
    router = ToolRouter()
    router.register(_FakeProcessListTool(procs))
    return Executor(router, SafetyGuard())


class TestObservationVerifierFileWrite(unittest.TestCase):
    def test_content_match_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "x.txt")
            FileWriteTool().run({"path": path, "content": "hello"})
            v = ObservationVerifier(_make_executor()).verify(
                Step(action="file.write", args={"path": path, "content": "hello"}),
                ToolResult(success=True),
            )
            self.assertTrue(v.success)
            self.assertEqual(v.layer, "observation")

    def test_content_differs_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "x.txt")
            FileWriteTool().run({"path": path, "content": "what was actually written"})
            v = ObservationVerifier(_make_executor()).verify(
                Step(action="file.write", args={"path": path, "content": "what we expected"}),
                ToolResult(success=True),
            )
            self.assertFalse(v.success)
            self.assertIn("differs", v.reason)

    def test_missing_path_fails(self):
        v = ObservationVerifier(_make_executor()).verify(
            Step(action="file.write", args={"path": "/definitely/not/a/path/xyz", "content": "x"}),
            ToolResult(success=True),
        )
        self.assertFalse(v.success)
        self.assertIn("not present", v.reason)

    def test_bad_args_returns_none(self):
        v = ObservationVerifier(_make_executor()).verify(
            Step(action="file.write", args={"path": 123}),
            ToolResult(success=True),
        )
        self.assertIsNone(v)


class TestObservationVerifierFileDelete(unittest.TestCase):
    def test_file_gone_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "gone.txt")
            v = ObservationVerifier(_make_executor()).verify(
                Step(action="file.delete", args={"path": path}),
                ToolResult(success=True),
            )
            self.assertTrue(v.success)

    def test_file_still_present_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "still.txt"
            path.write_text("oops")
            v = ObservationVerifier(_make_executor()).verify(
                Step(action="file.delete", args={"path": str(path)}),
                ToolResult(success=True),
            )
            self.assertFalse(v.success)
            self.assertIn("still present", v.reason)


class TestObservationVerifierProcessKill(unittest.TestCase):
    def test_pid_gone_succeeds(self):
        # The "killed" pid 1234 is not in the listing → success
        executor = _make_executor_with_proc_list([{"pid": 5555, "name": "other"}])
        v = ObservationVerifier(executor).verify(
            Step(action="process.kill", args={"pid": 1234}),
            ToolResult(success=True),
        )
        self.assertTrue(v.success)

    def test_pid_still_present_fails_and_retries(self):
        executor = _make_executor_with_proc_list([{"pid": 1234, "name": "stubborn"}])
        v = ObservationVerifier(executor).verify(
            Step(action="process.kill", args={"pid": 1234}),
            ToolResult(success=True),
        )
        self.assertFalse(v.success)
        self.assertTrue(v.needs_retry)

    def test_bad_pid_returns_none(self):
        v = ObservationVerifier(_make_executor_with_proc_list([])).verify(
            Step(action="process.kill", args={"pid": "not-int"}),
            ToolResult(success=True),
        )
        self.assertIsNone(v)


# ============================================================================
# SemanticVerifier
# ============================================================================


def _router_with_mock(content: str):
    backend = MockBackend(script=[content], name="local", model="local-1")
    router = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
    router.register(backend)
    router.register(MockBackend(script=[], name="cloud", model="cloud-1"))
    return router, backend


class TestSemanticVerifier(unittest.TestCase):
    def test_success_parsed(self):
        router, backend = _router_with_mock(json.dumps({"success": True, "reason": "looks good"}))
        v = SemanticVerifier(router).verify(
            Step(action="shell.run", expected_state="audio service running"),
            ToolResult(success=True),
        )
        self.assertTrue(v.success)
        self.assertIn("looks good", v.reason)
        self.assertEqual(v.layer, "semantic")
        # Routed to local backend
        self.assertEqual(len(backend.calls), 1)

    def test_failure_parsed(self):
        router, _ = _router_with_mock(json.dumps({"success": False, "reason": "wrong device"}))
        v = SemanticVerifier(router).verify(
            Step(action="shell.run", expected_state="x"),
            ToolResult(success=True),
        )
        self.assertFalse(v.success)
        self.assertIn("wrong device", v.reason)

    def test_no_expected_state_fails_fast(self):
        router, backend = _router_with_mock(json.dumps({"success": True, "reason": "x"}))
        v = SemanticVerifier(router).verify(
            Step(action="shell.run", expected_state=None),
            ToolResult(success=True),
        )
        self.assertFalse(v.success)
        self.assertIn("expected_state", v.reason)
        # Should not have called the LLM
        self.assertEqual(len(backend.calls), 0)

    def test_non_json_response_handled(self):
        router, _ = _router_with_mock("not json at all")
        v = SemanticVerifier(router).verify(
            Step(action="shell.run", expected_state="x"),
            ToolResult(success=True),
        )
        self.assertFalse(v.success)
        self.assertIn("non-JSON", v.reason)

    def test_non_object_json_response_handled(self):
        router, _ = _router_with_mock(json.dumps(["a", "b"]))
        v = SemanticVerifier(router).verify(
            Step(action="shell.run", expected_state="x"),
            ToolResult(success=True),
        )
        self.assertFalse(v.success)
        self.assertIn("non-object", v.reason)

    def test_backend_error_marks_needs_retry(self):
        # Empty script causes the mock to raise — semantic verifier treats this as transient
        backend = MockBackend(script=[], name="local", model="local-1")
        router = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
        router.register(backend)
        router.register(MockBackend(script=[], name="cloud", model="cloud-1"))
        v = SemanticVerifier(router).verify(
            Step(action="shell.run", expected_state="x"),
            ToolResult(success=True),
        )
        self.assertFalse(v.success)
        self.assertTrue(v.needs_retry)
        self.assertIn("unreachable", v.reason)

    def test_uses_verifier_role_not_planner(self):
        """Critical: this is the cost-saving routing — verifier must hit local."""
        router, local_backend = _router_with_mock(json.dumps({"success": True, "reason": "ok"}))
        SemanticVerifier(router).verify(
            Step(action="shell.run", expected_state="x"),
            ToolResult(success=True),
        )
        # Local backend was called; cloud backend's call count would be in router.call_counts
        self.assertEqual(router.call_counts.get("local"), 1)
        self.assertIsNone(router.call_counts.get("cloud"))


# ============================================================================
# Verifier orchestration
# ============================================================================


class _StaticHard(HardVerifier):
    def __init__(self, value):
        self.value = value

    def verify(self, step, result):
        return self.value


class _StaticObs:
    def __init__(self, value):
        self.value = value

    def verify(self, step, result):
        return self.value


class _StaticSem:
    def __init__(self, value):
        self.value = value

    def verify(self, step, result):
        return self.value


class TestVerifierOrchestration(unittest.TestCase):
    def test_hard_definitive_short_circuits(self):
        verdict = Verification(success=True, reason="hard ok", layer="hard")
        v = Verifier(
            hard=_StaticHard(verdict),
            observation=_StaticObs(Verification(success=False, reason="obs", layer="observation")),
            semantic=_StaticSem(Verification(success=False, reason="sem", layer="semantic")),
        ).verify(Step(action="x"), ToolResult(success=True))
        self.assertEqual(v.reason, "hard ok")
        self.assertEqual(v.layer, "hard")

    def test_falls_through_to_observation(self):
        obs_verdict = Verification(success=True, reason="obs ok", layer="observation")
        v = Verifier(
            hard=_StaticHard(None),
            observation=_StaticObs(obs_verdict),
            semantic=_StaticSem(Verification(success=False, reason="sem", layer="semantic")),
        ).verify(Step(action="x"), ToolResult(success=True))
        self.assertEqual(v.reason, "obs ok")
        self.assertEqual(v.layer, "observation")

    def test_falls_through_to_semantic(self):
        sem_verdict = Verification(success=True, reason="sem ok", layer="semantic")
        v = Verifier(
            hard=_StaticHard(None),
            observation=_StaticObs(None),
            semantic=_StaticSem(sem_verdict),
        ).verify(Step(action="x"), ToolResult(success=True))
        self.assertEqual(v.reason, "sem ok")
        self.assertEqual(v.layer, "semantic")

    def test_no_layer_resolves_falls_back(self):
        v = Verifier(hard=_StaticHard(None)).verify(Step(action="x"), ToolResult(success=True))
        self.assertFalse(v.success)
        self.assertEqual(v.layer, "none")
        self.assertIn("no verifier layer", v.reason)

    def test_hard_only_factory(self):
        v = Verifier.hard_only().verify(
            Step(action="system.info"),
            ToolResult(success=True),
        )
        self.assertTrue(v.success)
        self.assertEqual(v.layer, "hard")


if __name__ == "__main__":
    unittest.main()
