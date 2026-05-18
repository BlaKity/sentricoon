from __future__ import annotations

import io
import unittest

from sentricoon.confirmation import (
    AlwaysDenyProvider,
    AutoApproveProvider,
    CLIConfirmationProvider,
    ConfirmationProvider,
    ConfirmationRequest,
    ConfirmationResponse,
    _format_prompt,
    new_confirmation_token,
)
from sentricoon.executor import Executor
from sentricoon.router import ToolRouter
from sentricoon.safety.guard import GuardError, SafetyGuard
from sentricoon.state import Step
from sentricoon.tools.base import Tool, ToolResult


class _NoOpKill(Tool):
    name = "process.kill"
    risk_level = "high"

    def __init__(self):
        self.calls = []

    def run(self, args):
        self.calls.append(args)
        return ToolResult(success=True, output={"killed": args.get("pid")})


def _req(action="process.kill", **args) -> ConfirmationRequest:
    return ConfirmationRequest(
        step=Step(action=action, args=args, expected_state="x", risk_level="high"),
        rationale="agent thinks this is needed",
    )


# ============================================================================
# Token generation
# ============================================================================


class TestTokenGeneration(unittest.TestCase):
    def test_tokens_are_unique(self):
        tokens = {new_confirmation_token() for _ in range(100)}
        self.assertEqual(len(tokens), 100)

    def test_tokens_are_non_trivial_length(self):
        tok = new_confirmation_token()
        self.assertGreaterEqual(len(tok), 16)


# ============================================================================
# Prompt formatting (pure function)
# ============================================================================


class TestPromptFormat(unittest.TestCase):
    def test_includes_action_and_risk(self):
        out = _format_prompt(_req(pid=1234))
        self.assertIn("process.kill", out)
        self.assertIn("risk: high", out)

    def test_includes_args(self):
        out = _format_prompt(_req(pid=1234))
        self.assertIn("pid", out)
        self.assertIn("1234", out)

    def test_no_args_shows_none(self):
        req = ConfirmationRequest(step=Step(action="process.list", risk_level="low"))
        out = _format_prompt(req)
        self.assertIn("(none)", out)

    def test_includes_rationale(self):
        out = _format_prompt(_req(pid=1))
        self.assertIn("agent thinks this is needed", out)

    def test_includes_expected_state(self):
        out = _format_prompt(_req(pid=1))
        self.assertIn("Expected", out)

    def test_ends_with_approve_question(self):
        out = _format_prompt(_req(pid=1))
        self.assertTrue(out.rstrip().endswith("Approve? [y/N]:"))

    def test_no_privilege_banner_for_normal_shell(self):
        from sentricoon.confirmation import ConfirmationRequest
        from sentricoon.state import Step
        req = ConfirmationRequest(
            step=Step(action="shell.run", args={"argv": ["find", "/tmp"]},
                      expected_state="x", risk_level="low"),
        )
        out = _format_prompt(req)
        self.assertNotIn("PRIVILEGE ESCALATION", out)

    def test_privilege_banner_for_sudo(self):
        from sentricoon.confirmation import ConfirmationRequest
        from sentricoon.state import Step
        req = ConfirmationRequest(
            step=Step(action="shell.run", args={"argv": ["sudo", "find", "/tmp"]},
                      expected_state="x", risk_level="medium"),
        )
        out = _format_prompt(req)
        self.assertIn("PRIVILEGE ESCALATION DETECTED", out)
        self.assertIn("'sudo'", out)
        self.assertIn("Deny if you did NOT authorize", out)

    def test_privilege_banner_for_runas(self):
        from sentricoon.confirmation import ConfirmationRequest
        from sentricoon.state import Step
        req = ConfirmationRequest(
            step=Step(action="shell.run",
                      args={"argv": ["runas", "/user:Administrator", "powershell"]},
                      expected_state="x", risk_level="high"),
        )
        out = _format_prompt(req)
        self.assertIn("PRIVILEGE ESCALATION DETECTED", out)
        self.assertIn("'runas'", out)

    def test_banner_case_insensitive_match(self):
        from sentricoon.confirmation import ConfirmationRequest
        from sentricoon.state import Step
        req = ConfirmationRequest(
            step=Step(action="shell.run", args={"argv": ["SUDO", "x"]},
                      expected_state="x", risk_level="low"),
        )
        self.assertIn("PRIVILEGE ESCALATION DETECTED", _format_prompt(req))

    def test_no_banner_for_non_shell_actions(self):
        from sentricoon.confirmation import ConfirmationRequest
        from sentricoon.state import Step
        # process.kill of pid 1234 — looks scary but isn't privilege escalation
        req = ConfirmationRequest(
            step=Step(action="process.kill", args={"pid": 1234},
                      expected_state="x", risk_level="high"),
        )
        out = _format_prompt(req)
        self.assertNotIn("PRIVILEGE ESCALATION", out)


# ============================================================================
# CLIConfirmationProvider — uses injected input/output
# ============================================================================


class _IO:
    """Captures output and feeds canned input."""

    def __init__(self, answer: str):
        self.answer = answer
        self.outputs: list[str] = []
        self.prompts: list[str] = []

    def input(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer

    def output(self, text: str) -> None:
        self.outputs.append(text)


class TestCLIProvider(unittest.TestCase):
    def test_y_grants_with_token(self):
        io_ = _IO("y")
        p = CLIConfirmationProvider(input_fn=io_.input, output_fn=io_.output)
        resp = p.request(_req(pid=99))
        self.assertTrue(resp.granted)
        self.assertIsNotNone(resp.token)
        self.assertGreaterEqual(len(resp.token), 16)

    def test_yes_grants(self):
        io_ = _IO("YES")  # case-insensitive
        resp = CLIConfirmationProvider(input_fn=io_.input, output_fn=io_.output).request(_req(pid=1))
        self.assertTrue(resp.granted)

    def test_n_denies(self):
        io_ = _IO("n")
        resp = CLIConfirmationProvider(input_fn=io_.input, output_fn=io_.output).request(_req(pid=1))
        self.assertFalse(resp.granted)
        self.assertIsNone(resp.token)

    def test_empty_input_denies(self):
        """Pressing Enter on a [y/N] prompt is denial. Default-no is the safe default."""
        io_ = _IO("")
        resp = CLIConfirmationProvider(input_fn=io_.input, output_fn=io_.output).request(_req(pid=1))
        self.assertFalse(resp.granted)

    def test_garbage_input_denies(self):
        io_ = _IO("maybe")
        resp = CLIConfirmationProvider(input_fn=io_.input, output_fn=io_.output).request(_req(pid=1))
        self.assertFalse(resp.granted)
        self.assertIn("unrecognized", resp.reason)

    def test_eof_treated_as_denial(self):
        def eof(prompt):
            raise EOFError
        resp = CLIConfirmationProvider(input_fn=eof, output_fn=lambda s: None).request(_req(pid=1))
        self.assertFalse(resp.granted)
        self.assertIn("EOF", resp.reason)

    def test_keyboard_interrupt_treated_as_denial(self):
        def interrupt(prompt):
            raise KeyboardInterrupt
        resp = CLIConfirmationProvider(input_fn=interrupt, output_fn=lambda s: None).request(_req(pid=1))
        self.assertFalse(resp.granted)
        self.assertIn("interrupted", resp.reason)

    def test_prompt_is_printed_to_output(self):
        io_ = _IO("n")
        CLIConfirmationProvider(input_fn=io_.input, output_fn=io_.output).request(_req(pid=42))
        printed = "\n".join(io_.outputs)
        self.assertIn("process.kill", printed)
        self.assertIn("42", printed)

    def test_uses_injected_token_factory(self):
        io_ = _IO("y")
        p = CLIConfirmationProvider(
            input_fn=io_.input, output_fn=io_.output, token_factory=lambda: "fixed-test-token-1234"
        )
        resp = p.request(_req(pid=1))
        self.assertEqual(resp.token, "fixed-test-token-1234")


# ============================================================================
# AutoApprove / AlwaysDeny
# ============================================================================


class TestAutoApproveProvider(unittest.TestCase):
    def test_always_grants(self):
        p = AutoApproveProvider()
        for _ in range(5):
            r = p.request(_req(pid=1))
            self.assertTrue(r.granted)
            self.assertIsNotNone(r.token)

    def test_distinct_tokens(self):
        p = AutoApproveProvider()
        tokens = {p.request(_req(pid=1)).token for _ in range(20)}
        self.assertEqual(len(tokens), 20)

    def test_satisfies_protocol(self):
        self.assertIsInstance(AutoApproveProvider(), ConfirmationProvider)


class TestAlwaysDenyProvider(unittest.TestCase):
    def test_always_denies(self):
        p = AlwaysDenyProvider()
        for _ in range(5):
            r = p.request(_req(pid=1))
            self.assertFalse(r.granted)
            self.assertIsNone(r.token)

    def test_satisfies_protocol(self):
        self.assertIsInstance(AlwaysDenyProvider(), ConfirmationProvider)


# ============================================================================
# End-to-end: confirmation → guard → executor → tool
# ============================================================================


class TestEndToEndConfirmationFlow(unittest.TestCase):
    def test_approve_flow_runs_step(self):
        provider = AutoApproveProvider()
        guard = SafetyGuard()
        tool = _NoOpKill()
        router = ToolRouter()
        router.register(tool)
        executor = Executor(router, guard)

        step = Step(action="process.kill", args={"pid": 1234},
                    expected_state="pid 1234 gone", risk_level="high")

        # Agent loop pattern: request confirmation, grant token, execute
        resp = provider.request(ConfirmationRequest(step=step))
        self.assertTrue(resp.granted)
        guard.grant_confirmation(resp.token, step.action)
        result = executor.execute(step, confirmation_token=resp.token)

        self.assertTrue(result.success)
        self.assertEqual(tool.calls, [{"pid": 1234}])

    def test_deny_flow_blocks_step(self):
        provider = AlwaysDenyProvider()
        guard = SafetyGuard()
        tool = _NoOpKill()
        router = ToolRouter()
        router.register(tool)
        executor = Executor(router, guard)

        step = Step(action="process.kill", args={"pid": 1234},
                    expected_state="gone", risk_level="high")

        resp = provider.request(ConfirmationRequest(step=step))
        self.assertFalse(resp.granted)
        # Agent loop should NOT execute when denied. If it tried with a fake
        # token, the guard blocks it:
        with self.assertRaises(GuardError):
            executor.execute(step, confirmation_token="fake-token")
        self.assertEqual(tool.calls, [])

    def test_token_cannot_be_reused_for_second_action(self):
        provider = AutoApproveProvider()
        guard = SafetyGuard()
        tool = _NoOpKill()
        router = ToolRouter()
        router.register(tool)
        executor = Executor(router, guard)

        step1 = Step(action="process.kill", args={"pid": 1}, expected_state="x", risk_level="high")
        step2 = Step(action="process.kill", args={"pid": 2}, expected_state="x", risk_level="high")

        # Approve once
        resp = provider.request(ConfirmationRequest(step=step1))
        guard.grant_confirmation(resp.token, step1.action)
        executor.execute(step1, confirmation_token=resp.token)

        # Try to chain the same token to a second high-risk action
        with self.assertRaisesRegex(GuardError, "already-used"):
            executor.execute(step2, confirmation_token=resp.token)
        self.assertEqual(tool.calls, [{"pid": 1}])  # only the first ran


if __name__ == "__main__":
    unittest.main()
