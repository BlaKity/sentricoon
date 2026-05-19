from __future__ import annotations

import unittest

from sentricoon.router import RouterError, ToolRouter
from sentricoon.safety.guard import GuardError, SafetyGuard
from sentricoon.safety.policies import ACTION_ALLOWLIST, REQUIRES_CONFIRMATION
from sentricoon.state import Step
from sentricoon.tools.base import Tool, ToolResult


class _OkTool(Tool):
    name = "system.info"
    risk_level = "low"

    def run(self, args):
        return ToolResult(success=True, output={"info": "ok"})


class _RogueTool(Tool):
    name = "obviously.not.allowlisted"
    risk_level = "low"

    def run(self, args):
        return ToolResult(success=True)


class TestSafetyGuard(unittest.TestCase):
    def test_allowed_action_passes(self):
        SafetyGuard().validate(Step(action="system.info"))

    def test_rejected_when_not_in_allowlist(self):
        with self.assertRaises(GuardError):
            SafetyGuard().validate(Step(action="nope.bad"))

    def test_empty_action_rejected(self):
        with self.assertRaises(GuardError):
            SafetyGuard().validate(Step(action=""))

    def test_confirmation_required_for_high_risk(self):
        g = SafetyGuard()
        for action in REQUIRES_CONFIRMATION:
            with self.assertRaises(GuardError):
                g.validate(Step(action=action), confirmation_token=None)

    def test_confirmation_granted_unlocks(self):
        g = SafetyGuard()
        g.grant_confirmation("tok-123", "process.kill")
        g.validate(Step(action="process.kill", args={"pid": 1234}), confirmation_token="tok-123")

    def test_confirmation_token_must_be_granted(self):
        g = SafetyGuard()
        with self.assertRaises(GuardError):
            g.validate(Step(action="shell.run"), confirmation_token="fake")

    def test_confirmation_token_is_single_use(self):
        g = SafetyGuard()
        g.grant_confirmation("tok-once", "process.kill")
        g.validate(Step(action="process.kill", args={"pid": 1}), confirmation_token="tok-once")
        # Reusing the same token must fail — single-use is the safety story
        with self.assertRaisesRegex(GuardError, "already-used"):
            g.validate(Step(action="process.kill", args={"pid": 2}), confirmation_token="tok-once")

    def test_token_bound_to_specific_action(self):
        g = SafetyGuard()
        # A token granted for process.kill must NOT authorize file.delete
        g.grant_confirmation("tok-kill", "process.kill")
        with self.assertRaisesRegex(GuardError, "authorizes action"):
            g.validate(Step(action="file.delete", args={"path": "/tmp/x"}), confirmation_token="tok-kill")
        # Token wasn't consumed by the mismatch; still usable for its action
        self.assertTrue(g.has_pending("tok-kill"))

    def test_grant_rejects_empty_token(self):
        with self.assertRaises(GuardError):
            SafetyGuard().grant_confirmation("", "shell.run")

    def test_grant_rejects_empty_action(self):
        with self.assertRaises(GuardError):
            SafetyGuard().grant_confirmation("tok", "")

    def test_low_risk_actions_dont_require_token(self):
        # mcp-style or read-only actions not in REQUIRES_CONFIRMATION
        SafetyGuard().validate(Step(action="system.info"))
        SafetyGuard().validate(Step(action="file.read", args={"path": "/x"}))
        SafetyGuard().validate(Step(action="process.list"))

    def test_has_pending_is_inspection_only(self):
        g = SafetyGuard()
        g.grant_confirmation("inspect-me", "shell.run")
        self.assertTrue(g.has_pending("inspect-me"))
        # has_pending must NOT consume
        self.assertTrue(g.has_pending("inspect-me"))


class TestToolRouter(unittest.TestCase):
    def test_register_and_get(self):
        router = ToolRouter()
        router.register(_OkTool())
        tool = router.get("system.info")
        self.assertIsInstance(tool, _OkTool)

    def test_register_rejects_non_allowlisted(self):
        with self.assertRaises(RouterError):
            ToolRouter().register(_RogueTool())

    def test_get_rejects_non_allowlisted(self):
        with self.assertRaises(RouterError):
            ToolRouter().get("nope.bad")

    def test_get_missing_registration(self):
        with self.assertRaises(RouterError):
            ToolRouter().get("system.info")

    def test_allowlist_unchanged(self):
        # Sanity check: catch accidental allowlist edits
        self.assertEqual(
            set(ACTION_ALLOWLIST),
            {
                "shell.run", "file.read", "file.write", "file.delete",
                "process.list", "process.kill", "system.info",
                "app.open", "browser.open",
                "ubuntu.systemd.failed_units", "ubuntu.systemd.status",
                "ubuntu.systemd.restart",
            },
        )

    def test_ubuntu_systemd_restart_requires_confirmation(self):
        self.assertIn("ubuntu.systemd.restart", REQUIRES_CONFIRMATION)
        with self.assertRaises(GuardError):
            SafetyGuard().validate(Step(action="ubuntu.systemd.restart", args={"unit": "ssh.service"}))


if __name__ == "__main__":
    unittest.main()
