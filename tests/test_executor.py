from __future__ import annotations

import unittest

from sentricoon.executor import Executor
from sentricoon.router import ToolRouter
from sentricoon.safety.guard import GuardError, SafetyGuard
from sentricoon.state import Step
from sentricoon.tools.system import SystemInfoTool


class TestExecutor(unittest.TestCase):
    def _make(self):
        guard = SafetyGuard()
        router = ToolRouter()
        router.register(SystemInfoTool())
        return Executor(router, guard)

    def test_executes_allowed_step(self):
        ex = self._make()
        r = ex.execute(Step(action="system.info"))
        self.assertTrue(r.success)

    def test_blocks_disallowed_step(self):
        ex = self._make()
        with self.assertRaises(GuardError):
            ex.execute(Step(action="not.allowed"))

    def test_blocks_unregistered_allowlisted(self):
        ex = self._make()
        # 'shell.run' is in allowlist but not registered with router
        with self.assertRaises(Exception):
            ex.execute(Step(action="shell.run"), confirmation_token=None)


if __name__ == "__main__":
    unittest.main()
