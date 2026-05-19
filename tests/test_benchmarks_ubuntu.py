"""Ubuntu benchmark scenarios.

These are not performance benchmarks. They are behavior benchmarks for the
project's north star: diagnose safely, avoid mutation by default, verify
through named tools, and keep OS-specific actions out of the wrong prompt.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from sentricoon.catalog import catalog_for_prompt
from sentricoon.executor import Executor
from sentricoon.main import build_tool_router
from sentricoon.safety.guard import GuardError, SafetyGuard
from sentricoon.state import Step
from sentricoon.tools.ubuntu_systemd import UbuntuSystemdStatusTool


class TestUbuntuSystemdBenchmarks(unittest.TestCase):
    def test_failed_unit_status_is_successful_diagnostic_evidence(self):
        """Benchmark: a failed service should be reportable, not a tool error.

        `systemctl status` uses exit code 3 for failed/inactive units. The
        diagnostic action should still succeed so the agent can use the output
        as evidence instead of entering a repair loop just because the unit is
        failed.
        """
        with patch("sentricoon.tools.ubuntu_systemd.subprocess.run") as run:
            run.return_value.returncode = 3
            run.return_value.stdout = (
                "x demo-failed.service - Demo Failed Service\n"
                "     Active: failed (Result: exit-code)\n"
            )
            run.return_value.stderr = ""

            result = UbuntuSystemdStatusTool().run({"unit": "demo-failed.service"})

        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.extra["returncode"], 3)
        self.assertIn("Active: failed", result.output["stdout"])

    def test_restart_dry_run_never_mutates(self):
        """Benchmark: dry-run previews service restart without confirmation."""
        executor = Executor(build_tool_router(), SafetyGuard(), dry_run=True)

        result = executor.execute(Step(
            action="ubuntu.systemd.restart",
            args={"unit": "demo.service"},
            expected_state="demo.service restart would be previewed without mutation",
        ))

        self.assertTrue(result.success)
        self.assertEqual(result.extra["simulated"], True)
        self.assertEqual(
            result.output["would_do"]["would_run"],
            ["systemctl", "restart", "demo.service"],
        )

    def test_restart_requires_confirmation_in_real_mode(self):
        """Benchmark: service restart cannot run without an approval token."""
        executor = Executor(build_tool_router(), SafetyGuard())

        with self.assertRaises(GuardError):
            executor.execute(Step(
                action="ubuntu.systemd.restart",
                args={"unit": "demo.service"},
                expected_state="demo.service is restarted",
            ))

    def test_catalog_is_os_specific(self):
        """Benchmark: Ubuntu actions are not leaked into Windows prompts."""
        linux = catalog_for_prompt(system="Linux")
        windows = catalog_for_prompt(system="Windows")

        self.assertIn("ubuntu.systemd.failed_units", linux)
        self.assertIn("ubuntu.systemd.status", linux)
        self.assertIn("ubuntu.systemd.restart", linux)
        self.assertNotIn("ubuntu.systemd.", windows)
        self.assertNotIn("systemctl", windows)


if __name__ == "__main__":
    unittest.main()
