from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sentricoon.tools.file_ops import FileDeleteTool, FileReadTool, FileWriteTool
from sentricoon.tools.shell import ShellTool
from sentricoon.tools.system import SystemInfoTool
from sentricoon.tools.ubuntu_systemd import (
    UbuntuSystemdFailedUnitsTool,
    UbuntuSystemdRestartTool,
    UbuntuSystemdStatusTool,
)


class TestShellTool(unittest.TestCase):
    def test_rejects_string_command(self):
        # Critical safety check — string args mean shell injection risk
        r = ShellTool().run({"argv": "echo hi"})
        self.assertFalse(r.success)
        self.assertIn("argv", r.error or "")

    def test_rejects_missing_argv(self):
        r = ShellTool().run({})
        self.assertFalse(r.success)

    def test_runs_simple_command(self):
        cmd = ["cmd.exe", "/c", "echo hi"] if sys.platform == "win32" else ["echo", "hi"]
        r = ShellTool().run({"argv": cmd})
        self.assertTrue(r.success, msg=r.error)
        self.assertIn("hi", r.output["stdout"])
        self.assertEqual(r.extra["returncode"], 0)

    def test_command_not_found(self):
        r = ShellTool().run({"argv": ["definitely-not-a-real-binary-9999"]})
        self.assertFalse(r.success)

    def test_timeout(self):
        if sys.platform == "win32":
            self.skipTest("sleep isn't portable on Windows shell")
        r = ShellTool(timeout_s=0.1).run({"argv": ["sleep", "2"]})
        self.assertFalse(r.success)
        self.assertIn("timeout", (r.error or "").lower())


class TestFileTools(unittest.TestCase):
    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "x.txt")
            w = FileWriteTool().run({"path": path, "content": "hello"})
            self.assertTrue(w.success, msg=w.error)
            r = FileReadTool().run({"path": path})
            self.assertTrue(r.success, msg=r.error)
            self.assertEqual(r.output, "hello")

    def test_read_missing_path(self):
        r = FileReadTool().run({"path": "/definitely/not/a/path/xyz123"})
        self.assertFalse(r.success)

    def test_delete_then_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y.txt"
            path.write_text("z")
            d = FileDeleteTool().run({"path": str(path)})
            self.assertTrue(d.success)
            d2 = FileDeleteTool().run({"path": str(path)})
            self.assertFalse(d2.success)

    def test_write_rejects_non_string(self):
        r = FileWriteTool().run({"path": "/tmp/x.txt", "content": 123})
        self.assertFalse(r.success)


class TestSystemInfoTool(unittest.TestCase):
    def test_returns_keys(self):
        r = SystemInfoTool().run({})
        self.assertTrue(r.success)
        for key in ("platform", "system", "python", "cwd"):
            self.assertIn(key, r.output)


class TestUbuntuSystemdTools(unittest.TestCase):
    @patch("sentricoon.tools.ubuntu_systemd.subprocess.run")
    def test_failed_units_runs_narrow_systemctl_command(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = "0 loaded units listed.\n"
        run.return_value.stderr = ""

        r = UbuntuSystemdFailedUnitsTool().run({})

        self.assertTrue(r.success, msg=r.error)
        self.assertEqual(
            run.call_args.args[0],
            ["systemctl", "list-units", "--failed", "--no-pager", "--plain"],
        )
        self.assertIn("stdout", r.output)

    @patch("sentricoon.tools.ubuntu_systemd.subprocess.run")
    def test_status_requires_unit_and_runs_status(self, run):
        missing = UbuntuSystemdStatusTool().run({})
        self.assertFalse(missing.success)

        run.return_value.returncode = 0
        run.return_value.stdout = "Active: active (running)\n"
        run.return_value.stderr = ""

        r = UbuntuSystemdStatusTool().run({"unit": "ssh.service"})

        self.assertTrue(r.success, msg=r.error)
        self.assertEqual(
            run.call_args.args[0],
            ["systemctl", "status", "ssh.service", "--no-pager", "--plain"],
        )

    @patch("sentricoon.tools.ubuntu_systemd.subprocess.run")
    def test_status_treats_failed_unit_as_successful_inspection(self, run):
        run.return_value.returncode = 3
        run.return_value.stdout = "Active: failed (Result: exit-code)\n"
        run.return_value.stderr = ""

        r = UbuntuSystemdStatusTool().run({"unit": "broken.service"})

        self.assertTrue(r.success, msg=r.error)
        self.assertEqual(r.extra["returncode"], 3)
        self.assertIn("failed", r.output["stdout"])

    @patch("sentricoon.tools.ubuntu_systemd.subprocess.run")
    def test_restart_requires_unit_and_runs_restart(self, run):
        missing = UbuntuSystemdRestartTool().run({})
        self.assertFalse(missing.success)

        run.return_value.returncode = 0
        run.return_value.stdout = ""
        run.return_value.stderr = ""

        r = UbuntuSystemdRestartTool().run({"unit": "ssh.service"})

        self.assertTrue(r.success, msg=r.error)
        self.assertEqual(run.call_args.args[0], ["systemctl", "restart", "ssh.service"])

    def test_restart_dry_run_describes_action(self):
        d = UbuntuSystemdRestartTool().dry_run_describe({"unit": "ssh.service"})
        self.assertEqual(d["action"], "ubuntu.systemd.restart")
        self.assertEqual(d["would_run"], ["systemctl", "restart", "ssh.service"])


if __name__ == "__main__":
    unittest.main()
