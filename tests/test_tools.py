from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from sentricoon.tools.file_ops import FileDeleteTool, FileReadTool, FileWriteTool
from sentricoon.tools.shell import ShellTool
from sentricoon.tools.system import SystemInfoTool


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


if __name__ == "__main__":
    unittest.main()
