"""shell.run — runs a command via subprocess with args list (NEVER a string).

Critical: `shell=False`, args must be a list. Passing a string with
shell=True would re-introduce the shell-injection vector that
Share 2's `os.popen` example had.
"""

from __future__ import annotations

import subprocess
from typing import Any

from .base import Tool, ToolResult


class ShellTool(Tool):
    name = "shell.run"
    risk_level = "high"
    is_read_only = False

    def __init__(self, timeout_s: float = 30.0) -> None:
        self.timeout_s = timeout_s

    def dry_run_describe(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "action": self.name,
            "argv": args.get("argv"),
            "timeout_s": args.get("timeout_s", self.timeout_s),
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        argv = args.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            return ToolResult(
                success=False,
                error="shell.run requires args['argv'] = [list, of, strings] — never a raw command string",
            )
        try:
            cp = subprocess.run(
                argv,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.get("timeout_s", self.timeout_s),
            )
        except subprocess.TimeoutExpired as e:
            return ToolResult(success=False, error=f"timeout after {e.timeout}s")
        except FileNotFoundError as e:
            return ToolResult(success=False, error=f"command not found: {e.filename}")
        except OSError as e:
            return ToolResult(success=False, error=f"os error: {e}")

        return ToolResult(
            success=cp.returncode == 0,
            output={"stdout": cp.stdout, "stderr": cp.stderr},
            extra={"returncode": cp.returncode, "argv": argv},
        )
