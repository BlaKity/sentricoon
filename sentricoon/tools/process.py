"""process.{list,kill} — read-only listing and gated kill.

`process.list` is implemented via stdlib (`os` + `/proc` on Linux,
`tasklist` on Windows). `process.kill` is a real action; the safety guard
must ensure a confirmation token is present before this tool is reached.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from .base import Tool, ToolResult


class ProcessListTool(Tool):
    name = "process.list"
    risk_level = "low"
    is_read_only = True

    def run(self, args: dict[str, Any]) -> ToolResult:
        if sys.platform == "win32":
            return self._win()
        return self._unix()

    @staticmethod
    def _win() -> ToolResult:
        try:
            cp = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                shell=False, check=False, capture_output=True, text=True, timeout=15,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            return ToolResult(success=False, error=str(e))
        if cp.returncode != 0:
            return ToolResult(success=False, error=cp.stderr or f"tasklist exit {cp.returncode}")
        rows = [line.strip('"').split('","') for line in cp.stdout.splitlines() if line.strip()]
        procs = [{"name": r[0], "pid": int(r[1])} for r in rows if len(r) >= 2 and r[1].isdigit()]
        return ToolResult(success=True, output=procs, extra={"count": len(procs)})

    @staticmethod
    def _unix() -> ToolResult:
        procs: list[dict] = []
        try:
            for pid_dir in os.listdir("/proc"):
                if not pid_dir.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid_dir}/comm", encoding="utf-8") as f:
                        procs.append({"pid": int(pid_dir), "name": f.read().strip()})
                except OSError:
                    continue
        except OSError as e:
            return ToolResult(success=False, error=str(e))
        return ToolResult(success=True, output=procs, extra={"count": len(procs)})


class ProcessKillTool(Tool):
    name = "process.kill"
    risk_level = "high"
    is_read_only = False

    def dry_run_describe(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"action": self.name, "would_kill_pid": args.get("pid")}

    def run(self, args: dict[str, Any]) -> ToolResult:
        pid = args.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return ToolResult(success=False, error="process.kill requires args['pid'] (positive int)")
        try:
            if sys.platform == "win32":
                cp = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    shell=False, check=False, capture_output=True, text=True, timeout=10,
                )
                if cp.returncode != 0:
                    return ToolResult(success=False, error=cp.stderr or f"taskkill exit {cp.returncode}")
            else:
                import signal
                os.kill(pid, signal.SIGTERM)
        except (OSError, subprocess.TimeoutExpired) as e:
            return ToolResult(success=False, error=str(e))
        return ToolResult(success=True, output={"killed": pid})
