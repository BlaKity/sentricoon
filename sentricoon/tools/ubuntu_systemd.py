"""Ubuntu systemd tools.

These are intentionally narrow, named actions rather than a generic shell
escape hatch. Read-only actions expose common diagnostics directly; restart is
write-effect and must pass through confirmation like other risky actions.
"""

from __future__ import annotations

import subprocess
from typing import Any

from .base import Tool, ToolResult


def _run_systemctl(
    argv: list[str],
    *,
    timeout_s: float = 15.0,
    success_returncodes: set[int] | None = None,
) -> ToolResult:
    try:
        cp = subprocess.run(
            ["systemctl", *argv],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        return ToolResult(success=False, error=f"timeout after {e.timeout}s")
    except FileNotFoundError:
        return ToolResult(success=False, error="systemctl not found")
    except OSError as e:
        return ToolResult(success=False, error=f"os error: {e}")

    success_returncodes = success_returncodes or {0}
    success = cp.returncode in success_returncodes
    return ToolResult(
        success=success,
        output={"stdout": cp.stdout, "stderr": cp.stderr},
        error=(cp.stderr.strip() or f"systemctl exit {cp.returncode}") if not success else None,
        extra={"returncode": cp.returncode, "argv": ["systemctl", *argv]},
    )


class UbuntuSystemdFailedUnitsTool(Tool):
    name = "ubuntu.systemd.failed_units"
    risk_level = "low"
    is_read_only = True

    def run(self, args: dict[str, Any]) -> ToolResult:
        return _run_systemctl(
            ["list-units", "--failed", "--no-pager", "--plain"],
            timeout_s=float(args.get("timeout_s", 15.0)),
        )


class UbuntuSystemdStatusTool(Tool):
    name = "ubuntu.systemd.status"
    risk_level = "low"
    is_read_only = True

    def run(self, args: dict[str, Any]) -> ToolResult:
        unit = args.get("unit")
        if not isinstance(unit, str) or not unit.strip():
            return ToolResult(success=False, error="ubuntu.systemd.status requires args['unit']")
        return _run_systemctl(
            ["status", unit, "--no-pager", "--plain"],
            timeout_s=float(args.get("timeout_s", 15.0)),
            # systemctl status returns 3 for inactive/failed units. For this
            # read-only diagnostic tool, that is still a successful inspection:
            # the output is exactly the evidence the agent needs.
            success_returncodes={0, 3},
        )


class UbuntuSystemdRestartTool(Tool):
    name = "ubuntu.systemd.restart"
    risk_level = "high"
    is_read_only = False

    def dry_run_describe(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "action": self.name,
            "unit": args.get("unit"),
            "would_run": ["systemctl", "restart", args.get("unit")],
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        unit = args.get("unit")
        if not isinstance(unit, str) or not unit.strip():
            return ToolResult(success=False, error="ubuntu.systemd.restart requires args['unit']")
        return _run_systemctl(
            ["restart", unit],
            timeout_s=float(args.get("timeout_s", 30.0)),
        )
