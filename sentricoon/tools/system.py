"""system.info — read-only system inspection (no actions, no LLM)."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

from .base import Tool, ToolResult


class SystemInfoTool(Tool):
    name = "system.info"
    risk_level = "low"
    is_read_only = True

    def run(self, args: dict[str, Any]) -> ToolResult:
        info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "cwd": os.getcwd(),
        }
        return ToolResult(success=True, output=info)
