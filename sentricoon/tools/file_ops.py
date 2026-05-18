"""file.{read,write,delete} — small, deterministic, no shell."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tool, ToolResult


class FileReadTool(Tool):
    name = "file.read"
    risk_level = "low"
    is_read_only = True

    def run(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        if not isinstance(path, str):
            return ToolResult(success=False, error="file.read requires args['path'] (str)")
        try:
            data = Path(path).read_text(encoding=args.get("encoding", "utf-8"))
        except OSError as e:
            return ToolResult(success=False, error=str(e))
        return ToolResult(success=True, output=data, extra={"path": path, "bytes": len(data)})


class FileWriteTool(Tool):
    name = "file.write"
    risk_level = "medium"
    is_read_only = False

    def dry_run_describe(self, args: dict[str, Any]) -> dict[str, Any]:
        content = args.get("content")
        return {
            "action": self.name,
            "path": args.get("path"),
            "bytes": len(content) if isinstance(content, str) else None,
            "would_overwrite": (
                Path(args["path"]).exists()
                if isinstance(args.get("path"), str) else None
            ),
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            return ToolResult(success=False, error="file.write requires args['path'] (str) and ['content'] (str)")
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding=args.get("encoding", "utf-8"))
        except OSError as e:
            return ToolResult(success=False, error=str(e))
        return ToolResult(success=True, output={"path": path}, extra={"bytes": len(content)})


class FileDeleteTool(Tool):
    name = "file.delete"
    risk_level = "high"
    is_read_only = False

    def dry_run_describe(self, args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path")
        return {
            "action": self.name,
            "path": path,
            "exists_now": (Path(path).exists() if isinstance(path, str) else None),
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        if not isinstance(path, str):
            return ToolResult(success=False, error="file.delete requires args['path'] (str)")
        p = Path(path)
        if not p.exists():
            return ToolResult(success=False, error=f"path does not exist: {path}")
        try:
            p.unlink()
        except OSError as e:
            return ToolResult(success=False, error=str(e))
        return ToolResult(success=True, output={"deleted": path})
