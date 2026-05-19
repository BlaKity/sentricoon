"""LLM-facing tool catalog — the schema the planner sees.

Hand-curated, NOT auto-derived from Tool classes. The catalog is the
contract between the planner LLM and the executor: it stays stable across
tool implementation refactors. Adding a new action means updating two
places — this catalog AND [[project-goals-and-structure]] § "Action
allowlist". The allowlist (`safety/policies.py`) is "what the executor
permits"; this is "what the LLM is told it can use."
"""

from __future__ import annotations

from typing import Final

TOOL_CATALOG: Final[list[dict]] = [
    {
        "name": "system.info",
        "description": (
            "Read static system information. Returns a dict with keys: "
            "platform, system (Linux | Windows | Darwin), release (kernel version, "
            "e.g. '6.8.0-111-generic' or '10.0.19045'), machine (architecture), "
            "processor, python, cwd. "
            "USE FOR any task asking about: OS / OS family, kernel version, "
            "architecture, or current directory — those answers are ALREADY in this "
            "result; do NOT also run shell.run uname/ver to re-fetch them."
        ),
        "args": {},
        "risk": "low",
        "requires_confirmation": False,
    },
    {
        "name": "process.list",
        "description": "List running processes. Returns [{pid, name}, ...].",
        "args": {},
        "risk": "low",
        "requires_confirmation": False,
    },
    {
        "name": "process.kill",
        "description": "Terminate a process by pid. Verify with process.list afterwards.",
        "args": {"pid": "int — positive process ID"},
        "risk": "high",
        "requires_confirmation": True,
    },
    {
        "name": "file.read",
        "description": "Read text content from a file path.",
        "args": {"path": "str — absolute or relative path", "encoding": "str, optional, default utf-8"},
        "risk": "low",
        "requires_confirmation": False,
    },
    {
        "name": "file.write",
        "description": "Write text content to a file path (creates parent dirs).",
        "args": {"path": "str", "content": "str", "encoding": "str, optional, default utf-8"},
        "risk": "medium",
        "requires_confirmation": False,
    },
    {
        "name": "file.delete",
        "description": "Delete a file at path. Fails if path is missing.",
        "args": {"path": "str"},
        "risk": "high",
        "requires_confirmation": True,
    },
    {
        "name": "shell.run",
        "description": (
            "Run a command. argv MUST be a list of strings (e.g. ['net','start','Audiosrv']); "
            "passing a single string will be rejected as it would re-introduce shell-injection risk."
        ),
        "args": {
            "argv": "list[str] — the command as separate tokens",
            "timeout_s": "float, optional, default 30.0",
        },
        "risk": "high",
        "requires_confirmation": True,
    },
    {
        "name": "app.open",
        "description": "Open a desktop application by name or path.",
        "args": {"target": "str — application name or path"},
        "risk": "medium",
        "requires_confirmation": False,
    },
    {
        "name": "browser.open",
        "description": "Open a URL in the default browser.",
        "args": {"url": "str"},
        "risk": "low",
        "requires_confirmation": False,
    },
    {
        "name": "ubuntu.systemd.failed_units",
        "supported_os": ["Linux"],
        "description": (
            "Ubuntu/Linux diagnostic: list failed systemd units using "
            "`systemctl list-units --failed --no-pager --plain`. Prefer this "
            "over shell.run for tasks asking about failed services."
        ),
        "args": {"timeout_s": "float, optional, default 15.0"},
        "risk": "low",
        "requires_confirmation": False,
    },
    {
        "name": "ubuntu.systemd.status",
        "supported_os": ["Linux"],
        "description": (
            "Ubuntu/Linux diagnostic: read the status of one systemd unit. "
            "Use for service status checks before attempting a restart."
        ),
        "args": {"unit": "str — service/unit name, e.g. 'ssh.service'", "timeout_s": "float, optional"},
        "risk": "low",
        "requires_confirmation": False,
    },
    {
        "name": "ubuntu.systemd.restart",
        "supported_os": ["Linux"],
        "description": (
            "Ubuntu/Linux repair: restart one systemd unit. Diagnose first "
            "with ubuntu.systemd.status or ubuntu.systemd.failed_units, then "
            "verify with ubuntu.systemd.status after restart."
        ),
        "args": {"unit": "str — service/unit name, e.g. 'ssh.service'", "timeout_s": "float, optional"},
        "risk": "high",
        "requires_confirmation": True,
    },
]


def catalog_for_prompt(system: str | None = None) -> str:
    """Markdown-ish catalog block for inclusion in the planner's system prompt."""
    lines: list[str] = []
    for entry in TOOL_CATALOG:
        supported = entry.get("supported_os")
        if system is not None and supported is not None and system not in supported:
            continue
        lines.append(f"### {entry['name']}  (risk: {entry['risk']})")
        lines.append(entry["description"])
        if entry["args"]:
            lines.append("Args:")
            for arg, desc in entry["args"].items():
                lines.append(f"  - {arg}: {desc}")
        else:
            lines.append("Args: (none)")
        if entry["requires_confirmation"]:
            lines.append("Note: requires explicit user confirmation before execution.")
        lines.append("")
    return "\n".join(lines).rstrip()
