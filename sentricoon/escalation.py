"""Escalation / failure-handling layer.

This module owns what happens when the agent *cannot* fix the user's
problem. Step-level retries belong in `repair.py` (not yet built).
This layer covers six concerns:

  1. Task-level budget (independent of per-step retry counts).
  2. Strategy-exhaustion detection (planner emits the same shape twice).
  3. Partial-progress tracking (what we ruled out, what we observed).
  4. Structured failure report for human handoff.
  5. Quarantine hook (strategies known to fail — feeds future procedural memory).
  6. Rollback manifest (paths to pre-action state snapshots).

The memory persistence layer is intentionally not wired here. The
escalation manager accepts an optional `quarantine_source` callable so
the procedural-memory layer can plug in later without changing this code.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from .state import Step, StepRecord, TaskState, step_record_to_dict


class EscalationReason(str, Enum):
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIME_EXHAUSTED = "time_exhausted"
    STRATEGY_EXHAUSTED = "strategy_exhausted"
    STRATEGY_LOOP = "strategy_loop"
    ABORT_REQUESTED = "abort_requested"
    UNRECOVERABLE_ERROR = "unrecoverable_error"


@dataclass(frozen=True)
class TaskBudget:
    max_total_attempts: int = 20
    max_wall_time_s: float = 600.0
    max_distinct_strategies: int = 5


@dataclass
class FailureReport:
    task: str
    goal_class: str | None
    reason: EscalationReason
    summary: str
    started_at: float
    ended_at: float
    attempts: int
    strategies_tried: list[str]
    observations: list[str]
    ruled_out: list[str]
    next_manual_actions: list[str]
    snapshots: list[str]
    history: list[dict]

    def to_json(self) -> str:
        d = asdict(self)
        d["reason"] = self.reason.value
        return json.dumps(d, indent=2, default=str)

    def to_markdown(self) -> str:
        lines = [
            f"# Failure report — {self.task}",
            "",
            f"- Reason: **{self.reason.value}**",
            f"- Goal class: `{self.goal_class or 'unknown'}`",
            f"- Attempts: {self.attempts}",
            f"- Wall time: {self.ended_at - self.started_at:.1f}s",
            f"- Strategies tried: {len(self.strategies_tried)}",
            "",
            "## Summary",
            self.summary,
            "",
        ]
        lines += _attempt_log_section(self.history)
        if self.observations:
            lines += ["## What the agent observed", *(f"- {o}" for o in self.observations), ""]
        if self.ruled_out:
            lines += ["## Ruled out", *(f"- {r}" for r in self.ruled_out), ""]
        if self.next_manual_actions:
            lines += [
                "## Suggested manual next steps",
                *(f"- {a}" for a in self.next_manual_actions),
                "",
            ]
        if self.snapshots:
            lines += [
                "## State snapshots (for rollback)",
                *(f"- `{s}`" for s in self.snapshots),
                "",
            ]
        return "\n".join(lines)


# ============================================================================
# Markdown helpers (private)
# ============================================================================


_TABLE_CELL_LIMIT = 80


def _md_escape(s: str) -> str:
    """Escape pipes and collapse newlines for safe markdown-table cells."""
    return s.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _args_brief(action: str, args: dict) -> str:
    if action == "shell.run":
        argv = args.get("argv")
        if isinstance(argv, list):
            return " ".join(str(a) for a in argv)
        return ""
    if action in {"file.read", "file.write", "file.delete"}:
        return str(args.get("path", ""))
    if action == "process.kill":
        return f"pid={args.get('pid')}"
    return ""


def _result_brief(rec: dict) -> str:
    success = rec.get("success")
    verif = rec.get("verification") or {}
    if success:
        reason = verif.get("reason") or "ok"
        return f"OK ({reason})"
    err = rec.get("error")
    msg = verif.get("reason") or err or "failed"
    return f"FAIL: {msg}"


def _truncate_for_cell(s: str) -> str:
    s = _md_escape(s)
    if len(s) <= _TABLE_CELL_LIMIT:
        return s
    return s[: _TABLE_CELL_LIMIT - 3] + "..."


def _attempt_log_section(history: list[dict]) -> list[str]:
    """Render the per-attempt history as a markdown table.

    Empty list when history is empty (so the section is omitted).
    """
    if not history:
        return []
    lines = [
        "## Attempt log",
        "",
        "| # | Action | Args | Result |",
        "|---|--------|------|--------|",
    ]
    for i, rec in enumerate(history, start=1):
        action = rec.get("action", "?")
        args_brief = _args_brief(action, rec.get("args") or {})
        result = _result_brief(rec)
        lines.append(
            f"| {i} | {_md_escape(action)} | "
            f"{_truncate_for_cell(args_brief)} | "
            f"{_truncate_for_cell(result)} |"
        )
    lines.append("")
    return lines


def strategy_fingerprint(plan: list[Step]) -> str:
    """Stable hash of a plan's action sequence.

    Two plans with the same action sequence but different argument values
    share a fingerprint — the planner is trying the same approach again.
    Args are deliberately excluded; arg-level variation is what `fix_input`
    repair is for, not a new strategy.
    """
    payload = "|".join(s.action for s in plan)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def is_repeat_strategy(state: TaskState, plan: list[Step]) -> bool:
    return strategy_fingerprint(plan) in state.strategies_tried


def record_strategy(state: TaskState, plan: list[Step]) -> str:
    fp = strategy_fingerprint(plan)
    if fp not in state.strategies_tried:
        state.strategies_tried.append(fp)
    return fp


def should_escalate(
    state: TaskState,
    budget: TaskBudget,
    *,
    abort_requested: bool = False,
    quarantine_source: Callable[[str], bool] | None = None,
) -> EscalationReason | None:
    """Return an EscalationReason if the task must stop, else None.

    Checked in priority order — explicit abort > budgets > strategy state.
    Callers run this after every step (and after every replan).
    """
    if abort_requested:
        return EscalationReason.ABORT_REQUESTED

    if state.attempts() >= budget.max_total_attempts:
        return EscalationReason.BUDGET_EXHAUSTED

    if state.elapsed_s() >= budget.max_wall_time_s:
        return EscalationReason.TIME_EXHAUSTED

    if len(state.strategies_tried) >= budget.max_distinct_strategies:
        return EscalationReason.STRATEGY_EXHAUSTED

    if _detect_strategy_loop(state):
        return EscalationReason.STRATEGY_LOOP

    if quarantine_source is not None and state.plan:
        fp = strategy_fingerprint(state.plan)
        if quarantine_source(fp):
            return EscalationReason.STRATEGY_EXHAUSTED

    return None


def _detect_strategy_loop(state: TaskState) -> bool:
    """A strategy 'loops' when the same fingerprint reappears with no
    new failed-step evidence between the two occurrences.

    We check the last 4 strategies — if the most recent fingerprint
    already appeared earlier in that window without intervening progress,
    that's the planner reasoning in a circle.
    """
    if len(state.strategies_tried) < 2:
        return False
    window = state.strategies_tried[-4:]
    last = window[-1]
    return last in window[:-1]


def extract_observations(history: list[StepRecord]) -> tuple[list[str], list[str]]:
    """Pull (observations, ruled_out) out of step history.

    Observations: things the verifier or step output established as TRUE.
    Ruled_out: things diagnoses tagged as eliminated causes.
    """
    observations: list[str] = []
    ruled_out: list[str] = []
    for record in history:
        if record.success and record.verification:
            reason = record.verification.get("reason")
            if reason:
                observations.append(f"[{record.step.action}] {reason}")
        if record.diagnosis:
            cause = record.diagnosis.get("ruled_out")
            if cause:
                ruled_out.append(str(cause))
            evidence = record.diagnosis.get("evidence")
            if evidence:
                observations.append(str(evidence))
    return observations, ruled_out


def suggest_manual_actions(
    state: TaskState,
    reason: EscalationReason,
    observations: list[str],
) -> list[str]:
    """Heuristic suggestions for the human.

    A future critic-LLM call can replace this with model-generated advice,
    but the heuristic gives non-empty output even when no LLM is available.
    """
    suggestions: list[str] = []

    if reason == EscalationReason.TIME_EXHAUSTED:
        suggestions.append(
            "Increase max_wall_time_s if the agent was making progress, "
            "or kill the task — it may be stuck waiting on a hung subprocess."
        )
    if reason == EscalationReason.BUDGET_EXHAUSTED:
        suggestions.append(
            "Review the run log — the agent burned its attempt budget. "
            "Likely cause: a step is failing verification on a check that "
            "doesn't reflect actual goal progress."
        )
    if reason in (EscalationReason.STRATEGY_EXHAUSTED, EscalationReason.STRATEGY_LOOP):
        suggestions.append(
            "The planner ran out of distinct approaches. Provide the agent "
            "with more context (recent system changes, error messages it "
            "didn't see) and re-run, or attempt manually."
        )
    if reason == EscalationReason.ABORT_REQUESTED:
        suggestions.append(
            "Agent aborted due to a high-risk condition. Inspect the last "
            "diagnosis in the report before retrying."
        )

    if state.goal_class == "audio_no_sound":
        suggestions.append(
            "Manual audio checklist: (1) Settings → System → Sound → Output: "
            "verify the right device is default. (2) `services.msc`: ensure "
            "'Windows Audio' and 'Windows Audio Endpoint Builder' are Running. "
            "(3) Device Manager → Sound: check for yellow exclamation marks."
        )

    if state.goal_class == "audio_routing":
        suggestions.append(
            "Most common cause on Windows: Realtek HD Audio Manager → "
            "Device Advanced Settings → uncheck 'Multi-streaming mode'. "
            "Apply, then unplug and re-plug the earphones to test."
        )
        suggestions.append(
            "Alternative: Settings → System → Sound → choose the earphone "
            "device as Default, OR right-click the speaker icon in the system "
            "tray → Open Sound settings → manage devices."
        )
        suggestions.append(
            "Advanced (back up the registry first): inspect "
            "`HKLM\\SOFTWARE\\Realtek\\Audio\\HDA` for a multistream-related "
            "REG_DWORD; setting it to 0 disables concurrent output."
        )

    if state.goal_class == "disk_audit":
        suggestions.append(
            "For partial results: check the run JSONL — find/du commands "
            "often emit accessible-file paths to stdout even when they hit "
            "permission errors on restricted dirs. The answer may already "
            "be in step 1's output."
        )
        suggestions.append(
            "For sudo-free completeness: try "
            "`du -sh /tmp/* 2>/dev/null | sort -h | tail` — the `2>/dev/null` "
            "silently drops permission errors and `du` traverses what it can."
        )
        suggestions.append(
            "For complete results with elevation: re-run the same task "
            "without the 'do not use sudo' constraint."
        )

    if state.goal_class == "service_audit":
        suggestions.append(
            "On Linux: `systemctl list-units --failed` and "
            "`journalctl -p 3 -b` together give a fast failed-service summary."
        )
        suggestions.append(
            "On Windows: `Get-Service | Where-Object Status -eq 'Running'` in "
            "PowerShell, or `sc query type=service state=all` in cmd."
        )

    if not observations:
        suggestions.append(
            "The agent collected no successful observations — start from "
            "scratch manually; no partial progress to build on."
        )

    return suggestions


def build_report(
    state: TaskState,
    reason: EscalationReason,
    *,
    summary: str | None = None,
) -> FailureReport:
    observations, ruled_out = extract_observations(state.history)
    return FailureReport(
        task=state.task,
        goal_class=state.goal_class,
        reason=reason,
        summary=summary or _default_summary(state, reason),
        started_at=state.started_at,
        ended_at=time.time(),
        attempts=state.attempts(),
        strategies_tried=list(state.strategies_tried),
        observations=observations,
        ruled_out=ruled_out,
        next_manual_actions=suggest_manual_actions(state, reason, observations),
        snapshots=list(state.snapshots),
        history=[step_record_to_dict(r) for r in state.history],
    )


def _default_summary(state: TaskState, reason: EscalationReason) -> str:
    return (
        f"Could not complete task '{state.task}': {reason.value}. "
        f"Attempted {state.attempts()} step(s) across "
        f"{len(state.strategies_tried)} distinct strategy(ies)."
    )


def write_report(report: FailureReport, out_dir: Path) -> tuple[Path, Path]:
    """Persist the report as both Markdown (human) and JSON (machine).

    Returns (markdown_path, json_path). Files are namespaced by goal_class
    so the same problem class accumulates a history the user can read.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = report.goal_class or "unclassified"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(report.ended_at))
    md = out_dir / f"{slug}-{stamp}.md"
    js = out_dir / f"{slug}-{stamp}.json"
    md.write_text(report.to_markdown(), encoding="utf-8")
    js.write_text(report.to_json(), encoding="utf-8")
    return md, js


def register_snapshot(state: TaskState, snapshot_path: str | Path) -> None:
    """Record where pre-action state was captured. Snapshot creation
    itself lives in the executor — this is just the manifest.
    """
    state.snapshots.append(str(snapshot_path))
