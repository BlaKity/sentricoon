"""Memory layers — see [[plan-memory-architecture]] for the four-layer plan.

Currently implemented:
  - Layer 1 (working): in-process TaskState, defined in state.py
  - Layer 2 (episodic): EpisodicLog below — append-only JSONL audit log

Parked (NotImplementedError until built):
  - Layer 3 (semantic): per-machine SQLite facts with decay
  - Layer 4 (procedural): learned strategy playbooks, feeds escalation
    via the quarantine_source hook
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from .state import StepRecord, json_safe, step_record_to_dict


# ============================================================================
# Layer 2 — Episodic memory (built)
# ============================================================================


def new_run_id() -> str:
    """Timestamped, collision-resistant run identifier."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def default_runs_dir() -> Path:
    return Path.home() / ".os-technician" / "runs"


def default_run_path(run_id: str) -> Path:
    return default_runs_dir() / f"{run_id}.jsonl"


class EpisodicLog:
    """Append-only JSONL event log for a single agent run.

    One file per run. Every event is a single JSON object on one line.
    Writes fsync to survive crashes. Reads tolerate a torn final line
    (which happens if the process was killed mid-append).

    The log records more than just step executions — task starts, plans
    generated, strategies recorded, escalations, confirmations, etc.
    A single event type per line keeps the schema discriminator simple.
    """

    KNOWN_TYPES = frozenset({
        "task_start",
        "plan",
        "plan_warning",
        "step",
        "strategy",
        "confirmation",
        "escalation",
        "snapshot",
        "task_end",
    })

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ----- low-level append -----

    def append(self, event_type: str, payload: dict) -> None:
        """Append one event. Generic primitive; prefer typed helpers below."""
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event_type must be a non-empty string")
        event = {
            "ts_ms": int(time.time() * 1000),
            "type": event_type,
            "payload": json_safe(payload) if isinstance(payload, dict) else {"value": json_safe(payload)},
        }
        line = json.dumps(event, separators=(",", ":"))
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    # ----- typed helpers (preferred) -----

    def append_task_start(self, run_id: str, task: str, goal_class: str | None) -> None:
        self.append("task_start", {"run_id": run_id, "task": task, "goal_class": goal_class})

    def append_plan(self, run_id: str, plan: list[dict], strategy_fingerprint: str | None = None) -> None:
        self.append("plan", {"run_id": run_id, "plan": plan, "strategy_fingerprint": strategy_fingerprint})

    def append_plan_warning(self, run_id: str, message: str) -> None:
        self.append("plan_warning", {"run_id": run_id, "message": message})

    def append_step(self, run_id: str, record: StepRecord) -> None:
        payload = {"run_id": run_id, **step_record_to_dict(record)}
        self.append("step", payload)

    def append_strategy(self, run_id: str, fingerprint: str, actions: list[str]) -> None:
        self.append("strategy", {"run_id": run_id, "fingerprint": fingerprint, "actions": actions})

    def append_confirmation(self, run_id: str, action: str, token: str, granted: bool) -> None:
        # Don't log the full token — short prefix is enough for forensics
        token_repr = (token[:8] + "...") if token and len(token) > 8 else token
        self.append("confirmation", {
            "run_id": run_id, "action": action, "token": token_repr, "granted": granted,
        })

    def append_escalation(self, run_id: str, reason: str, attempts: int, strategies: int) -> None:
        self.append("escalation", {
            "run_id": run_id, "reason": reason, "attempts": attempts, "strategies": strategies,
        })

    def append_snapshot(self, run_id: str, kind: str, path: str) -> None:
        self.append("snapshot", {"run_id": run_id, "kind": kind, "path": path})

    def append_task_end(self, run_id: str, status: str, summary: str | None = None) -> None:
        self.append("task_end", {"run_id": run_id, "status": status, "summary": summary})

    # ----- reads -----

    def read_all(self) -> list[dict]:
        """Return every event in the log, in append order.

        Tolerates a torn last line (process crashed mid-write).
        """
        if not self.path.exists():
            return []
        events: list[dict] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    events.append(json.loads(stripped))
                except json.JSONDecodeError:
                    # Partial trailing line — stop reading; don't crash on partial event
                    break
        return events

    def read_events(self, event_type: str) -> list[dict]:
        return [e for e in self.read_all() if e.get("type") == event_type]

    def count(self) -> int:
        return len(self.read_all())


# ============================================================================
# Layer 3 + 4 — Semantic + Procedural memory (parked)
# ============================================================================


class Memory:
    """Facade for semantic + procedural memory. Layers 3 and 4 still parked.

    Where Layer 4 plugs in: `is_strategy_quarantined` is the hook
    `escalation.should_escalate` consumes via `quarantine_source=`.
    Returning False now is the safe default — nothing quarantined until
    procedural memory exists.
    """

    def add_fact(self, kind: str, content: str, ttl_s: float | None = None) -> None:
        raise NotImplementedError("Layer 3 (semantic) not yet built — see plan-memory-architecture")

    def recall(self, query: str, *, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Layer 3 (semantic) not yet built — see plan-memory-architecture")

    def record_strategy_outcome(self, fingerprint: str, outcome: str) -> None:
        raise NotImplementedError("Layer 4 (procedural) not yet built — see plan-memory-architecture")

    def is_strategy_quarantined(self, fingerprint: str) -> bool:
        # Default: nothing is quarantined until procedural memory layer exists.
        # This makes `escalation.should_escalate(..., quarantine_source=mem.is_strategy_quarantined)`
        # callable today without behavior change.
        return False

    def quarantined_action_sequences(self) -> list[list[str]]:
        """Return action sequences known to have failed before.

        Used by the planner to tell the LLM "don't replicate these approaches."
        Empty until Layer 4 (procedural memory) is built.
        """
        return []
