"""Core data types shared across the agent.

Only the types the escalation layer actually reads are defined here.
Planner / executor / verifier will add their own fields later — this is
intentionally the smallest viable surface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    expected_state: str | None = None
    risk_level: str = "low"


@dataclass
class StepRecord:
    step: Step
    started_at: float
    ended_at: float
    success: bool
    output: Any = None
    verification: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    repair_strategy: str | None = None
    error: str | None = None


@dataclass
class TaskState:
    task: str
    goal_class: str | None = None
    plan: list[Step] = field(default_factory=list)
    current_step: int = 0
    history: list[StepRecord] = field(default_factory=list)
    strategies_tried: list[str] = field(default_factory=list)
    snapshots: list[str] = field(default_factory=list)
    plan_warnings: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    machine_fingerprint: str | None = None

    def attempts(self) -> int:
        return len(self.history)

    def elapsed_s(self) -> float:
        return time.time() - self.started_at


def json_safe(value: object) -> object:
    """Coerce arbitrary values into JSON-serializable form.

    Primitives pass through. Lists/tuples/dicts recurse. Anything else
    falls back to repr() — we never want serialization to crash mid-write
    just because a tool returned an exotic object.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return repr(value)


def step_to_dict(step: Step) -> dict:
    return {
        "action": step.action,
        "args": json_safe(step.args),
        "expected_state": step.expected_state,
        "risk_level": step.risk_level,
    }


def step_record_to_dict(record: StepRecord) -> dict:
    return {
        "action": record.step.action,
        "args": json_safe(record.step.args),
        "started_at": record.started_at,
        "ended_at": record.ended_at,
        "success": record.success,
        "output": json_safe(record.output),
        "verification": record.verification,
        "diagnosis": record.diagnosis,
        "repair_strategy": record.repair_strategy,
        "error": record.error,
    }
