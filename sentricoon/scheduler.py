"""Scheduler — Phase 5 (autonomy layer). Stub.

Only meaningful once Phase 1–4 are running end-to-end. Don't build before
the core loop is reliable — a scheduler running a flaky agent is worse
than no scheduler.
"""

from __future__ import annotations

from .agent import Agent


class Scheduler:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def add_task(self, task: str, interval_s: float) -> None:
        raise NotImplementedError("Scheduler is Phase 5 — not yet built")

    def start(self) -> None:
        raise NotImplementedError("Scheduler is Phase 5 — not yet built")
