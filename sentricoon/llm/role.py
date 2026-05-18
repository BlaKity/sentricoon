"""LLM roles — the categories the router dispatches on.

These are stable identities, not free-form strings. Adding a role is
deliberate: it means a new place in the agent loop is calling the LLM
and the router needs to decide which backend serves it.
"""

from __future__ import annotations

from enum import Enum


class LLMRole(str, Enum):
    PLANNER = "planner"          # cloud, high-quality, infrequent (1-2 per task)
    VERIFIER = "verifier"        # local, frequent (1 per step) — the cost lever
    INPUT_REPAIR = "input_repair"  # local first, cloud on repeated failure
    DIAGNOSER = "diagnoser"      # cloud (different family from planner at stage 3)
    CRITIC = "critic"            # cloud, different family, only on verification failure
