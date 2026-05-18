"""LLMRouter — maps LLMRole → LLMBackend per config.

This is the cost lever from [[plan-local-cloud-hybrid]]. Default policy:
  - PLANNER, DIAGNOSER, CRITIC → cloud backend
  - VERIFIER, INPUT_REPAIR     → local backend

Anything not configured falls back to `default_backend`. The router does
NOT make LLM calls itself — it returns the chosen backend and the caller
invokes `.complete(request)`. That keeps the cost decision visible to
the call site rather than hidden in a "thin wrapper."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backend import LLMBackend
from .role import LLMRole
from .types import LLMRequest, LLMResponse


class RouterError(Exception):
    """Raised when a role cannot be routed to a backend."""


@dataclass
class RouterConfig:
    role_map: dict[LLMRole, str] = field(default_factory=dict)
    default_backend: str | None = None

    @classmethod
    def default(cls, cloud: str, local: str) -> "RouterConfig":
        """Standard hybrid: cloud for reasoning, local for high-frequency checks."""
        return cls(
            role_map={
                LLMRole.PLANNER: cloud,
                LLMRole.DIAGNOSER: cloud,
                LLMRole.CRITIC: cloud,
                LLMRole.VERIFIER: local,
                LLMRole.INPUT_REPAIR: local,
            },
            default_backend=cloud,
        )

    @classmethod
    def all_one(cls, backend: str) -> "RouterConfig":
        """Stage-1 fallback: every role on one backend."""
        return cls(role_map={r: backend for r in LLMRole}, default_backend=backend)


class LLMRouter:
    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self._backends: dict[str, LLMBackend] = {}
        self.call_counts: dict[str, int] = {}

    def register(self, backend: LLMBackend) -> None:
        self._backends[backend.name] = backend

    def backend_for(self, role: LLMRole) -> LLMBackend:
        target_name = self.config.role_map.get(role) or self.config.default_backend
        if target_name is None:
            raise RouterError(f"no backend mapped for role {role.value!r} and no default_backend")
        backend = self._backends.get(target_name)
        if backend is None:
            raise RouterError(
                f"role {role.value!r} routed to backend {target_name!r}, "
                f"but no such backend is registered. "
                f"Registered: {sorted(self._backends)}"
            )
        return backend

    def complete(self, role: LLMRole, request: LLMRequest) -> LLMResponse:
        backend = self.backend_for(role)
        self.call_counts[backend.name] = self.call_counts.get(backend.name, 0) + 1
        return backend.complete(request)

    def registered_backends(self) -> list[str]:
        return sorted(self._backends.keys())
