"""MockBackend — deterministic backend for tests and offline development.

Two modes:
  - script:   pre-load a list of responses; calls return them in order
  - callable: pass a function (request) -> str/dict
"""

from __future__ import annotations

from collections.abc import Callable

from .types import LLMRequest, LLMResponse


class MockBackend:
    def __init__(
        self,
        *,
        script: list[str] | None = None,
        responder: Callable[[LLMRequest], str] | None = None,
        name: str = "mock",
        model: str = "mock-1",
    ) -> None:
        if script is None and responder is None:
            raise ValueError("MockBackend needs either script= or responder=")
        self.name = name
        self.model = model
        self._script = list(script) if script else None
        self._responder = responder
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self._script is not None:
            if not self._script:
                raise RuntimeError(f"MockBackend {self.name!r} script exhausted")
            content = self._script.pop(0)
        else:
            assert self._responder is not None
            content = self._responder(request)
        return LLMResponse(
            content=content,
            model=self.model,
            backend=self.name,
            finish_reason="stop",
        )
