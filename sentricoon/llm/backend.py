"""LLMBackend protocol — every backend exposes a single `complete` method.

Using `typing.Protocol` so backends don't need to inherit a common base —
the MockBackend, OpenAICompatBackend, and any future Anthropic/local
variant just need the right method signature.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import LLMRequest, LLMResponse


@runtime_checkable
class LLMBackend(Protocol):
    name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...
