"""Request / response types shared across backends.

Mirrors the OpenAI / Anthropic chat-completions shape so existing backends
need no translation layer. JSON-mode and tool-call support are explicit
fields; backends that don't support them ignore the flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMRequest:
    messages: list[Message]
    temperature: float = 0.2
    max_tokens: int | None = None
    json_mode: bool = False
    stop: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    model: str
    backend: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None
