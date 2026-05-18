"""OpenAI-compatible HTTP backend.

One implementation serves both cloud (OpenAI, OpenRouter, Together, …)
and local (Ollama at :11434/v1, llama.cpp server, LM Studio). All speak
the same `/v1/chat/completions` shape — only base_url, api_key, and
model_name differ.

Uses stdlib urllib so the LLM client adds no third-party dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .types import LLMRequest, LLMResponse


class BackendError(Exception):
    """Raised when an HTTP backend call fails."""


@dataclass
class OpenAICompatConfig:
    base_url: str           # e.g. "https://api.openai.com/v1" or "http://localhost:11434/v1"
    model: str              # e.g. "gpt-4o-mini" or "llama3.2:3b"
    api_key: str | None = None
    timeout_s: float = 60.0
    name: str = "openai-compat"  # display name in logs / response.backend


class OpenAICompatBackend:
    def __init__(self, config: OpenAICompatConfig) -> None:
        self.config = config
        self.name = config.name
        self.model = config.model

    def complete(self, request: LLMRequest) -> LLMResponse:
        body: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.stop:
            body["stop"] = request.stop
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(request.extra)

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise BackendError(f"{self.name} HTTP {e.code}: {detail[:500]}") from e
        except urllib.error.URLError as e:
            raise BackendError(f"{self.name} unreachable at {url}: {e.reason}") from e
        except json.JSONDecodeError as e:
            raise BackendError(f"{self.name} returned non-JSON: {e}") from e

        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError) as e:
            raise BackendError(f"{self.name} response shape unexpected: missing {e}") from e

        usage = payload.get("usage") or {}
        return LLMResponse(
            content=content,
            model=payload.get("model", self.model),
            backend=self.name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=finish_reason,
            raw=payload,
        )


def cloud_openai(api_key: str, model: str = "gpt-4o-mini") -> OpenAICompatBackend:
    """Convenience constructor for OpenAI cloud."""
    return OpenAICompatBackend(OpenAICompatConfig(
        base_url="https://api.openai.com/v1",
        model=model,
        api_key=api_key,
        name="openai-cloud",
    ))


def local_ollama(model: str = "llama3.2:3b", base_url: str = "http://localhost:11434/v1") -> OpenAICompatBackend:
    """Convenience constructor for a local Ollama server."""
    return OpenAICompatBackend(OpenAICompatConfig(
        base_url=base_url,
        model=model,
        api_key=None,  # Ollama OpenAI-compat endpoint ignores auth
        name="ollama-local",
    ))
