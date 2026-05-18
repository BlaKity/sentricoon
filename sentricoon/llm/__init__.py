"""Role-routed LLM client. See [[plan-local-cloud-hybrid]] for design rationale.

Public surface:
  - LLMRole              — which agent role is making the call
  - LLMRequest, Message  — request payload
  - LLMResponse          — response payload
  - LLMBackend           — Protocol every backend implements
  - LLMRouter            — maps role → backend per config
  - OpenAICompatBackend  — single HTTP backend serving both OpenAI cloud
                           and Ollama local (same /v1/chat/completions API)
  - MockBackend          — test/dev backend
"""

from .backend import LLMBackend
from .mock import MockBackend
from .openai_compat import OpenAICompatBackend
from .role import LLMRole
from .router import LLMRouter, RouterConfig
from .types import LLMRequest, LLMResponse, Message

__all__ = [
    "LLMBackend",
    "LLMRequest",
    "LLMResponse",
    "LLMRole",
    "LLMRouter",
    "Message",
    "MockBackend",
    "OpenAICompatBackend",
    "RouterConfig",
]
