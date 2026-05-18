"""Runtime configuration — read once at process start.

Defaults are sensible for the local + cloud hybrid the user asked for:
  - Cloud planner / diagnoser: OpenAI-compatible (works with OpenAI direct,
    OpenRouter, Together, etc.). API key from env.
  - Local verifier / input_repair: Ollama at localhost:11434.

Everything is overridable via env vars. CLI flags layer on top in main.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_root() -> Path:
    """Where sentricoon keeps runs, snapshots, and failure reports.

    Override with SENTRICOON_DATA_ROOT. Defaults to ~/.os-technician/ — same
    layout as plan-memory-architecture describes.
    """
    return Path(os.environ.get("SENTRICOON_DATA_ROOT") or (Path.home() / ".os-technician"))


@dataclass
class MyBotConfig:
    # LLM — cloud (planner, diagnoser, critic, AND input_repair when local disabled)
    cloud_api_key: str | None
    cloud_base_url: str
    cloud_model: str

    # LLM — local (verifier, input_repair) — OPT-IN
    local_enabled: bool
    local_base_url: str
    local_model: str

    # Data dirs
    runs_dir: Path
    snapshots_dir: Path
    failures_dir: Path

    # Behavior knobs
    cloud_timeout_s: float = 60.0
    local_timeout_s: float = 30.0
    max_attempts_per_step: int = 3

    # Lazily-built — populated by from_env(); not part of the env contract
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "MyBotConfig":
        """Read config from `env` (or os.environ if not given).

        Local LLM is OPT-IN. Enable it by either:
          - setting SENTRICOON_LOCAL_BASE_URL (implicit opt-in), or
          - setting SENTRICOON_LOCAL_ENABLED=1 (explicit, uses default URL)

        Without either, the agent runs all LLM roles on the cloud backend
        and skips the SemanticVerifier layer entirely. Hard + observation
        verifiers still operate.
        """
        env = env if env is not None else dict(os.environ)
        root = Path(env.get("SENTRICOON_DATA_ROOT") or (Path.home() / ".os-technician"))

        local_url_set = "SENTRICOON_LOCAL_BASE_URL" in env
        local_flag = (env.get("SENTRICOON_LOCAL_ENABLED") or "").lower() in {"1", "true", "yes"}

        return cls(
            cloud_api_key=env.get("SENTRICOON_CLOUD_API_KEY")
                or env.get("OPENAI_API_KEY")
                or env.get("ANTHROPIC_API_KEY"),
            cloud_base_url=env.get("SENTRICOON_CLOUD_BASE_URL") or "https://api.openai.com/v1",
            cloud_model=env.get("SENTRICOON_CLOUD_MODEL") or "gpt-4o-mini",
            local_enabled=local_url_set or local_flag,
            local_base_url=env.get("SENTRICOON_LOCAL_BASE_URL") or "http://localhost:11434/v1",
            local_model=env.get("SENTRICOON_LOCAL_MODEL") or "llama3.2:3b",
            runs_dir=root / "runs",
            snapshots_dir=root / "snapshots",
            failures_dir=root / "failures",
            cloud_timeout_s=float(env.get("SENTRICOON_CLOUD_TIMEOUT_S") or "60"),
            local_timeout_s=float(env.get("SENTRICOON_LOCAL_TIMEOUT_S") or "30"),
            max_attempts_per_step=int(env.get("SENTRICOON_MAX_RETRIES_PER_STEP") or "3"),
        )

    def cloud_configured(self) -> bool:
        return bool(self.cloud_api_key)

    def local_configured(self) -> bool:
        return self.local_enabled
