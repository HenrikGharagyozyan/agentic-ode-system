"""Runtime settings: model access, paths and the limits of the graph, in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # model
    model: str = ""
    api_key: str | None = None  # OpenRouter key; when unset, MODEL's own provider is used
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    max_tokens: int = 16000
    request_timeout_s: float = 600
    request_retries: int = 3  # HTTP-level retries in the client
    provider_retries: int = 5  # retries of a whole model call on transient provider errors

    # paths
    data_file: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "SNLS80mV.mat")
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")

    # graph limits
    max_decisions: int = 8
    max_finish_retries: int = 3

    # executor
    run_timeout_s: float = 300
    max_tool_rounds: int = 10
    max_runs_per_turn: int = 2
    max_focus_calls_per_turn: int = 8
    max_output_chars: int = 12000  # per stream in the dialogue; run records keep the full text

    @classmethod
    def from_env(cls, env_file: Path | None = PROJECT_ROOT / ".env") -> Settings:
        """Settings from the environment (and a .env file). MODEL is required to call a model."""
        if env_file is not None:
            load_dotenv(env_file)
        defaults = cls()
        return cls(
            model=os.getenv("MODEL", ""),
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", defaults.base_url),
            data_file=Path(os.getenv("AGENTIC_ODE_DATA_FILE", defaults.data_file)),
            runs_dir=Path(os.getenv("AGENTIC_ODE_RUNS_DIR", defaults.runs_dir)),
        )
