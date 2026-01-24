"""desktop_agent.config

Central configuration for MVP v0.

Keep it simple: read environment variables with sane defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Models exposed in the UI model picker.
SUPPORTED_MODELS: tuple[str, ...] = (
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.2-codex",
    "gpt-5-mini",
    "gpt-5-nano",
)

DEFAULT_MODEL: str = "gpt-5-mini"


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None = None
    openai_model: str = DEFAULT_MODEL
    step_mode_default: bool = False
    always_on_top: bool = False


def load_config() -> AppConfig:
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    if model not in SUPPORTED_MODELS:
        model = DEFAULT_MODEL

    return AppConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_model=model,
        step_mode_default=os.environ.get("STEP_MODE", "0") in {"1", "true", "True"},
        always_on_top=os.environ.get("ALWAYS_ON_TOP", "0") in {"1", "true", "True"},
    )
