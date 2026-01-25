"""desktop_agent.config

Central configuration for MVP v0.

Keep it simple: read environment variables with sane defaults.

This module also supports loading a local `.env` file for developer
convenience. `.env` is git-ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _load_dotenv_best_effort() -> None:
    """Best-effort `.env` loader.

    We avoid adding a hard dependency on `python-dotenv`.

    Supported format: `KEY=VALUE` per line, with optional quotes.
    Lines starting with `#` are ignored.

    Only sets keys that are not already present in `os.environ`.
    """

    try:
        from pathlib import Path

        env_path = Path.cwd() / ".env"
        if not env_path.exists() or not env_path.is_file():
            return

        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            if not key:
                continue
            os.environ.setdefault(key, val)
    except Exception:
        # Never fail app startup due to dotenv parsing.
        return


# Load `.env` once at import time.
_load_dotenv_best_effort()

# Models exposed in the UI model picker.
SUPPORTED_MODELS: tuple[str, ...] = (
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.2-codex",
    "gpt-5.2-2025-12-11",
    "gpt-5-mini",
    "gpt-5-nano",
)

DEFAULT_MODEL: str = "gpt-5-nano"


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
