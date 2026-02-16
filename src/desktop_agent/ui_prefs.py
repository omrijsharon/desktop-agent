"""desktop_agent.ui_prefs

Persistence for Chat UI defaults across app restarts.

We keep this separate from per-chat saved conversations:
- Per-chat history lives in `chat_history/<chat_id>.json`
- UI defaults live in `chat_history/chat_ui_defaults.json`
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


def defaults_path(repo_root: Path) -> Path:
    return (repo_root / "chat_history" / "chat_ui_defaults.json").resolve()


def load_defaults(repo_root: Path) -> JsonDict:
    p = defaults_path(repo_root)
    try:
        if not p.exists():
            return {}
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_defaults(repo_root: Path, data: JsonDict) -> None:
    p = defaults_path(repo_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def dataclass_to_json_dict(obj: Any) -> JsonDict:
    if is_dataclass(obj):
        d = asdict(obj)
        return d if isinstance(d, dict) else {}
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    return {}


def apply_overrides(target: Any, overrides: JsonDict) -> None:
    """Best-effort: assign `overrides` keys onto `target` attrs if they exist."""

    if not isinstance(overrides, dict):
        return
    for k, v in overrides.items():
        if not isinstance(k, str):
            continue
        if not hasattr(target, k):
            continue
        try:
            setattr(target, k, v)
        except Exception:
            continue

