from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingModels:
    riddler: str
    actor: str
    solver: str
    partner: str
    critic: str


@dataclass(frozen=True)
class TrainingConfig:
    sweet_spot_pct: float = 50.0
    std_pct: float = 10.0
    solver_count: int = 10
    max_partners_total: int = 6
    max_partners_per_solver: int = 1
    riddle_style: str = "math"
    episodes: int = 1
    sleep_seconds_between_episodes: float = 0.0
    status_every_solvers: int = 0
    stream_grade_each_solver: bool = False
    dataset_filename: str = "dataset.jsonl"
    models: TrainingModels = TrainingModels(
        riddler="gpt-5.2-2025-12-11",
        actor="gpt-5.2-2025-12-11",
        solver="gpt-5.2-2025-12-11",
        partner="gpt-5.2-2025-12-11",
        critic="gpt-5.2-2025-12-11",
    )


def _require_number(obj: dict[str, Any], key: str) -> float:
    v = obj.get(key)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise ValueError(f"'{key}' must be a number")
    return float(v)


def _require_int(obj: dict[str, Any], key: str) -> int:
    v = obj.get(key)
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(f"'{key}' must be an int")
    return int(v)


def _require_str(obj: dict[str, Any], key: str) -> str:
    v = obj.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return v.strip()


def load_training_config(path: str | Path) -> TrainingConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")

    models_obj = data.get("models", {})
    if not isinstance(models_obj, dict):
        raise ValueError("'models' must be an object")

    models = TrainingModels(
        riddler=_require_str(models_obj, "riddler"),
        actor=_require_str(models_obj, "actor"),
        solver=_require_str(models_obj, "solver"),
        partner=_require_str(models_obj, "partner"),
        critic=_require_str(models_obj, "critic"),
    )

    def _opt_bool(key: str, default: bool) -> bool:
        v = data.get(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip() in {"1", "0", "true", "false", "True", "False"}:
            return v.strip() in {"1", "true", "True"}
        raise ValueError(f"'{key}' must be a bool")

    def _opt_int(key: str, default: int) -> int:
        if key not in data:
            return default
        return _require_int(data, key)

    def _opt_float(key: str, default: float) -> float:
        if key not in data:
            return default
        return _require_number(data, key)

    def _opt_str(key: str, default: str) -> str:
        v = data.get(key, default)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"'{key}' must be a non-empty string")
        return v.strip()

    return TrainingConfig(
        sweet_spot_pct=_require_number(data, "sweet_spot_pct"),
        std_pct=_require_number(data, "std_pct"),
        solver_count=_require_int(data, "solver_count"),
        max_partners_total=_require_int(data, "max_partners_total"),
        max_partners_per_solver=_require_int(data, "max_partners_per_solver"),
        riddle_style=_opt_str("riddle_style", "math"),
        episodes=_opt_int("episodes", 1),
        sleep_seconds_between_episodes=_opt_float("sleep_seconds_between_episodes", 0.0),
        status_every_solvers=_opt_int("status_every_solvers", 0),
        stream_grade_each_solver=_opt_bool("stream_grade_each_solver", False),
        dataset_filename=_opt_str("dataset_filename", "dataset.jsonl"),
        models=models,
    )
