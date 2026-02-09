from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvolvingModels:
    narrator: str
    agent: str


@dataclass(frozen=True)
class ReproductionInheritance:
    mode: str = "crossover"
    mutation_rate: float = 0.05


@dataclass(frozen=True)
class ChildConfig:
    becomes_agent_immediately: bool = True
    dependency_days: int = 7
    allowed_ops: tuple[str, ...] = (
        "talk",
        "self_talk",
        "rest",
        "eat",
        "ask_parent_help",
        "write_memory",
        "edit_instructions",
    )


@dataclass(frozen=True)
class ReproductionConfig:
    consent_required: bool = True
    max_children_per_pair: int = 3
    inheritance: ReproductionInheritance = ReproductionInheritance()
    child: ChildConfig = ChildConfig()


@dataclass(frozen=True)
class RelationshipConfig:
    cooperative_delta: float = 1.0
    competitive_delta: float = -1.0
    matrix_path: str = "relationship_matrix.json"
    backup_path: str = "relationship_matrix.backup.json"


@dataclass(frozen=True)
class DayConfig:
    steps_per_day: int = 12


@dataclass(frozen=True)
class NightConfig:
    context_reset: bool = True
    morning_reread_memory: bool = True


@dataclass(frozen=True)
class LifeConfig:
    start_age_days: int = 0
    reproduction_age_days: int = 7
    max_age_days: int = 200
    start_energy: float = 100.0


@dataclass(frozen=True)
class LimitsConfig:
    max_instruction_chars: int = 4000
    max_memory_chars: int = 12000
    max_edits_per_day: int = 2
    max_instruction_history_chars: int = 40000


@dataclass(frozen=True)
class EvolvingConfig:
    agents_n: int = 12
    models: EvolvingModels = EvolvingModels(narrator="gpt-5.2-2025-12-11", agent="gpt-5.2-2025-12-11")
    day: DayConfig = DayConfig()
    night: NightConfig = NightConfig()
    life: LifeConfig = LifeConfig()
    reproduction: ReproductionConfig = ReproductionConfig()
    relationships: RelationshipConfig = RelationshipConfig()
    limits: LimitsConfig = LimitsConfig()


def _require_obj(obj: Any, key: str) -> dict[str, Any]:
    v = obj.get(key)
    if not isinstance(v, dict):
        raise ValueError(f"'{key}' must be an object")
    return v


def _require_int(obj: dict[str, Any], key: str) -> int:
    v = obj.get(key)
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(f"'{key}' must be an int")
    return int(v)


def _require_num(obj: dict[str, Any], key: str) -> float:
    v = obj.get(key)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise ValueError(f"'{key}' must be a number")
    return float(v)


def _require_str(obj: dict[str, Any], key: str) -> str:
    v = obj.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return v.strip()


def _opt_bool(obj: dict[str, Any], key: str, default: bool) -> bool:
    if key not in obj:
        return default
    v = obj.get(key)
    if isinstance(v, bool):
        return v
    raise ValueError(f"'{key}' must be a bool")


def _opt_str(obj: dict[str, Any], key: str, default: str) -> str:
    if key not in obj:
        return default
    return _require_str(obj, key)


def load_evolving_config(path: str | Path) -> EvolvingConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")

    models_obj = _require_obj(data, "models")
    day_obj = _require_obj(data, "day")
    night_obj = _require_obj(data, "night")
    life_obj = _require_obj(data, "life")
    repro_obj = _require_obj(data, "reproduction")
    inh_obj = _require_obj(repro_obj, "inheritance")
    child_obj = _require_obj(repro_obj, "child")
    rel_obj = _require_obj(data, "relationships")
    limits_obj = _require_obj(data, "limits")
    def _opt_int_in(obj: dict[str, Any], key: str, default: int) -> int:
        if key not in obj:
            return default
        return _require_int(obj, key)

    return EvolvingConfig(
        agents_n=_require_int(data, "agents_n"),
        models=EvolvingModels(narrator=_require_str(models_obj, "narrator"), agent=_require_str(models_obj, "agent")),
        day=DayConfig(steps_per_day=_require_int(day_obj, "steps_per_day")),
        night=NightConfig(
            context_reset=_opt_bool(night_obj, "context_reset", True),
            morning_reread_memory=_opt_bool(night_obj, "morning_reread_memory", True),
        ),
        life=LifeConfig(
            start_age_days=_require_int(life_obj, "start_age_days"),
            reproduction_age_days=_require_int(life_obj, "reproduction_age_days"),
            max_age_days=_require_int(life_obj, "max_age_days"),
            start_energy=_require_num(life_obj, "start_energy") if "start_energy" in life_obj else 100.0,
        ),
        reproduction=ReproductionConfig(
            consent_required=_opt_bool(repro_obj, "consent_required", True),
            max_children_per_pair=_require_int(repro_obj, "max_children_per_pair"),
            inheritance=ReproductionInheritance(
                mode=_opt_str(inh_obj, "mode", "crossover"),
                mutation_rate=_require_num(inh_obj, "mutation_rate"),
            ),
            child=ChildConfig(
                becomes_agent_immediately=_opt_bool(child_obj, "becomes_agent_immediately", True),
                dependency_days=_require_int(child_obj, "dependency_days"),
                allowed_ops=tuple(child_obj.get("allowed_ops", ChildConfig().allowed_ops)),
            ),
        ),
        relationships=RelationshipConfig(
            cooperative_delta=_require_num(rel_obj, "cooperative_delta"),
            competitive_delta=_require_num(rel_obj, "competitive_delta"),
            matrix_path=_require_str(rel_obj, "matrix_path"),
            backup_path=_require_str(rel_obj, "backup_path"),
        ),
        limits=LimitsConfig(
            max_instruction_chars=_require_int(limits_obj, "max_instruction_chars"),
            max_memory_chars=_require_int(limits_obj, "max_memory_chars"),
            max_edits_per_day=_require_int(limits_obj, "max_edits_per_day"),
            max_instruction_history_chars=_opt_int_in(limits_obj, "max_instruction_history_chars", 40000),
        ),
    )
