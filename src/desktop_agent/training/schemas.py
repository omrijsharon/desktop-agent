from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict


class RiddlerOutput(TypedDict):
    riddle_id: str
    riddle: str
    solution: str
    solution_rationale: str
    tags: list[str]
    difficulty_guess_pct: float
    grading_criteria: str


class HelpRequest(TypedDict):
    needed: bool
    what_i_tried: str
    where_i_am_stuck: str


class SolverOutput(TypedDict):
    final_answer: str
    derivation: str
    confidence: float
    checks: list[str]
    help_request: HelpRequest


class SolverRun(TypedDict, total=False):
    solver_id: str
    instruction_profile: str
    final_answer: str
    derivation: str
    confidence: float
    asked_partner: bool
    partner_summary: str
    checks: list[str]


class ActorOutput(TypedDict):
    solver_instruction_profiles: list[str]
    actor_summary: str
    next_instruction_hypotheses: list[str]


class CriticGrading(TypedDict):
    solver_id: str
    correct: bool
    notes: str


class CriticReport(TypedDict):
    riddle_valid: bool
    validity_notes: str
    grading: list[CriticGrading]
    solve_rate_pct: float
    reward: float
    best_solver_id: str
    best_derivation: str


@dataclass(frozen=True)
class EpisodeArtifacts:
    episode_id: str
    dir_path: str


SolverOutcome = Literal["correct", "incorrect", "unknown"]


def _require_str(obj: dict[str, Any], key: str) -> str:
    v = obj.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return v.strip()


def _require_bool(obj: dict[str, Any], key: str) -> bool:
    v = obj.get(key)
    if not isinstance(v, bool):
        raise ValueError(f"'{key}' must be a bool")
    return v


def _require_number(obj: dict[str, Any], key: str) -> float:
    v = obj.get(key)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise ValueError(f"'{key}' must be a number")
    return float(v)


def _require_str_list(obj: dict[str, Any], key: str) -> list[str]:
    v = obj.get(key)
    if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
        raise ValueError(f"'{key}' must be a list of non-empty strings")
    return [str(x).strip() for x in v]


def validate_riddler_output(payload: Any) -> RiddlerOutput:
    if not isinstance(payload, dict):
        raise ValueError("riddler output must be an object")
    return RiddlerOutput(
        riddle_id=_require_str(payload, "riddle_id"),
        riddle=_require_str(payload, "riddle"),
        solution=_require_str(payload, "solution"),
        solution_rationale=_require_str(payload, "solution_rationale"),
        tags=_require_str_list(payload, "tags"),
        difficulty_guess_pct=_require_number(payload, "difficulty_guess_pct"),
        grading_criteria=_require_str(payload, "grading_criteria"),
    )


def validate_actor_output(payload: Any, *, expected_solver_count: int) -> ActorOutput:
    if not isinstance(payload, dict):
        raise ValueError("actor output must be an object")
    profiles = _require_str_list(payload, "solver_instruction_profiles")
    if len(profiles) != expected_solver_count:
        raise ValueError(f"'solver_instruction_profiles' must have length {expected_solver_count}")
    return ActorOutput(
        solver_instruction_profiles=profiles,
        actor_summary=_require_str(payload, "actor_summary"),
        next_instruction_hypotheses=_require_str_list(payload, "next_instruction_hypotheses"),
    )


def validate_solver_output(payload: Any) -> SolverOutput:
    if not isinstance(payload, dict):
        raise ValueError("solver output must be an object")
    help_obj = payload.get("help_request")
    if not isinstance(help_obj, dict):
        raise ValueError("'help_request' must be an object")
    return SolverOutput(
        final_answer=_require_str(payload, "final_answer"),
        derivation=_require_str(payload, "derivation"),
        confidence=_require_number(payload, "confidence"),
        checks=_require_str_list(payload, "checks"),
        help_request=HelpRequest(
            needed=_require_bool(help_obj, "needed"),
            what_i_tried=str(help_obj.get("what_i_tried", "") or ""),
            where_i_am_stuck=str(help_obj.get("where_i_am_stuck", "") or ""),
        ),
    )


def validate_critic_report(payload: Any, *, expected_solver_ids: list[str]) -> CriticReport:
    if not isinstance(payload, dict):
        raise ValueError("critic report must be an object")
    grading = payload.get("grading")
    if not isinstance(grading, list):
        raise ValueError("'grading' must be a list")
    rows: list[CriticGrading] = []
    seen: set[str] = set()
    for row in grading:
        if not isinstance(row, dict):
            raise ValueError("each grading entry must be an object")
        sid = _require_str(row, "solver_id")
        if sid in seen:
            raise ValueError(f"duplicate solver_id in grading: {sid}")
        seen.add(sid)
        rows.append(CriticGrading(solver_id=sid, correct=_require_bool(row, "correct"), notes=str(row.get("notes", "") or "")))
    missing = [sid for sid in expected_solver_ids if sid not in seen]
    if missing:
        raise ValueError(f"critic grading missing solver_ids: {missing}")
    return CriticReport(
        riddle_valid=bool(payload.get("riddle_valid", False)),
        validity_notes=str(payload.get("validity_notes", "") or ""),
        grading=rows,
        solve_rate_pct=_require_number(payload, "solve_rate_pct"),
        reward=_require_number(payload, "reward"),
        best_solver_id=str(payload.get("best_solver_id", "") or ""),
        best_derivation=str(payload.get("best_derivation", "") or ""),
    )
