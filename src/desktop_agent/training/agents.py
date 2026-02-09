from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from desktop_agent.llm import OpenAIResponsesClient

from .config import TrainingConfig
from .llm_json import JSONCallConfig, ResponsesClient, call_json_object
from .schemas import (
    ActorOutput,
    CriticReport,
    RiddlerOutput,
    SolverOutput,
    validate_actor_output,
    validate_critic_report,
    validate_riddler_output,
    validate_solver_output,
)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


@dataclass(frozen=True)
class TrainingClients:
    client: ResponsesClient

    @staticmethod
    def from_env(*, api_key: Optional[str]) -> "TrainingClients":
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for training runs (set it in .env or environment).")
        return TrainingClients(client=OpenAIResponsesClient(api_key=api_key))


class RiddlerAgent:
    def __init__(self, *, clients: TrainingClients, model: str) -> None:
        self._client = clients.client
        self._model = model

    def generate(
        self,
        *,
        sweet_spot_pct: float,
        std_pct: float,
        riddler_memory_excerpt: str,
        riddle_style: str,
    ) -> RiddlerOutput:
        style = (riddle_style or "math").strip().lower()
        if style not in {"math", "mathematical"}:
            style = "math"
        system = (
            "You are Riddler.\n"
            "Your job: generate a riddle calibrated so an ensemble of diverse solvers will solve it about the target rate.\n"
            "You must output ONLY valid JSON (no markdown).\n"
            "Output schema:\n"
            "{\n"
            '  "riddle_id": string,\n'
            '  "riddle": string,\n'
            '  "solution": string,\n'
            '  "solution_rationale": string,\n'
            '  "grading_criteria": string,\n'
            '  "tags": [string, ...],\n'
            '  "difficulty_guess_pct": number\n'
            "}\n"
            "Rules:\n"
            "- Make it self-contained (no external knowledge beyond common reasoning).\n"
            "- Make it mathematical: the riddle should reduce to a well-defined math problem with a checkable derivation.\n"
            "- Include grading_criteria that lets a critic grade solver answers deterministically.\n"
            "- Do not include multiple riddles.\n"
        )
        user = (
            f"Target solve-rate sweet spot: {sweet_spot_pct:.1f}% (std {std_pct:.1f}%).\n"
            f"Style: {style}\n"
            "Recent memory excerpt (for calibration; may be empty):\n"
            "-----\n"
            f"{riddler_memory_excerpt.strip()}\n"
            "-----\n"
            "Generate ONE new riddle now."
        )
        obj = call_json_object(client=self._client, cfg=JSONCallConfig(model=self._model), system=system, user=user)
        return validate_riddler_output(obj)


class ActorAgent:
    def __init__(self, *, clients: TrainingClients, model: str) -> None:
        self._client = clients.client
        self._model = model

    def propose_solver_profiles(
        self,
        *,
        solver_count: int,
        riddle: str,
        actor_memory_excerpt: str,
    ) -> ActorOutput:
        system = (
            "You are Actor.\n"
            "Your job: propose diverse solver instruction profiles for the same riddle.\n"
            "Return ONLY valid JSON (no markdown).\n"
            "Output schema:\n"
            "{\n"
            '  "solver_instruction_profiles": [string, ...],  // length must equal solver_count\n'
            '  "actor_summary": string,\n'
            '  "next_instruction_hypotheses": [string, ...]\n'
            "}\n"
            "Rules:\n"
            "- Each profile must be short (1-4 sentences) and different in strategy.\n"
            "- Profiles must not reveal the answer.\n"
        )
        user = (
            f"solver_count={solver_count}\n"
            "Riddle:\n"
            "-----\n"
            f"{riddle.strip()}\n"
            "-----\n"
            "Recent actor memory excerpt (may be empty):\n"
            "-----\n"
            f"{actor_memory_excerpt.strip()}\n"
            "-----\n"
            "Return solver_instruction_profiles now."
        )
        obj = call_json_object(client=self._client, cfg=JSONCallConfig(model=self._model), system=system, user=user)
        return validate_actor_output(obj, expected_solver_count=solver_count)

    def summarize_episode(
        self,
        *,
        riddle_id: str,
        riddle: str,
        solver_runs: list[dict[str, Any]],
        critic_grading: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system = (
            "You are Actor.\n"
            "Your job: summarize what worked/failed across solver attempts to improve future solver instructions.\n"
            "Return ONLY valid JSON (no markdown).\n"
            "Output schema:\n"
            "{\n"
            '  "memory_append_title": string,\n'
            '  "memory_append_body": string,\n'
            '  "instruction_takeaways": [string, ...]\n'
            "}\n"
            "Rules:\n"
            "- Do NOT include chain-of-thought from solvers; focus on observable patterns.\n"
            "- Make memory_append_body concise and actionable.\n"
        )
        user = (
            f"riddle_id={riddle_id}\n"
            "Riddle:\n"
            "-----\n"
            f"{riddle.strip()}\n"
            "-----\n"
            "Solver runs (JSON):\n"
            f"{_json_dumps(solver_runs)}\n"
            "Critic grading (JSON):\n"
            f"{_json_dumps(critic_grading)}\n"
        )
        return call_json_object(client=self._client, cfg=JSONCallConfig(model=self._model), system=system, user=user)

    def status_report(
        self,
        *,
        riddle_id: str,
        riddle: str,
        solver_runs_so_far: list[dict[str, Any]],
        remaining_solver_ids: list[str],
    ) -> dict[str, Any]:
        system = (
            "You are Actor.\n"
            "Your job: provide a short status report while solvers are running.\n"
            "Return ONLY valid JSON.\n"
            "Schema:\n"
            "{\n"
            '  "status_summary": string,\n'
            '  "likely_to_succeed": [string, ...],\n'
            '  "risks": [string, ...]\n'
            "}\n"
            "Rules:\n"
            "- Base your assessment only on solver_runs_so_far content.\n"
            "- Do not reveal the final answer.\n"
            "- Keep it concise.\n"
        )
        user = (
            f"riddle_id={riddle_id}\n"
            "Riddle:\n-----\n"
            f"{riddle.strip()}\n"
            "-----\n"
            "solver_runs_so_far (JSON):\n"
            f"{_json_dumps(solver_runs_so_far)}\n"
            f"remaining_solver_ids={_json_dumps(remaining_solver_ids)}\n"
        )
        return call_json_object(client=self._client, cfg=JSONCallConfig(model=self._model), system=system, user=user)

    def answer_question(
        self,
        *,
        riddle_id: str,
        riddle: str,
        solver_runs_so_far: list[dict[str, Any]],
        remaining_solver_ids: list[str],
        question: str,
    ) -> dict[str, Any]:
        system = (
            "You are Actor.\n"
            "You are running multiple solvers on a riddle. The user may ask about progress or request a summary.\n"
            "Return ONLY valid JSON.\n"
            "Schema:\n"
            '{ "answer": string }\n'
            "Rules:\n"
            "- Base your answer only on provided solver_runs_so_far and remaining_solver_ids.\n"
            "- If asked which solvers are likely to succeed, you may make a best-effort guess based on what they wrote.\n"
            "- Do not reveal the final answer.\n"
            "- Keep it concise.\n"
        )
        user = (
            f"riddle_id={riddle_id}\n"
            "Riddle:\n-----\n"
            f"{riddle.strip()}\n"
            "-----\n"
            f"remaining_solver_ids={_json_dumps(remaining_solver_ids)}\n"
            "solver_runs_so_far (JSON):\n"
            f"{_json_dumps(solver_runs_so_far)}\n"
            "User question:\n"
            f"{question.strip()}\n"
        )
        return call_json_object(client=self._client, cfg=JSONCallConfig(model=self._model), system=system, user=user)


class PartnerAgent:
    def __init__(self, *, clients: TrainingClients, model: str) -> None:
        self._client = clients.client
        self._model = model

    def advise(self, *, riddle: str, solver_profile: str, what_i_tried: str, where_stuck: str) -> dict[str, Any]:
        system = (
            "You are Partner.\n"
            "Your job: help a solver by suggesting hints, checks, or alternative approaches.\n"
            "Return ONLY valid JSON.\n"
            "Output schema:\n"
            '{ "partner_message": string }\n'
            "Rules:\n"
            "- Do not reveal the full final answer directly unless it is unavoidable; prefer hints and verification steps.\n"
            "- Keep it short (<= 12 sentences).\n"
        )
        user = (
            "Riddle:\n-----\n"
            f"{riddle.strip()}\n"
            "-----\n"
            "Solver profile:\n-----\n"
            f"{solver_profile.strip()}\n"
            "-----\n"
            "Solver says what they tried:\n-----\n"
            f"{(what_i_tried or '').strip()}\n"
            "-----\n"
            "Solver says where stuck:\n-----\n"
            f"{(where_stuck or '').strip()}\n"
            "-----\n"
            "Provide partner_message now."
        )
        return call_json_object(client=self._client, cfg=JSONCallConfig(model=self._model), system=system, user=user)


class SolverAgent:
    def __init__(self, *, clients: TrainingClients, model: str, solver_id: str) -> None:
        self._client = clients.client
        self._model = model
        self._solver_id = solver_id

    def solve(
        self,
        *,
        instruction_profile: str,
        riddle: str,
        partner_message: Optional[str] = None,
    ) -> SolverOutput:
        system = (
            "You are Solver.\n"
            "You must solve the provided riddle.\n"
            "You may think privately, but you must output ONLY valid JSON with the schema below.\n"
            "Schema:\n"
            "{\n"
            '  "final_answer": string,\n'
            '  "derivation": string,  // concise derivation/explanation, enough to check\n'
            '  "confidence": number,  // 0..1\n'
            '  "checks": [string, ...],\n'
            '  "help_request": { "needed": bool, "what_i_tried": string, "where_i_am_stuck": string }\n'
            "}\n"
            "Rules:\n"
            "- Put your final solution in final_answer.\n"
            "- Put a concise, checkable derivation in derivation (no long rambling).\n"
            "- If you are stuck, set help_request.needed=true and explain briefly.\n"
            "- Do not include chain-of-thought; keep checks short.\n"
        )
        user = (
            f"Solver id: {self._solver_id}\n"
            "Solver instructions:\n-----\n"
            f"{instruction_profile.strip()}\n"
            "-----\n"
            "Riddle:\n-----\n"
            f"{riddle.strip()}\n"
            "-----\n"
        )
        if partner_message:
            user += "Partner message:\n-----\n" + partner_message.strip() + "\n-----\n"
        obj = call_json_object(client=self._client, cfg=JSONCallConfig(model=self._model), system=system, user=user)
        return validate_solver_output(obj)


class CriticAgent:
    def __init__(self, *, clients: TrainingClients, model: str) -> None:
        self._client = clients.client
        self._model = model

    def precheck(self, *, riddle: RiddlerOutput) -> dict[str, Any]:
        system = (
            "You are Critic.\n"
            "Your job: verify the riddle is internally consistent and solvable GIVEN the provided solution.\n"
            "Do NOT solve the riddle from scratch; only check consistency with the provided solution/rationale.\n"
            "Return ONLY JSON:\n"
            '{ "riddle_valid": bool, "validity_notes": string }\n'
        )
        user = "Riddle package (JSON):\n" + _json_dumps(riddle)
        return call_json_object(client=self._client, cfg=JSONCallConfig(model=self._model), system=system, user=user)

    def grade(
        self,
        *,
        riddle: RiddlerOutput,
        solver_runs: list[dict[str, Any]],
        expected_solver_ids: list[str],
        reward_cfg: TrainingConfig,
    ) -> CriticReport:
        system = (
            "You are Critic.\n"
            "Your job:\n"
            "1) Re-confirm riddle validity given riddler's solution.\n"
            "2) Grade each solver's final_answer as correct/incorrect.\n"
            "3) If there are multiple correct solvers, pick the best derivation for dataset use.\n"
            "Return ONLY JSON with schema:\n"
            "{\n"
            '  "riddle_valid": bool,\n'
            '  "validity_notes": string,\n'
            '  "grading": [ { "solver_id": string, "correct": bool, "notes": string }, ... ],\n'
            '  "solve_rate_pct": number,\n'
            '  "reward": number,\n'
            '  "best_solver_id": string,  // one of the solver_ids with correct=true, else empty\n'
            '  "best_derivation": string  // copy from that solver, else empty\n'
            "}\n"
            "Rules:\n"
            "- Grade based on the riddler-provided solution and grading_criteria.\n"
            "- Include every solver_id exactly once in grading.\n"
            "- solve_rate_pct must equal (correct_count / total)*100.\n"
            "- reward must be computed as exp(-0.5*((p-mu)/sigma)^2) with:\n"
            f"  mu={reward_cfg.sweet_spot_pct}, sigma={reward_cfg.std_pct}\n"
            "- best_derivation should be the most clear, concise, and checkable derivation among correct solvers.\n"
        )
        user = (
            "Riddle package (JSON):\n"
            f"{_json_dumps(riddle)}\n"
            "Solver runs (JSON):\n"
            f"{_json_dumps(solver_runs)}\n"
        )
        obj = call_json_object(client=self._client, cfg=JSONCallConfig(model=self._model), system=system, user=user)
        return validate_critic_report(obj, expected_solver_ids=expected_solver_ids)

    def grade_single(
        self,
        *,
        riddle: RiddlerOutput,
        solver_run: dict[str, Any],
        reward_cfg: TrainingConfig,
    ) -> dict[str, Any]:
        system = (
            "You are Critic.\n"
            "Your job: grade a single solver answer as correct/incorrect based on the provided riddle solution and criteria.\n"
            "Return ONLY JSON:\n"
            '{ "solver_id": string, "correct": bool, "notes": string }\n'
        )
        user = (
            "Riddle package (JSON):\n"
            f"{_json_dumps(riddle)}\n"
            "Solver run (JSON):\n"
            f"{_json_dumps(solver_run)}\n"
            f"Reward target mu={reward_cfg.sweet_spot_pct}, sigma={reward_cfg.std_pct}\n"
        )
        return call_json_object(client=self._client, cfg=JSONCallConfig(model=self._model, temperature=0.0), system=system, user=user)
