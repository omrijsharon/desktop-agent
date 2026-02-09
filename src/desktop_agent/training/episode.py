from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from desktop_agent.config import load_config

from .agents import ActorAgent, CriticAgent, PartnerAgent, RiddlerAgent, SolverAgent, TrainingClients
from .config import TrainingConfig
from .memory import MemoryPaths, append_md_section, read_tail, utc_ts
from .reward import gaussian_reward
from .schemas import CriticReport, RiddlerOutput


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    artifacts_dir: Path
    riddler: RiddlerOutput
    solver_runs: list[dict[str, Any]]
    critic: CriticReport
    computed_solve_rate_pct: float
    computed_reward: float


ProgressEvent = dict[str, Any]
ProgressCallback = Callable[[ProgressEvent], None]
GetActorQuestions = Callable[[], Sequence[str]]


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _episode_id() -> str:
    return utc_ts().replace(":", "").replace("-", "")


def run_episode(
    *,
    training_cfg: TrainingConfig,
    memory_paths: MemoryPaths,
    artifacts_root: Path,
    dataset_path: Path,
    api_key: Optional[str] = None,
    max_invalid_regens: int = 2,
    on_progress: ProgressCallback | None = None,
    get_actor_questions: GetActorQuestions | None = None,
) -> EpisodeResult:
    # Ensure `.env` is loaded (desktop_agent.config does this at import, but keep explicit).
    app_cfg = load_config()
    api_key = api_key or app_cfg.openai_api_key

    clients = TrainingClients.from_env(api_key=api_key)

    episode_id = _episode_id()

    riddler_agent = RiddlerAgent(clients=clients, model=training_cfg.models.riddler)
    actor_agent = ActorAgent(clients=clients, model=training_cfg.models.actor)
    partner_agent = PartnerAgent(clients=clients, model=training_cfg.models.partner)
    critic_agent = CriticAgent(clients=clients, model=training_cfg.models.critic)

    riddler_mem = read_tail(memory_paths.riddler, max_chars=6000)
    actor_mem = read_tail(memory_paths.actor, max_chars=6000)

    # --- Riddler -> Critic precheck (regenerate invalid riddles) ---
    last_precheck: dict[str, Any] | None = None
    riddle: RiddlerOutput | None = None
    for attempt in range(max_invalid_regens + 1):
        riddle = riddler_agent.generate(
            sweet_spot_pct=training_cfg.sweet_spot_pct,
            std_pct=training_cfg.std_pct,
            riddler_memory_excerpt=riddler_mem,
            riddle_style=training_cfg.riddle_style,
        )
        pre = critic_agent.precheck(riddle=riddle)
        last_precheck = pre
        if isinstance(pre, dict) and bool(pre.get("riddle_valid", False)):
            break
        if attempt >= max_invalid_regens:
            break

    if riddle is None:
        raise RuntimeError("failed to generate riddle")

    if on_progress:
        on_progress({"type": "riddle", "episode_id": episode_id, "riddle": riddle, "precheck": last_precheck or {}})

    # --- Actor proposes solver instruction profiles ---
    actor_out = actor_agent.propose_solver_profiles(
        solver_count=training_cfg.solver_count,
        riddle=riddle["riddle"],
        actor_memory_excerpt=actor_mem,
    )
    profiles = list(actor_out["solver_instruction_profiles"])

    if on_progress:
        on_progress({"type": "actor_profiles", "episode_id": episode_id, "actor_output": actor_out})

    # --- Run solvers (with optional partner help) ---
    solver_runs: list[dict[str, Any]] = []
    total_partners_used = 0
    streamed_correct = 0
    streamed_graded = 0

    for i in range(training_cfg.solver_count):
        # Optional: allow the user to "talk to the actor" during the run (non-blocking).
        if get_actor_questions is not None:
            try:
                qs0 = list(get_actor_questions() or [])
            except Exception:
                qs0 = []
            for q0 in qs0:
                q_txt0 = str(q0 or "").strip()
                if not q_txt0:
                    continue
                try:
                    ans0 = actor_agent.answer_question(
                        riddle_id=riddle["riddle_id"],
                        riddle=riddle["riddle"],
                        solver_runs_so_far=solver_runs,
                        remaining_solver_ids=[f"s{j+1:02d}" for j in range(i, training_cfg.solver_count)],
                        question=q_txt0,
                    )
                    if on_progress:
                        on_progress(
                            {
                                "type": "actor_answer",
                                "episode_id": episode_id,
                                "question": q_txt0,
                                "answer": str(ans0.get("answer", "") or ""),
                            }
                        )
                except Exception as e:
                    if on_progress:
                        on_progress(
                            {
                                "type": "actor_answer",
                                "episode_id": episode_id,
                                "question": q_txt0,
                                "answer": f"(error asking actor) {e}",
                            }
                        )

        solver_id = f"s{i+1:02d}"
        profile = profiles[i]
        solver = SolverAgent(clients=clients, model=training_cfg.models.solver, solver_id=solver_id)

        out = solver.solve(instruction_profile=profile, riddle=riddle["riddle"], partner_message=None)
        run_row: dict[str, Any] = {
            "solver_id": solver_id,
            "instruction_profile": profile,
            "final_answer": out["final_answer"],
            "derivation": out["derivation"],
            "confidence": out["confidence"],
            "checks": out["checks"],
            "asked_partner": False,
            "partner_summary": "",
        }

        partners_used_for_solver = 0
        if (
            out["help_request"]["needed"]
            and total_partners_used < training_cfg.max_partners_total
            and partners_used_for_solver < training_cfg.max_partners_per_solver
        ):
            partner = partner_agent.advise(
                riddle=riddle["riddle"],
                solver_profile=profile,
                what_i_tried=out["help_request"].get("what_i_tried", ""),
                where_stuck=out["help_request"].get("where_i_am_stuck", ""),
            )
            partner_msg = str(partner.get("partner_message", "") or "").strip()
            total_partners_used += 1
            partners_used_for_solver += 1

            out2 = solver.solve(instruction_profile=profile, riddle=riddle["riddle"], partner_message=partner_msg)
            run_row["asked_partner"] = True
            run_row["partner_summary"] = partner_msg
            run_row["final_answer"] = out2["final_answer"]
            run_row["derivation"] = out2["derivation"]
            run_row["confidence"] = out2["confidence"]
            run_row["checks"] = out2["checks"]

        solver_runs.append(run_row)

        if on_progress:
            on_progress(
                {
                    "type": "solver_done",
                    "episode_id": episode_id,
                    "i": i + 1,
                    "n": training_cfg.solver_count,
                    "solver_id": solver_id,
                    "asked_partner": bool(run_row.get("asked_partner", False)),
                    "confidence": float(run_row.get("confidence", 0.0)),
                }
            )

        # Optional: stream-grade each solver so we can show "solved so far" progress.
        if training_cfg.stream_grade_each_solver:
            try:
                row_grade = critic_agent.grade_single(riddle=riddle, solver_run=run_row, reward_cfg=training_cfg)
                streamed_graded += 1
                if bool(row_grade.get("correct", False)):
                    streamed_correct += 1
                if on_progress:
                    on_progress(
                        {
                            "type": "stream_grade",
                            "episode_id": episode_id,
                            "solver_id": str(row_grade.get("solver_id", solver_id)),
                            "correct": bool(row_grade.get("correct", False)),
                            "notes": str(row_grade.get("notes", "") or ""),
                            "graded": streamed_graded,
                            "correct_so_far": streamed_correct,
                        }
                    )
            except Exception:
                # Best-effort; skip streaming grade failures.
                pass

        # Optional: periodic actor status report.
        if training_cfg.status_every_solvers > 0 and ((i + 1) % training_cfg.status_every_solvers == 0):
            try:
                remaining = [f"s{j+1:02d}" for j in range(i + 1, training_cfg.solver_count)]
                status = actor_agent.status_report(
                    riddle_id=riddle["riddle_id"],
                    riddle=riddle["riddle"],
                    solver_runs_so_far=solver_runs,
                    remaining_solver_ids=remaining,
                )
                if on_progress:
                    on_progress({"type": "actor_status", "episode_id": episode_id, "at": i + 1, "status": status})
            except Exception:
                pass

        # Optional: allow the user to "talk to the actor" during the run (non-blocking).
        if get_actor_questions is not None:
            try:
                qs = list(get_actor_questions() or [])
            except Exception:
                qs = []
            for q in qs:
                q_txt = str(q or "").strip()
                if not q_txt:
                    continue
                try:
                    ans = actor_agent.answer_question(
                        riddle_id=riddle["riddle_id"],
                        riddle=riddle["riddle"],
                        solver_runs_so_far=solver_runs,
                        remaining_solver_ids=[f"s{j+1:02d}" for j in range(i + 1, training_cfg.solver_count)],
                        question=q_txt,
                    )
                    if on_progress:
                        on_progress(
                            {
                                "type": "actor_answer",
                                "episode_id": episode_id,
                                "question": q_txt,
                                "answer": str(ans.get("answer", "") or ""),
                            }
                        )
                except Exception as e:
                    if on_progress:
                        on_progress(
                            {
                                "type": "actor_answer",
                                "episode_id": episode_id,
                                "question": q_txt,
                                "answer": f"(error asking actor) {e}",
                            }
                        )

    # --- Critic grades ---
    expected_ids = [r["solver_id"] for r in solver_runs]
    critic_report = critic_agent.grade(
        riddle=riddle,
        solver_runs=solver_runs,
        expected_solver_ids=expected_ids,
        reward_cfg=training_cfg,
    )

    correct_count = sum(1 for g in critic_report["grading"] if g.get("correct") is True)
    computed_solve_rate = (float(correct_count) / max(1, len(expected_ids))) * 100.0
    computed_reward = gaussian_reward(
        solve_rate_pct=computed_solve_rate,
        sweet_spot_pct=training_cfg.sweet_spot_pct,
        std_pct=training_cfg.std_pct,
    )

    # --- Save artifacts ---
    ep_dir = artifacts_root / f"episode_{episode_id}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    _write_json(ep_dir / "riddle.json", riddle)
    _write_jsonl(ep_dir / "solver_runs.jsonl", solver_runs)
    _write_json(ep_dir / "critic_report.json", critic_report)
    _write_json(
        ep_dir / "episode_summary.json",
        {
            "episode_id": episode_id,
            "precheck": last_precheck or {},
            "correct_count": correct_count,
            "solver_count": len(expected_ids),
            "computed_solve_rate_pct": computed_solve_rate,
            "computed_reward": computed_reward,
            "critic_solve_rate_pct": float(critic_report.get("solve_rate_pct", 0.0)),
            "critic_reward": float(critic_report.get("reward", 0.0)),
            "streamed_graded": streamed_graded,
            "streamed_correct": streamed_correct,
        },
    )

    # --- Append dataset row (best derivation among correct solvers) ---
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    best_solver_id = str(critic_report.get("best_solver_id", "") or "").strip()
    best_derivation = str(critic_report.get("best_derivation", "") or "").strip()
    if best_solver_id and best_derivation:
        dataset_row = {
            "ts_utc": utc_ts(),
            "episode_id": episode_id,
            "riddle_id": riddle["riddle_id"],
            "tags": riddle.get("tags", []),
            "riddle": riddle["riddle"],
            "answer": riddle["solution"],
            "grading_criteria": riddle["grading_criteria"],
            "best_solver_id": best_solver_id,
            "best_derivation": best_derivation,
            "solve_rate_pct": computed_solve_rate,
            "reward": computed_reward,
        }
        with dataset_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dataset_row, ensure_ascii=False) + "\n")
        if on_progress:
            on_progress(
                {
                    "type": "dataset_append",
                    "episode_id": episode_id,
                    "path": str(dataset_path),
                    "best_solver_id": best_solver_id,
                }
            )

    # --- Update memories ---
    riddler_title = f"{utc_ts()} — episode {episode_id} — {riddle['riddle_id']}"
    riddler_body = (
        f"Tags: {', '.join(riddle.get('tags', []))}\n\n"
        f"Difficulty guess: {riddle.get('difficulty_guess_pct')}\n\n"
        f"Solve-rate: {computed_solve_rate:.1f}%\n\n"
        f"Reward (gaussian): {computed_reward:.4f}\n\n"
        "Notes:\n"
        f"- Critic validity: {bool(critic_report.get('riddle_valid', False))}\n"
        f"- Critic notes: {critic_report.get('validity_notes','')}\n"
    )
    append_md_section(memory_paths.riddler, title=riddler_title, body=riddler_body)

    summary_obj = actor_agent.summarize_episode(
        riddle_id=riddle["riddle_id"],
        riddle=riddle["riddle"],
        solver_runs=solver_runs,
        critic_grading=critic_report["grading"],
    )
    actor_title = str(summary_obj.get("memory_append_title") or "").strip() or f"{utc_ts()} — episode {episode_id}"
    actor_body = str(summary_obj.get("memory_append_body") or "").strip()
    takeaways = summary_obj.get("instruction_takeaways")
    if isinstance(takeaways, list) and takeaways:
        actor_body = actor_body + "\n\nTakeaways:\n" + "\n".join(f"- {str(t).strip()}" for t in takeaways if str(t).strip())
    append_md_section(memory_paths.actor, title=actor_title, body=actor_body)

    if on_progress:
        on_progress(
            {
                "type": "done",
                "episode_id": episode_id,
                "solve_rate_pct": computed_solve_rate,
                "reward": computed_reward,
            }
        )

    return EpisodeResult(
        episode_id=episode_id,
        artifacts_dir=ep_dir,
        riddler=riddle,
        solver_runs=solver_runs,
        critic=critic_report,
        computed_solve_rate_pct=computed_solve_rate,
        computed_reward=computed_reward,
    )
