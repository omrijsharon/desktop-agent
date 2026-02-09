from __future__ import annotations

import os
from pathlib import Path

from desktop_agent.training.config import load_training_config
from desktop_agent.training.episode import run_episode
from desktop_agent.training.memory import MemoryPaths, ensure_file


def main() -> int:
    here = Path(__file__).resolve().parent
    cfg_path = here / "config.json"
    artifacts_root = here / "artifacts"

    training_cfg = load_training_config(cfg_path)

    riddler_mem = ensure_file(here / "riddler_memory.md")
    actor_mem = ensure_file(here / "actor_memory.md")
    dataset_path = here / training_cfg.dataset_filename

    # Non-blocking "chat with actor" via files:
    # - Write a question into actor_inbox.md while the script runs.
    # - The runner will clear the inbox and append answers to actor_outbox.md.
    actor_inbox = ensure_file(here / "actor_inbox.md")
    actor_outbox = ensure_file(here / "actor_outbox.md")

    # Optional progress bar (created lazily after printing the riddle).
    tqdm = None
    try:
        from tqdm import tqdm as _tqdm  # type: ignore

        tqdm = _tqdm
    except Exception:
        tqdm = None

    pbar = None
    solved_so_far = 0
    current_episode_id = ""

    def _write_line(line: str) -> None:
        if tqdm is not None:
            try:
                tqdm.write(line)
                return
            except Exception:
                pass
        print(line, flush=True)

    def get_actor_questions() -> list[str]:
        try:
            q = actor_inbox.read_text(encoding="utf-8").strip()
            if not q:
                return []
            actor_inbox.write_text("", encoding="utf-8")
            return [q]
        except Exception:
            return []

    def on_progress(evt: dict) -> None:
        nonlocal solved_so_far
        nonlocal pbar
        nonlocal current_episode_id
        t = evt.get("type")
        if t == "riddle":
            r = evt.get("riddle", {})
            _write_line("")
            _write_line("=== RIDDLE ===")
            _write_line(str(r.get("riddle", "") or ""))
            _write_line("=============")
            _write_line("")
        elif t == "solver_done":
            if pbar is None and tqdm is not None:
                pbar = tqdm(total=training_cfg.solver_count, desc="Solvers", unit="solver", leave=True)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "id": evt.get("solver_id", ""),
                        "partner": bool(evt.get("asked_partner", False)),
                        "conf": f"{float(evt.get('confidence', 0.0)):.2f}",
                        "solved": solved_so_far if training_cfg.stream_grade_each_solver else "n/a",
                    }
                )
            else:
                idx = int(evt.get("i", 0))
                n = int(evt.get("n", training_cfg.solver_count))
                sid = evt.get("solver_id", "")
                conf = float(evt.get("confidence", 0.0))
                partner = " +partner" if bool(evt.get("asked_partner", False)) else ""
                solved = f", solved_so_far={solved_so_far}" if training_cfg.stream_grade_each_solver else ""
                _write_line(f"[{idx}/{n}] {sid} done (conf={conf:.2f}{partner}{solved})")
        elif t == "stream_grade":
            if bool(evt.get("correct", False)):
                solved_so_far = int(evt.get("correct_so_far", solved_so_far))
        elif t == "actor_status":
            st = evt.get("status", {})
            _write_line("")
            _write_line("=== ACTOR STATUS ===")
            _write_line(str(st.get("status_summary", "") or ""))
            likely = st.get("likely_to_succeed", [])
            if isinstance(likely, list) and likely:
                _write_line("Likely to succeed: " + ", ".join(str(x) for x in likely))
            risks = st.get("risks", [])
            if isinstance(risks, list) and risks:
                _write_line("Risks: " + "; ".join(str(x) for x in risks))
            _write_line("====================")
            _write_line("")
        elif t == "actor_answer":
            q = str(evt.get("question", "") or "").strip()
            a = str(evt.get("answer", "") or "").strip()
            current_episode_id = str(evt.get("episode_id", "") or current_episode_id)
            _write_line("")
            _write_line("=== ACTOR ANSWER ===")
            if q:
                _write_line("Q: " + q)
            if a:
                _write_line("A: " + a)
            _write_line("====================")
            _write_line("")
            try:
                actor_outbox.write_text(
                    actor_outbox.read_text(encoding="utf-8")
                    + f"\n\n## {current_episode_id or 'episode'} — {q}\n\n{a}\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
        elif t == "dataset_append":
            # Keep quiet; path printed at end.
            return

    print(f"Actor chat enabled. Write a question to: {actor_inbox}", flush=True)
    print(f"Actor answers will append to: {actor_outbox}", flush=True)

    res = run_episode(
        training_cfg=training_cfg,
        memory_paths=MemoryPaths(riddler=riddler_mem, actor=actor_mem),
        artifacts_root=artifacts_root,
        dataset_path=dataset_path,
        api_key=os.environ.get("OPENAI_API_KEY"),
        on_progress=on_progress,
        get_actor_questions=get_actor_questions,
    )
    if pbar is not None:
        pbar.close()

    print(f"Episode: {res.episode_id}")
    print(f"Riddle id: {res.riddler['riddle_id']}")
    print(f"Solve-rate: {res.computed_solve_rate_pct:.1f}%")
    print(f"Reward: {res.computed_reward:.4f}")
    print(f"Artifacts: {res.artifacts_dir}")
    print(f"Dataset (appended if available): {dataset_path}")
    print(f"Actor chat: write to {actor_inbox} and read {actor_outbox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
