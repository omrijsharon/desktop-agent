from __future__ import annotations

import os
import time
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

    actor_inbox = ensure_file(here / "actor_inbox.md")
    actor_outbox = ensure_file(here / "actor_outbox.md")

    print(f"Curriculum episodes: {training_cfg.episodes}", flush=True)
    print(f"Actor chat: write to {actor_inbox} and read {actor_outbox}", flush=True)

    def get_actor_questions() -> list[str]:
        try:
            q = actor_inbox.read_text(encoding="utf-8").strip()
            if not q:
                return []
            actor_inbox.write_text("", encoding="utf-8")
            return [q]
        except Exception:
            return []

    for ep in range(1, max(1, training_cfg.episodes) + 1):
        print(f"\n===== EPISODE {ep}/{training_cfg.episodes} =====\n", flush=True)
        res = run_episode(
            training_cfg=training_cfg,
            memory_paths=MemoryPaths(riddler=riddler_mem, actor=actor_mem),
            artifacts_root=artifacts_root,
            dataset_path=dataset_path,
            api_key=os.environ.get("OPENAI_API_KEY"),
            get_actor_questions=get_actor_questions,
        )
        print(f"Episode: {res.episode_id}", flush=True)
        print(f"Solve-rate: {res.computed_solve_rate_pct:.1f}% | Reward: {res.computed_reward:.4f}", flush=True)
        if ep < training_cfg.episodes and training_cfg.sleep_seconds_between_episodes > 0:
            time.sleep(float(training_cfg.sleep_seconds_between_episodes))

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

