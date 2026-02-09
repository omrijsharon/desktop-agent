from __future__ import annotations

import argparse
from pathlib import Path

from desktop_agent.evolving.config import load_evolving_config
from desktop_agent.evolving.engine import run_sim
from desktop_agent.evolving.tensorboard_logger import TensorboardLogger, tensorboard_available


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the evolving agent community simulator.")
    ap.add_argument("--days", type=int, default=3, help="Number of days to simulate.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (used for initial genders/children IDs in fake mode).")
    ap.add_argument(
        "--world-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "world"),
        help="Directory to store persistent world state and agent files.",
    )
    ap.add_argument(
        "--tb-logdir",
        type=str,
        default="",
        help="TensorBoard log directory (defaults to <world-dir>/tb).",
    )
    ap.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging even if tensorboard is installed.",
    )
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    cfg = load_evolving_config(here / "config.json")

    tb = None
    if not args.no_tensorboard and tensorboard_available():
        logdir = Path(args.tb_logdir) if args.tb_logdir else (Path(args.world_dir) / "tb")
        tb = TensorboardLogger(logdir=logdir)
        print(f"TensorBoard logs: {logdir}", flush=True)
        print(f"Run: tensorboard --logdir \"{logdir}\" --port 6006", flush=True)
        print("Then open: http://localhost:6006", flush=True)

    def on_progress(evt: dict) -> None:
        t = evt.get("type")
        if t == "day_start":
            print(f"\n=== DAY {evt.get('day')} ===\n", flush=True)
        elif t == "step":
            actor = evt.get("actor_id")
            op = evt.get("op")
            tgt = evt.get("target_id") or ""
            energy = float(evt.get("energy", 0.0))
            alive = bool(evt.get("alive", True))
            suf = "" if not tgt else f" -> {tgt}"
            dead = "" if alive else " [DEAD]"
            print(f"[day {evt.get('day')} step {evt.get('step_in_day')}] {actor}: {op}{suf} (E={energy:.1f}){dead}", flush=True)

    try:
        run_sim(
            cfg=cfg,
            root=Path(args.world_dir),
            days=int(args.days),
            seed=int(args.seed),
            on_progress=on_progress,
            telemetry=tb,
        )
    finally:
        if tb is not None:
            tb.close()
    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
