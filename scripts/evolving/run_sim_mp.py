from __future__ import annotations

import argparse
from pathlib import Path

from desktop_agent.evolving.config import load_evolving_config
from desktop_agent.evolving.engine_mp import RealtimeConfig, run_sim_mp
from desktop_agent.evolving.tensorboard_logger import TensorboardLogger, tensorboard_available


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the evolving agent community simulator (multiprocess pubsub).")
    ap.add_argument("--days", type=int, default=3, help="Number of simulated days to run.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (used for fake mode + initial population).")
    ap.add_argument(
        "--world-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "world"),
        help="Directory to store persistent world state and agent files.",
    )
    ap.add_argument("--day-seconds", type=float, default=120.0, help="Real-time seconds per simulated day.")
    ap.add_argument("--night-seconds", type=float, default=20.0, help="Real-time seconds for the night routine.")
    ap.add_argument("--night-warning-seconds", type=float, default=20.0, help="Warn agents this many seconds before night.")
    ap.add_argument("--agent-cooldown-seconds", type=float, default=1.0, help="Minimum real-time seconds between actions per agent.")
    ap.add_argument("--status-every-seconds", type=float, default=1.0, help="Print a status line every N seconds (0 disables).")
    ap.add_argument("--tb-logdir", type=str, default="", help="TensorBoard log directory (defaults to <world-dir>/tb).")
    ap.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")
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

    rt = RealtimeConfig(
        day_seconds=float(args.day_seconds),
        night_seconds=float(args.night_seconds),
        night_warning_seconds=float(args.night_warning_seconds),
        agent_cooldown_s=float(args.agent_cooldown_seconds),
        status_every_s=float(args.status_every_seconds),
    )

    def on_progress(evt: dict) -> None:
        t = evt.get("type")
        if t == "heartbeat":
            day = evt.get("day")
            phase = evt.get("phase")
            rem = float(evt.get("day_remaining_s", 0.0))
            alive = evt.get("alive")
            inflight = evt.get("inflight")
            births = evt.get("births")
            deaths = evt.get("deaths")
            coop = evt.get("coop")
            comp = evt.get("comp")
            line = f"day={day} phase={phase} remaining={rem:5.1f}s alive={alive} inflight={inflight} births={births} deaths={deaths} coop={coop} comp={comp}"
            print(line, end="\r", flush=True)
        elif t == "night_warning":
            print("\n[NARRATOR] Night is coming soon.", flush=True)
        elif t == "night_start":
            print("\n[NARRATOR] Night has started.", flush=True)
        elif t == "day_end":
            print("\n[NARRATOR] Day ended.", flush=True)

    try:
        run_sim_mp(
            cfg=cfg,
            root=Path(args.world_dir),
            days=int(args.days),
            seed=int(args.seed),
            realtime=rt,
            telemetry=tb,
            on_progress=on_progress,
        )
    finally:
        if tb is not None:
            tb.close()

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
