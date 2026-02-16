"""desktop_agent.process_manager

Single-command launcher for the Desktop Agent "stack" so you don't need
multiple terminals.

Typical usage:
  .\\.venv\\Scripts\\python.exe -m desktop_agent.process_manager

This starts:
- Telegram relay (headless)
- Chat UI (GUI)

Notes:
- All subprocesses run under the same Python interpreter (`sys.executable`),
  so your venv is honored.
- Relay stdout/stderr are prefixed in this terminal for easy debugging.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional


def _prefix_stream(proc: subprocess.Popen[str], name: str) -> None:
    """Read combined stdout/stderr line-by-line and prefix to this stdout."""

    try:
        assert proc.stdout is not None  # for type checkers
        for raw in iter(proc.stdout.readline, ""):
            if raw == "":
                break
            line = raw.rstrip("\r\n")
            if line:
                print(f"[{name}] {line}", flush=True)
    except Exception:
        return


@dataclass(frozen=True)
class Child:
    name: str
    args: list[str]
    proc: subprocess.Popen[str]


def _start_child(*, name: str, args: list[str]) -> Child:
    # CREATE_NEW_PROCESS_GROUP helps keep Ctrl+C for the manager without
    # instantly killing GUI apps; we still terminate children on exit.
    creationflags = 0
    try:
        creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    except Exception:
        creationflags = 0

    proc = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    t = threading.Thread(target=_prefix_stream, args=(proc, name), daemon=True)
    t.start()
    return Child(name=name, args=args, proc=proc)


def _terminate_children(children: Iterable[Child]) -> None:
    for c in children:
        try:
            if c.proc.poll() is None:
                c.proc.terminate()
        except Exception:
            pass
    # Give a short grace period.
    deadline = time.time() + 3.0
    for c in children:
        try:
            if c.proc.poll() is None:
                remaining = max(0.0, deadline - time.time())
                c.proc.wait(timeout=remaining)
        except Exception:
            pass
    # Hard kill if needed.
    for c in children:
        try:
            if c.proc.poll() is None:
                c.proc.kill()
        except Exception:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="desktop_agent.process_manager")
    ap.add_argument("--no-telegram", action="store_true", help="Do not start Telegram relay")
    ap.add_argument("--no-chat-ui", action="store_true", help="Do not start Chat UI")
    args = ap.parse_args(argv)

    if not args.no_telegram and not (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip():
        print("[manager] WARNING: TELEGRAM_BOT_TOKEN is not set. Telegram relay will exit.", flush=True)

    children: list[Child] = []
    try:
        if not args.no_telegram:
            children.append(
                _start_child(name="telegram", args=[sys.executable, "-m", "desktop_agent.telegram_relay"])
            )
        if not args.no_chat_ui:
            # Chat UI: GUI app; logs go to chat_history/chat_ui.log (and some stdout).
            children.append(_start_child(name="chat_ui", args=[sys.executable, "-m", "desktop_agent.chat_ui"]))

        if not children:
            print("[manager] Nothing to run (both --no-telegram and --no-chat-ui set).", flush=True)
            return 2

        # Main loop: exit if all children exit.
        while True:
            alive = [c for c in children if c.proc.poll() is None]
            if not alive:
                return 0
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("[manager] Stopping…", flush=True)
        return 0
    finally:
        _terminate_children(children)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

