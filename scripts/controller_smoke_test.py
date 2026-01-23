"""Smoke test for the controller module.

Runs the controller as a subprocess module (`python -m desktop_agent.controller`),
sends a ping, expects an ok response, and then terminates the process.

This is intended for CI/local quick verification. It does not generate real input.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time


def main() -> int:
    p = subprocess.Popen(
        [sys.executable, "-m", "desktop_agent.controller"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        # Wait for ready line
        deadline = time.time() + 5
        ready = None
        while time.time() < deadline:
            line = p.stdout.readline() if p.stdout else ""
            if not line:
                time.sleep(0.01)
                continue
            ready = json.loads(line)
            break

        if not isinstance(ready, dict) or not ready.get("ok") or ready.get("op") != "ready":
            return 2

        # Send ping
        assert p.stdin is not None
        p.stdin.write(json.dumps({"op": "ping"}) + "\n")
        p.stdin.flush()

        line = p.stdout.readline() if p.stdout else ""
        resp = json.loads(line) if line else None
        if not isinstance(resp, dict) or resp.get("ok") is not True or resp.get("op") != "ping":
            return 3

        return 0
    finally:
        try:
            p.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
