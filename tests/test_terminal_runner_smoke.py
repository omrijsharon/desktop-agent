from __future__ import annotations

import time
from pathlib import Path


def test_terminal_runner_smoke(tmp_path) -> None:
    from desktop_agent.terminal_agent_ui import _TerminalRunner

    t = _TerminalRunner(initial_cwd=tmp_path)
    res = t.send_and_collect(block="Write-Output 'hello'", idle_ms=250, max_wait_s=5.0)
    assert "hello" in res.stdout

    # Stop/restart is quick.
    start = time.monotonic()
    t.stop_and_reset()
    assert (time.monotonic() - start) < 3.0
