from __future__ import annotations

import time


def test_conpty_terminal_smoke(tmp_path) -> None:
    # Optional test: skip if pywinpty isn't available on this machine.
    try:
        import winpty  # type: ignore  # noqa: F401
    except Exception:
        return

    from desktop_agent.terminal_agent_ui import _ConptyTerminal

    out: list[str] = []

    def on_chunk(s: str) -> None:
        out.append(s)

    t = _ConptyTerminal(initial_cwd=tmp_path, on_chunk=on_chunk)
    res = t.send_and_collect(block="Write-Output 'hello_conpty'", idle_ms=250, max_wait_s=5.0)
    # The prompt sentinel may be included; we just need our output to appear.
    joined = res.stdout + "".join(out)
    assert "hello_conpty" in joined

    # Reset shouldn't hang.
    start = time.monotonic()
    t.stop_and_reset()
    assert (time.monotonic() - start) < 5.0
