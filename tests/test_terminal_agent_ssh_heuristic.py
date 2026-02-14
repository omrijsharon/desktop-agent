from __future__ import annotations


def test_conpty_ssh_state_heuristic() -> None:
    from desktop_agent.terminal_agent_ui import _ConptyTerminal

    # Instantiate without starting a real PTY by bypassing __init__.
    t = object.__new__(_ConptyTerminal)
    t._in_ssh = False  # type: ignore[attr-defined]

    # SSH banner implies ssh mode.
    _ConptyTerminal._maybe_update_ssh_state_from_text(t, "Last login: Fri Feb 13 23:53:19 2026\n")  # type: ignore[arg-type]
    assert t._in_ssh is True  # type: ignore[attr-defined]

    # Linux prompt implies ssh mode.
    t._in_ssh = False  # type: ignore[attr-defined]
    _ConptyTerminal._maybe_update_ssh_state_from_text(t, "omrijsharon@omrijsharon:~ $ \n")  # type: ignore[arg-type]
    assert t._in_ssh is True  # type: ignore[attr-defined]

    # PowerShell prompt sentinel at end implies not in ssh.
    t._in_ssh = True  # type: ignore[attr-defined]
    _ConptyTerminal._maybe_update_ssh_state_from_text(t, "__TA_PROMPT__ C:\\Users\\me> ")  # type: ignore[arg-type]
    assert t._in_ssh is False  # type: ignore[attr-defined]

