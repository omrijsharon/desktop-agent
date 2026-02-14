from __future__ import annotations


def test_powershell_prompt_at_end() -> None:
    from desktop_agent.terminal_agent_ui import _powershell_prompt_at_end

    assert _powershell_prompt_at_end("__TA_PROMPT__ C:\\> ")
    assert _powershell_prompt_at_end("x\ny\n__TA_PROMPT__ C:\\Users\\me> ")
    assert not _powershell_prompt_at_end("__TA_PROMPT__ C:\\> \nhello\n")
    assert not _powershell_prompt_at_end("omrijsharon@pi:~ $ ")

