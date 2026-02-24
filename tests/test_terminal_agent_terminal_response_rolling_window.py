from __future__ import annotations


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "input_text", "text": text}]}


def test_sanitize_terminal_responses_keeps_only_latest() -> None:
    from desktop_agent.terminal_agent_ui import _sanitize_terminal_responses_in_conversation

    conv = [
        _msg("user", "u1\n<TerminalResponse>\nold\n</TerminalResponse>\n"),
        _msg("assistant", "a"),
        _msg("user", "u2\n<TerminalResponse>\nnew\n</TerminalResponse>\n"),
    ]

    _sanitize_terminal_responses_in_conversation(conv, latest_terminal_response_window=None, max_lines=200)
    assert "<TerminalResponse>" not in conv[0]["content"][0]["text"]
    assert "old" not in conv[0]["content"][0]["text"]

    last = conv[2]["content"][0]["text"]
    assert "<TerminalResponse>" in last
    assert "new" in last


def test_sanitize_terminal_responses_truncates_to_last_200_lines() -> None:
    from desktop_agent.terminal_agent_ui import _sanitize_terminal_responses_in_conversation

    conv = [_msg("user", "x\n<TerminalResponse>\nold\n</TerminalResponse>\n")]
    latest = "\n".join([f"line{i}" for i in range(1, 221)])
    _sanitize_terminal_responses_in_conversation(conv, latest_terminal_response_window=latest, max_lines=200)
    txt = conv[0]["content"][0]["text"]
    assert "\nline1\n" not in txt
    assert "\nline20\n" not in txt
    assert "\nline21\n" in txt
    assert "\nline220\n" in txt

