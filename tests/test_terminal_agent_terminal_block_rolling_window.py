from __future__ import annotations


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "input_text", "text": text}]}


def test_sanitize_terminal_blocks_keeps_only_latest_block() -> None:
    from desktop_agent.terminal_agent_ui import _sanitize_terminal_blocks_in_conversation

    conv = [
        _msg("user", "hi"),
        _msg("assistant", "a\n<Terminal>\nwhoami\n</Terminal>\n"),
        _msg("assistant", "b\n<Terminal>\npwd\n</Terminal>\n"),
    ]

    _sanitize_terminal_blocks_in_conversation(conv, latest_terminal_window=None)

    assert "<Terminal>" not in conv[1]["content"][0]["text"]
    assert "</Terminal>" not in conv[1]["content"][0]["text"]

    last_text = conv[2]["content"][0]["text"]
    assert "<Terminal>" in last_text
    assert "pwd" in last_text


def test_sanitize_terminal_blocks_truncates_to_last_20_lines() -> None:
    from desktop_agent.terminal_agent_ui import _sanitize_terminal_blocks_in_conversation

    conv = [
        _msg("user", "hi"),
        _msg("assistant", "a\n<Terminal>\nold\n</Terminal>\n"),
    ]

    latest = "\n".join([f"line{i}" for i in range(1, 221)])
    _sanitize_terminal_blocks_in_conversation(conv, latest_terminal_window=latest)

    txt = conv[1]["content"][0]["text"]
    assert "\nline1\n" not in txt
    assert "\nline20\n" not in txt
    assert "\nline21\n" in txt
    assert "\nline220\n" in txt
