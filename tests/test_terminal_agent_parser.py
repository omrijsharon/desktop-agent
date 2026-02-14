from __future__ import annotations


def test_terminal_block_extract_and_strip() -> None:
    from desktop_agent.terminal_agent_ui import extract_terminal_blocks, strip_terminal_blocks

    txt = "Hello\n<Terminal>\nwhoami\n</Terminal>\nDone\n<TERMINAL>pwd</TERMINAL>\n"
    blocks = extract_terminal_blocks(txt)
    assert blocks == ["whoami", "pwd"]
    stripped = strip_terminal_blocks(txt)
    assert "Hello" in stripped
    assert "Done" in stripped
    assert "whoami" not in stripped
    assert "pwd" not in stripped

