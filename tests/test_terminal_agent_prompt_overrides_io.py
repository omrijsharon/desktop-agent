from __future__ import annotations


def test_terminal_agent_prompt_overrides_roundtrip(tmp_path) -> None:
    from desktop_agent.terminal_agent_ui import _load_terminal_agent_prompt_overrides, _save_terminal_agent_prompt_overrides

    base = tmp_path
    overrides = {"Main": "hello", "T2": "world", "": "skip", "X": "   "}
    _save_terminal_agent_prompt_overrides(base_dir=base, overrides=overrides)

    loaded = _load_terminal_agent_prompt_overrides(base_dir=base)
    assert loaded == {"Main": "hello", "T2": "world"}

