from __future__ import annotations


def test_terminal_agent_wait_tool_spec_schema() -> None:
    from desktop_agent.terminal_agent_ui import _wait_tool_spec

    spec = _wait_tool_spec()
    assert spec["type"] == "function"
    assert spec["name"] == "wait"
    params = spec["parameters"]
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    assert "seconds" in params["properties"]


def test_terminal_agent_wait_tool_handler_clamps_without_sleeping_long() -> None:
    from desktop_agent.terminal_agent_ui import _wait_tool_handler

    # seconds=0 should return immediately
    out0 = _wait_tool_handler({"seconds": 0})
    assert '"ok": true' in out0.lower()

    # negative clamps to 0
    outn = _wait_tool_handler({"seconds": -5})
    assert '"ok": true' in outn.lower()

