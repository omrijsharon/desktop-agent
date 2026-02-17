from __future__ import annotations


def test_terminal_agent_ssh_file_tools_specs_are_strict_objects() -> None:
    from desktop_agent.terminal_agent_ui import _ssh_read_file_tool_spec, _ssh_replace_line_tool_spec, _ssh_write_file_tool_spec

    for spec in (_ssh_read_file_tool_spec(), _ssh_write_file_tool_spec(), _ssh_replace_line_tool_spec()):
        assert spec["type"] == "function"
        params = spec["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
