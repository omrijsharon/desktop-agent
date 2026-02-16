from __future__ import annotations


def test_agents_sdk_session_record_roundtrip(tmp_path) -> None:
    from desktop_agent.agent_sdk import AgentsSdkSession
    from desktop_agent.chat_session import ChatConfig

    cfg = ChatConfig(tool_base_dir=tmp_path)
    s = AgentsSdkSession(api_key="test", config=cfg)
    s.chat_id = "20260215T000000Z_ab12cd34"
    s.title = "Hello"
    s.set_system_prompt("sys")
    rec = s.to_record()

    s2 = AgentsSdkSession(api_key="test", config=ChatConfig(tool_base_dir=tmp_path))
    s2.load_record(rec)
    assert s2.chat_id == "20260215T000000Z_ab12cd34"
    assert s2.title == "Hello"
    assert s2.get_system_prompt() == "sys"
    assert bool(getattr(s2.cfg, "use_agents_sdk", False)) is True


def test_agents_sdk_session_has_expected_tools(tmp_path) -> None:
    from desktop_agent.agent_sdk import AgentsSdkSession
    from desktop_agent.chat_session import ChatConfig

    cfg = ChatConfig(tool_base_dir=tmp_path)
    s = AgentsSdkSession(api_key="test", config=cfg)
    names = set(s.registry.tools.keys())
    assert "read_file" in names
    assert "web_fetch" in names
    assert "playwright_browser" in names

