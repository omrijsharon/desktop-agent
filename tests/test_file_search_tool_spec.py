from __future__ import annotations


def test_extra_tools_includes_file_search_when_configured() -> None:
    from desktop_agent.chat_session import ChatConfig, ChatSession  # noqa: WPS433

    cfg = ChatConfig(
        enable_web_search=False,
        enable_file_search=True,
        file_search_vector_store_ids=["vs_123", "  ", "vs_456"],
        file_search_max_num_results=7,
    )
    s = ChatSession(api_key="sk-test", config=cfg)
    tools = s._extra_tools()  # noqa: SLF001 (unit test)
    assert {"type": "file_search", "vector_store_ids": ["vs_123", "vs_456"], "max_num_results": 7} in tools


def test_extra_tools_skips_file_search_without_vector_store_ids() -> None:
    from desktop_agent.chat_session import ChatConfig, ChatSession  # noqa: WPS433

    cfg = ChatConfig(enable_web_search=False, enable_file_search=True, file_search_vector_store_ids=[])
    s = ChatSession(api_key="sk-test", config=cfg)
    tools = s._extra_tools()  # noqa: SLF001 (unit test)
    assert all(t.get("type") != "file_search" for t in tools)

