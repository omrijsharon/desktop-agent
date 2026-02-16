import asyncio


def test_agents_sdk_session_send_stream_monkeypatched(monkeypatch, tmp_path):
    from agents import Agent
    from agents.run import Runner
    from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
    from agents.items import ToolCallItem, ToolCallOutputItem

    from desktop_agent.chat_session import ChatConfig
    from desktop_agent.agent_sdk.session import AgentsSdkSession

    class _FakeRun:
        def __init__(self, events):
            self._events = list(events)

        async def stream_events(self):
            for ev in self._events:
                await asyncio.sleep(0)
                yield ev

    def _fake_run_streamed(agent, input, max_turns=8):  # noqa: ARG001
        tool_call = ToolCallItem(
            agent=agent,
            raw_item={"name": "read_file", "arguments": '{"path":"README.md","start_line":1,"max_lines":1}'},
        )
        tool_out = ToolCallOutputItem(agent=agent, raw_item={}, output={"ok": True, "text": "hi"})
        events = [
            RawResponsesStreamEvent(data={"type": "response.output_text.delta", "delta": "Hello"}),
            RawResponsesStreamEvent(data={"type": "response.output_text.delta", "delta": " world"}),
            RunItemStreamEvent(name="tool_called", item=tool_call),
            RunItemStreamEvent(name="tool_output", item=tool_out),
        ]
        return _FakeRun(events)

    monkeypatch.setattr(Runner, "run_streamed", staticmethod(_fake_run_streamed))

    cfg = ChatConfig(
        model="gpt-4.1-mini",
        tool_base_dir=tmp_path,
        enable_web_search=False,
        allow_read_file=True,
        allow_write_files=False,
        allow_python_sandbox=False,
        enable_playwright_browser=False,
        allow_playwright_browser=False,
    )
    s = AgentsSdkSession(api_key="test", config=cfg)
    s.set_system_prompt("test")

    events = list(s.send_stream("hi"))
    deltas = "".join([e.get("delta", "") for e in events if e.get("type") == "assistant_delta"])
    assert deltas == "Hello world"

    td = [e for e in events if e.get("type") == "turn_done"]
    assert len(td) == 1
    new_items = td[0].get("new_items")
    assert isinstance(new_items, list)
    assert any(x.get("type") == "function_call" and x.get("name") == "read_file" for x in new_items if isinstance(x, dict))
    assert any(x.get("type") == "function_call_output" for x in new_items if isinstance(x, dict))

    usage = s.last_usage()
    assert int(usage.get("total_tokens", 0)) > 0
    assert "tok" in s.usage_ratio_text()
