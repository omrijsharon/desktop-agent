from __future__ import annotations

from typing import Any, Iterator


def test_send_stream_soft_fails_on_stream_disconnect(monkeypatch, tmp_path) -> None:
    import httpx

    from desktop_agent.chat_session import ChatConfig, ChatSession

    cfg = ChatConfig(
        model="gpt-4.1-mini",
        tool_base_dir=tmp_path,
        allow_submodels=False,
    )
    session = ChatSession(api_key="test", config=cfg)
    try:

        def fake_stream_runner(*args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
            # Emit some deltas via callback, then crash like a dropped SSE stream.
            cb = kwargs.get("on_text_delta")
            assert cb is not None
            cb("hello ")
            cb("world")
            raise httpx.RemoteProtocolError("incomplete chunked read")
            yield {}  # pragma: no cover

        monkeypatch.setattr("desktop_agent.chat_session.run_responses_with_function_tools_stream", fake_stream_runner)

        evs = list(session.send_stream("hi"))
        types = [e.get("type") for e in evs]
        assert "assistant_delta" in types
        assert "error" in types
        assert types[-1] == "turn_done"

        # Partial assistant text is preserved with an interruption note.
        last = session._conversation[-1]
        assert last.get("role") == "assistant"
        content = (last.get("content") or [{}])[0].get("text", "")
        assert "hello world" in content
        assert "stream interrupted" in content
    finally:
        session._stop_submodel_threads()

