from __future__ import annotations

import json


def test_playwright_eval_is_gated(monkeypatch) -> None:
    from desktop_agent.chat_session import ChatConfig, ChatSession

    cfg = ChatConfig(
        enable_playwright_browser=True,
        allow_playwright_browser=True,
        allow_playwright_eval=False,
    )
    s = ChatSession(api_key="test", config=cfg)
    try:
        h = s.registry.handlers.get("playwright_browser")
        assert h is not None
        out = json.loads(h({"action": "browser_evaluate", "params": {"expression": "1+1"}}))
        assert out["ok"] is False
        assert "disabled" in str(out.get("error", "")).lower()
    finally:
        s.shutdown()

