from __future__ import annotations

from desktop_agent.chat_session import ChatConfig, ChatSession


def test_load_record_with_offline_submodel() -> None:
    cfg = ChatConfig(allow_submodels=True, max_submodels=2, max_submodel_depth=1)
    s = ChatSession(api_key=None, config=cfg)

    rec = s.to_record()
    rec["submodels"] = {
        "sm1": {
            "chat_id": "sm1",
            "title": "Helper",
            "model": "gpt-x",
            "system_prompt": "SP",
            "state": "offline",
            "transcript": [{"role": "assistant", "text": "hi"}],
        }
    }

    s2 = ChatSession(api_key=None, config=cfg)
    s2.load_record(rec)
    metas = s2.list_submodels()
    assert len(metas) == 1
    assert metas[0]["title"] == "Helper"

