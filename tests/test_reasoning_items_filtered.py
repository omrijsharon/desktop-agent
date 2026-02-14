from desktop_agent.chat_session import ChatConfig, ChatSession


def test_load_record_strips_reasoning_items(tmp_path):
    cfg = ChatConfig(tool_base_dir=tmp_path, enable_web_search=False, allow_submodels=False)
    s = ChatSession(api_key="test", config=cfg)

    rec = {
        "chat_id": "c1",
        "title": "t",
        "model": cfg.model,
        "system_prompt": "sp",
        "conversation": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "reasoning", "id": "rs_123", "content": [{"type": "text", "text": "secret"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
        ],
    }
    s.load_record(rec)

    conv = s.to_record()["conversation"]
    assert all(not (isinstance(x, dict) and x.get("type") == "reasoning") for x in conv)
