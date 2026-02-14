from desktop_agent.chat_session import ChatConfig, ChatSession


def test_hash_commands_stripped_from_conversation_record(tmp_path):
    cfg = ChatConfig(tool_base_dir=tmp_path, enable_web_search=False, allow_submodels=False)
    s = ChatSession(api_key="test", config=cfg)

    rec = {
        "chat_id": "c1",
        "title": "t",
        "model": cfg.model,
        "system_prompt": "sp",
        "conversation": [
            {"role": "user", "content": [{"type": "input_text", "text": "hello #skip"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "#skip"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "# Heading stays"}]},
        ],
    }
    s.load_record(rec)
    out = s.to_record()["conversation"]

    # #commands (no space) removed
    assert "skip" not in str(out).casefold()
    # markdown heading "# Heading" (has a space) is preserved
    assert any(isinstance(x, dict) and "# Heading stays" in str(x) for x in out)

