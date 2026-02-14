from __future__ import annotations


def test_agent_hub_state_roundtrip(tmp_path) -> None:
    from desktop_agent.agent_hub import AgentHub
    from desktop_agent.chat_session import ChatConfig, ChatSession

    base_cfg = ChatConfig(model="gpt-4.1-mini", tool_base_dir=tmp_path, allow_submodels=False)

    def make_session(cfg: ChatConfig) -> ChatSession:
        # Avoid submodels threads for tests.
        cfg = ChatConfig(**vars(cfg))
        cfg.allow_submodels = False
        return ChatSession(api_key="test", config=cfg)

    main = make_session(base_cfg)
    main.chat_id = "main_123456"
    main.title = "Main chat"
    main.set_system_prompt("MAIN_SP")
    main.load_record(
        {
            "chat_id": main.chat_id,
            "title": main.title,
            "model": base_cfg.model,
            "system_prompt": "MAIN_SP",
            "conversation": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        }
    )

    hub = AgentHub(base_config=base_cfg, make_session=make_session, repo_root=tmp_path)
    hub.set_main(agent_id=main.chat_id, name="Main", session=main)
    a = hub.create_peer(name="Friend", model="gpt-4.1-mini", system_prompt="FRIEND_SP", memory_path=None)
    # Seed friend conversation by loading a record.
    a.session.load_record(
        {
            "chat_id": a.session.chat_id,
            "title": "Friend chat",
            "model": "gpt-4.1-mini",
            "system_prompt": a.session.system_prompt,
            "conversation": [{"role": "assistant", "content": [{"type": "input_text", "text": "hello"}]}],
        }
    )

    state = hub.export_state()

    # Load into a new hub/main session.
    main2 = make_session(base_cfg)
    main2.chat_id = "main_123456"
    hub2 = AgentHub(base_config=base_cfg, make_session=make_session, repo_root=tmp_path)
    hub2.set_main(agent_id=main2.chat_id, name="Main", session=main2)
    hub2.load_state(state)

    agents = hub2.list_agents()
    assert len(agents) == 2
    assert agents[0].name == "Main"
    assert agents[1].name == "Friend"
    # Friend conversation restored.
    rec2 = agents[1].session.to_record()
    conv2 = rec2.get("conversation")
    assert isinstance(conv2, list) and conv2

