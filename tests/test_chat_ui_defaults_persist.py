from pathlib import Path


def test_chat_ui_defaults_roundtrip(tmp_path: Path) -> None:
    from desktop_agent.chat_session import ChatConfig
    from desktop_agent.ui_prefs import apply_overrides, load_defaults, save_defaults

    repo = tmp_path
    (repo / "chat_history").mkdir(parents=True, exist_ok=True)

    cfg = ChatConfig(enable_web_search=False, web_search_context_size="low", allow_write_files=True)
    save_defaults(repo, {"chat_config": {"enable_web_search": cfg.enable_web_search, "web_search_context_size": cfg.web_search_context_size}})

    d = load_defaults(repo)
    assert isinstance(d, dict)
    over = d.get("chat_config")
    assert isinstance(over, dict)

    cfg2 = ChatConfig()
    apply_overrides(cfg2, over)
    assert cfg2.enable_web_search is False
    assert cfg2.web_search_context_size == "low"

