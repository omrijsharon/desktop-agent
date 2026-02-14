from __future__ import annotations

import json
from pathlib import Path


def _write_chat(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_prune_empty_chats(tmp_path) -> None:
    from desktop_agent.chat_store import prune_empty_chats

    store = tmp_path / "chat_history"
    store.mkdir(parents=True, exist_ok=True)

    # Empty conversation -> pruned
    _write_chat(
        store / "c1.json",
        {"chat_id": "20260206T012233Z_ab12cd34", "title": "New chat", "conversation": []},
    )

    # No conversation key -> pruned
    _write_chat(store / "c2.json", {"chat_id": "20260206T012233Z_ab12cd35", "title": "New chat"})

    # Only whitespace -> pruned
    _write_chat(
        store / "c3.json",
        {
            "chat_id": "20260206T012233Z_ab12cd36",
            "title": "New chat",
            "conversation": [{"role": "user", "content": [{"type": "input_text", "text": "   \n"}]}],
        },
    )

    # Real chat -> kept
    _write_chat(
        store / "c4.json",
        {
            "chat_id": "20260206T012233Z_ab12cd37",
            "title": "Hello",
            "conversation": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        },
    )

    # Empty conversation but has agent crew -> kept
    _write_chat(
        store / "c5.json",
        {
            "chat_id": "20260206T012233Z_ab12cd38",
            "title": "Crew only",
            "conversation": [],
            "agents_state": {
                "main": {"agent_id": "20260206T012233Z_ab12cd38", "name": "Main", "model": "gpt-4.1", "system_prompt": "", "record": {}},
                "friends": [{"agent_id": "friend1", "name": "Friend", "model": "gpt-4.1-mini", "system_prompt": "", "record": {}}],
            },
        },
    )

    n = prune_empty_chats(tmp_path)
    assert n == 3
    remaining = {p.name for p in store.glob("*.json")}
    assert remaining == {"c4.json", "c5.json"}
