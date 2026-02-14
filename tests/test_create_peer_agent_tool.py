from __future__ import annotations

import json


def test_create_peer_agent_tool_is_gated() -> None:
    from desktop_agent.tools import make_create_peer_agent_handler

    called = {"create": 0, "disarm": 0}

    def is_armed() -> bool:
        return False

    def disarm() -> None:
        called["disarm"] += 1

    def create_peer(name: str, model: str, system_prompt: str, memory_path: str | None) -> dict:
        called["create"] += 1
        return {"agent_id": "a1", "name": name, "model": model, "memory_path": memory_path}

    h = make_create_peer_agent_handler(is_armed=is_armed, disarm=disarm, create_peer=create_peer)
    out = json.loads(h({"name": "Friend", "model": "gpt-4.1-mini", "system_prompt": "hi"}))
    assert out["ok"] is False
    assert called["create"] == 0
    assert called["disarm"] == 0


def test_create_peer_agent_tool_calls_create_and_disarms() -> None:
    from desktop_agent.tools import make_create_peer_agent_handler

    called = {"create": 0, "disarm": 0}

    def is_armed() -> bool:
        return True

    def disarm() -> None:
        called["disarm"] += 1

    def create_peer(name: str, model: str, system_prompt: str, memory_path: str | None) -> dict:
        called["create"] += 1
        assert name == "Friend"
        assert model == "gpt-4.1-mini"
        assert system_prompt == "Hello"
        assert memory_path == "memory_friend.md"
        return {"agent_id": "a1"}

    h = make_create_peer_agent_handler(is_armed=is_armed, disarm=disarm, create_peer=create_peer)
    out = json.loads(
        h({"name": "Friend", "model": "gpt-4.1-mini", "system_prompt": "Hello", "memory_path": "memory_friend.md"})
    )
    assert out["ok"] is True
    assert out["agent_id"] == "a1"
    assert called["create"] == 1
    assert called["disarm"] == 1

