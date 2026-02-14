from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class _Cfg:
    model: str = "m0"


class _FakeSession:
    def __init__(self, chat_id: str, model: str) -> None:
        self.chat_id = chat_id
        self.cfg = _Cfg(model=model)
        self._system = ""

    @property
    def system_prompt(self) -> str:
        return self._system

    def set_system_prompt(self, s: str) -> None:
        self._system = str(s)

    def add_to_system_prompt(self, text: str, *, separator: str = "\n\n") -> None:
        self._system = (self._system.rstrip() + separator + str(text).strip()).strip()


def test_agent_hub_unique_names_and_memory(tmp_path: Path) -> None:
    from desktop_agent.agent_hub import AgentHub, AgentHubError  # noqa: WPS433
    from desktop_agent.chat_session import ChatConfig  # noqa: WPS433

    ids = iter(["a1", "a2", "a3"])

    def make_session(cfg: ChatConfig) -> Any:
        return _FakeSession(next(ids), cfg.model)

    base = ChatConfig(model="m-main", tool_base_dir=tmp_path)
    main = _FakeSession("main", "m-main")
    main.set_system_prompt("p-main")
    hub = AgentHub(base_config=base, make_session=make_session, repo_root=tmp_path)
    hub.set_main(agent_id="main", name="Main", session=main)

    a = hub.create_peer(name="Friend", model="m1", system_prompt="hi", memory_path="")
    assert a.name == "Friend"
    assert (tmp_path / "chat_history").exists()
    assert a.memory_path and a.memory_path.endswith(".md")
    assert (tmp_path / Path(a.memory_path)).exists()

    try:
        hub.create_peer(name="friend", model="m2", system_prompt="x", memory_path=None)
        raise AssertionError("expected duplicate name error")
    except AgentHubError:
        pass


def test_agent_hub_export_import(tmp_path: Path) -> None:
    from desktop_agent.agent_hub import AgentHub  # noqa: WPS433
    from desktop_agent.chat_session import ChatConfig  # noqa: WPS433

    ids = iter(["a1", "a2", "a3", "a4"])

    def make_session(cfg: ChatConfig) -> Any:
        return _FakeSession(next(ids), cfg.model)

    base = ChatConfig(model="m-main", tool_base_dir=tmp_path)
    main = _FakeSession("main", "m-main")
    main.set_system_prompt("p-main")
    hub = AgentHub(base_config=base, make_session=make_session, repo_root=tmp_path)
    hub.set_main(agent_id="main", name="Main", session=main)
    hub.create_peer(name="A", model="m1", system_prompt="p1", memory_path=None)
    hub.create_peer(name="B", model="m2", system_prompt="p2", memory_path=None)

    data = hub.export_config()
    hub.load_config(data)
    names = [a.name for a in hub.list_agents()]
    assert names[0] == "Main"
    assert set(names[1:]) == {"A", "B"}
    # Main base system prompt survives export/import.
    assert hub.get(hub.main_agent_id()).system_prompt == "p-main"
