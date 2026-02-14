from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .chat_session import ChatConfig


JsonDict = dict[str, Any]


class AgentHubError(RuntimeError):
    pass


@dataclass
class PeerAgent:
    agent_id: str
    name: str
    session: Any  # ChatSession
    model: str
    system_prompt: str
    memory_path: str | None = None  # repo-relative, if enabled


def _norm_name(name: str) -> str:
    return " ".join(str(name or "").strip().split()).casefold()


def _default_memory_path(agent_id: str) -> str:
    return f"chat_history/memory_agent_{agent_id}.md"


def _memory_note(mem_rel: str) -> str:
    return (
        f"Optional memory file (repo-relative): {mem_rel}\n"
        "If you want to persist notes, update it when appropriate."
    )


def _with_memory_note(prompt: str, mem_rel: str | None) -> str:
    p = str(prompt or "")
    if not mem_rel:
        return p
    note = _memory_note(mem_rel)
    # Avoid duplicating the note (or the path) on repeated edits.
    if mem_rel in p or "Optional memory file (repo-relative):" in p:
        return p
    sep = "\n\n" if p.strip() else ""
    return p.rstrip() + sep + note


_IDENT_BEGIN = "\n[agent_identity_v1]\n"
_IDENT_END = "\n[/agent_identity_v1]\n"


def _strip_identity_footer(prompt: str) -> str:
    s = str(prompt or "")
    i = s.find(_IDENT_BEGIN)
    if i < 0:
        return s
    j = s.find(_IDENT_END, i + len(_IDENT_BEGIN))
    if j < 0:
        return s[:i].rstrip()
    return (s[:i] + s[j + len(_IDENT_END) :]).rstrip()


def _identity_footer(*, agent_name: str, roster_names: list[str]) -> str:
    roster = ", ".join([n for n in roster_names if str(n).strip()])
    return (
        _IDENT_BEGIN
        + f"Your name is: {agent_name}\n"
        + "You are one of several agents in a shared chat.\n"
        + "To explicitly schedule another agent to speak next, write @ followed by their name (e.g. @Friend).\n"
        + "If you mention an agent with @Name, the UI will try to make that agent's turn be next.\n"
        + "To pass your turn without speaking, output: #skip\n"
        + "Note: any tokens like #skip / #something (a # immediately followed by non-space) are treated as commands by the UI and are removed from future chat context.\n"
        + (f"Known agents: {roster}\n" if roster else "")
        + _IDENT_END
    )


def _compose_effective_prompt(*, base_prompt: str, agent_name: str, roster_names: list[str], mem_rel: str | None) -> str:
    p = _strip_identity_footer(str(base_prompt or "")).rstrip()
    p = _with_memory_note(p, mem_rel).rstrip()
    p = p + _identity_footer(agent_name=str(agent_name), roster_names=list(roster_names))
    return p.strip()


def _ensure_repo_relative_path(repo_root: Path, rel: str) -> str:
    rel = str(rel or "").replace("\\", "/").lstrip("./")
    if not rel:
        raise AgentHubError("memory_path must be non-empty")
    p = Path(rel)
    if p.is_absolute():
        raise AgentHubError("memory_path must be repo-relative")
    abs_path = (repo_root / p).resolve()
    repo_root = repo_root.resolve()
    try:
        if not abs_path.is_relative_to(repo_root):
            raise AgentHubError("memory_path escapes repo root")
    except AttributeError:
        if str(abs_path).lower().find(str(repo_root).lower()) != 0:
            raise AgentHubError("memory_path escapes repo root")
    return p.as_posix()


class AgentHub:
    """Manage same-level peer agents for a single UI chat."""

    def __init__(
        self,
        *,
        base_config: ChatConfig,
        make_session: Callable[[ChatConfig], Any],
        repo_root: Path,
    ) -> None:
        self._base_cfg = base_config
        self._make_session = make_session
        self._repo_root = Path(repo_root)
        self._lock = threading.Lock()
        self._agents: dict[str, PeerAgent] = {}
        self._main_agent_id: str | None = None

    def set_main(self, *, agent_id: str, name: str, session: Any) -> None:
        with self._lock:
            aid = str(agent_id)
            self._main_agent_id = aid
            base_prompt = str(getattr(session, "system_prompt", "") or getattr(session, "_system_prompt", ""))
            base_prompt = _strip_identity_footer(base_prompt)
            self._agents[aid] = PeerAgent(
                agent_id=aid,
                name=str(name),
                session=session,
                model=str(getattr(session.cfg, "model", "")),
                system_prompt=base_prompt,
                memory_path=None,
            )
        self._refresh_identity_prompts()

    def main_agent_id(self) -> str:
        if not self._main_agent_id:
            raise AgentHubError("main agent not set")
        return self._main_agent_id

    def list_agents(self) -> list[PeerAgent]:
        with self._lock:
            # Stable order: main first, then name
            main = self._agents.get(self._main_agent_id or "")
            peers = [a for a in self._agents.values() if a.agent_id != (self._main_agent_id or "")]
            peers = sorted(peers, key=lambda a: _norm_name(a.name))
            return ([main] if main is not None else []) + peers

    def get(self, agent_id: str) -> PeerAgent:
        with self._lock:
            a = self._agents.get(str(agent_id))
            if a is None:
                raise AgentHubError(f"unknown agent_id: {agent_id}")
            return a

    def _assert_unique_name(self, *, name: str, exclude_agent_id: str | None = None) -> None:
        nn = _norm_name(name)
        if not nn:
            raise AgentHubError("agent name must be non-empty")
        with self._lock:
            for aid, a in self._agents.items():
                if exclude_agent_id and str(aid) == str(exclude_agent_id):
                    continue
                if _norm_name(a.name) == nn:
                    raise AgentHubError(f"agent name already exists: {a.name}")

    def ensure_memory_file(self, *, agent_id: str, memory_path: str | None) -> str | None:
        if memory_path is None:
            return None
        if not str(memory_path).strip():
            memory_path = _default_memory_path(agent_id)
        rel = _ensure_repo_relative_path(self._repo_root, str(memory_path))
        abs_path = (self._repo_root / Path(rel)).resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        if not abs_path.exists():
            abs_path.write_text("", encoding="utf-8")
        return rel

    def _refresh_identity_prompts(self) -> None:
        # Snapshot under lock, then apply outside (avoid holding the lock while
        # calling into session setters).
        with self._lock:
            agents = list(self._agents.values())
        roster = [str(a.name) for a in agents if str(a.name).strip()]
        for a in agents:
            try:
                eff = _compose_effective_prompt(
                    base_prompt=str(a.system_prompt or ""),
                    agent_name=str(a.name or ""),
                    roster_names=roster,
                    mem_rel=a.memory_path,
                )
                a.session.set_system_prompt(eff)
            except Exception:
                pass

    def create_peer(
        self,
        *,
        name: str,
        model: str,
        system_prompt: str,
        memory_path: str | None,
    ) -> PeerAgent:
        self._assert_unique_name(name=str(name), exclude_agent_id=None)
        cfg = ChatConfig(**vars(self._base_cfg))
        cfg.model = str(model)
        sess = self._make_session(cfg)
        agent_id = str(getattr(sess, "chat_id", "") or "").strip()
        if not agent_id:
            raise AgentHubError("failed to create agent session id")

        mem = self.ensure_memory_file(agent_id=agent_id, memory_path=memory_path)

        a = PeerAgent(
            agent_id=agent_id,
            name=str(name),
            session=sess,
            model=str(model),
            system_prompt=_strip_identity_footer(str(system_prompt)),
            memory_path=mem,
        )
        with self._lock:
            self._agents[a.agent_id] = a
        self._refresh_identity_prompts()
        return a

    def update_peer(
        self,
        *,
        agent_id: str,
        name: str,
        model: str,
        system_prompt: str,
        memory_path: str | None,
    ) -> PeerAgent:
        aid = str(agent_id)
        self._assert_unique_name(name=str(name), exclude_agent_id=aid)
        with self._lock:
            a = self._agents.get(aid)
            if a is None or aid == (self._main_agent_id or ""):
                raise AgentHubError("unknown or immutable agent")

        mem = self.ensure_memory_file(agent_id=aid, memory_path=memory_path)
        a = self.get(aid)
        a.name = str(name)
        a.model = str(model)
        a.system_prompt = _strip_identity_footer(str(system_prompt))
        a.memory_path = mem
        try:
            a.session.cfg.model = str(model)
        except Exception:
            pass
        self._refresh_identity_prompts()
        return a

    def update_main(
        self,
        *,
        model: str,
        system_prompt: str,
        memory_path: str | None,
    ) -> PeerAgent:
        aid = self.main_agent_id()
        mem = self.ensure_memory_file(agent_id=aid, memory_path=memory_path)
        a = self.get(aid)
        a.model = str(model)
        a.system_prompt = _strip_identity_footer(str(system_prompt))
        a.memory_path = mem
        try:
            a.session.cfg.model = str(model)
        except Exception:
            pass
        self._refresh_identity_prompts()
        return a

    def remove_peer(self, *, agent_id: str) -> None:
        aid = str(agent_id)
        with self._lock:
            if aid == (self._main_agent_id or ""):
                raise AgentHubError("cannot remove main agent")
            self._agents.pop(aid, None)
        self._refresh_identity_prompts()

    def remove_all_peers(self) -> None:
        with self._lock:
            main_id = self._main_agent_id
            self._agents = {main_id: self._agents[main_id]} if main_id and main_id in self._agents else {}
        self._refresh_identity_prompts()

    def export_config(self) -> JsonDict:
        main = self.get(self.main_agent_id())
        friends = []
        for a in self.list_agents():
            if a.agent_id == self.main_agent_id():
                continue
            friends.append(
                {
                    "name": a.name,
                    "model": a.model,
                    "system_prompt": a.system_prompt,
                    "memory_path": a.memory_path,
                }
            )
        return {
            "main": {"model": main.model, "system_prompt": str(main.system_prompt or ""), "memory_path": main.memory_path},
            "friends": friends,
        }

    def export_state(self) -> JsonDict:
        """Export full agent crew state (config + per-agent chat records).

        This is used to persist an entire multi-agent chat so reloading restores:
        - main system prompt/model
        - friend agents (name/model/system prompt/memory file)
        - each agent's ChatSession record (conversation + settings)
        """

        main = self.get(self.main_agent_id())
        out_friends: list[JsonDict] = []
        for a in self.list_agents():
            if a.agent_id == self.main_agent_id():
                continue
            rec = {}
            try:
                rec = a.session.to_record()  # type: ignore[attr-defined]
            except Exception:
                rec = {}
            out_friends.append(
                {
                    "agent_id": a.agent_id,
                    "name": a.name,
                    "model": a.model,
                    "system_prompt": a.system_prompt,
                    "memory_path": a.memory_path,
                    "record": rec,
                }
            )

        main_rec = {}
        try:
            main_rec = main.session.to_record()  # type: ignore[attr-defined]
        except Exception:
            main_rec = {}

        return {
            "main": {
                "agent_id": main.agent_id,
                "name": main.name,
                "model": main.model,
                "system_prompt": str(main.system_prompt or ""),
                "memory_path": main.memory_path,
                "record": main_rec,
            },
            "friends": out_friends,
        }

    def load_state(self, data: JsonDict) -> None:
        """Load full agent crew state (config + per-agent chat records)."""

        if not isinstance(data, dict):
            raise AgentHubError("agents_state must be an object")
        main_d = data.get("main") if isinstance(data.get("main"), dict) else {}
        friends = data.get("friends")
        if not isinstance(friends, list):
            friends = []

        # Reset peers and then restore.
        self.remove_all_peers()

        # Restore main config (do not replace the main session object; the UI owns it).
        try:
            cur_mem = self.get(self.main_agent_id()).memory_path
        except Exception:
            cur_mem = None
        mdl = str(main_d.get("model") or "").strip()
        sp = str(main_d.get("system_prompt") or "").strip()
        mem = main_d.get("memory_path")
        mem = str(mem).strip() if isinstance(mem, str) else cur_mem
        if mdl or sp or mem is not None:
            self.update_main(
                model=mdl or str(self.get(self.main_agent_id()).model),
                system_prompt=sp or str(self.get(self.main_agent_id()).system_prompt),
                memory_path=mem,
            )

        # Restore friend sessions (preserve their chat_id by loading their record).
        existing_names = {_norm_name(a.name) for a in self.list_agents()}
        for f in friends:
            if not isinstance(f, dict):
                continue
            nm = str(f.get("name") or "Friend").strip() or "Friend"
            base = nm
            i = 2
            while _norm_name(nm) in existing_names:
                nm = f"{base} {i}"
                i += 1
            existing_names.add(_norm_name(nm))

            mdl = str(f.get("model") or "").strip() or str(self._base_cfg.model)
            sp = str(f.get("system_prompt") or "You are a helpful assistant.").strip()
            mem = f.get("memory_path")
            mem = str(mem).strip() if isinstance(mem, str) else None
            rec = f.get("record") if isinstance(f.get("record"), dict) else None

            cfg = ChatConfig(**vars(self._base_cfg))
            cfg.model = mdl
            sess = self._make_session(cfg)
            if rec is not None:
                try:
                    sess.load_record(rec)  # type: ignore[attr-defined]
                except Exception:
                    pass

            agent_id = str(getattr(sess, "chat_id", "") or "").strip()
            mem2 = self.ensure_memory_file(agent_id=agent_id, memory_path=mem)
            a = PeerAgent(
                agent_id=agent_id,
                name=nm,
                session=sess,
                model=str(getattr(sess.cfg, "model", mdl)),
                system_prompt=_strip_identity_footer(sp),
                memory_path=mem2,
            )
            with self._lock:
                self._agents[a.agent_id] = a

        self._refresh_identity_prompts()

    def load_config(self, data: JsonDict) -> None:
        main = data.get("main")
        friends = data.get("friends")
        if not isinstance(friends, list):
            friends = []
        # Remove existing peers
        self.remove_all_peers()
        # Update main agent (model + base system prompt) if present.
        if isinstance(main, dict):
            mdl = str(main.get("model") or "").strip()
            sp = str(main.get("system_prompt") or "").strip()
            try:
                cur_mem = self.get(self.main_agent_id()).memory_path
            except Exception:
                cur_mem = None
            if mdl or sp:
                self.update_main(
                    model=mdl or str(self.get(self.main_agent_id()).model),
                    system_prompt=sp or str(self.get(self.main_agent_id()).system_prompt),
                    memory_path=cur_mem,
                )
        # Create peers
        existing_names = {_norm_name(a.name) for a in self.list_agents()}
        for f in friends:
            if not isinstance(f, dict):
                continue
            nm = str(f.get("name") or "Friend").strip() or "Friend"
            base = nm
            i = 2
            while _norm_name(nm) in existing_names:
                nm = f"{base} {i}"
                i += 1
            existing_names.add(_norm_name(nm))
            mdl = str(f.get("model") or "").strip() or str(self._base_cfg.model)
            sp = str(f.get("system_prompt") or "You are a helpful assistant.").strip()
            mem = f.get("memory_path")
            mem = str(mem).strip() if isinstance(mem, str) else None
            self.create_peer(name=nm, model=mdl, system_prompt=sp, memory_path=mem)


def save_agents_config(path: Path, data: JsonDict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_agents_config(path: Path) -> JsonDict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AgentHubError("agents config must be a JSON object")
    return data
