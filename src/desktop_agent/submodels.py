"""desktop_agent.submodels

Manage auxiliary "sub-model" chat sessions spawned by a main chat session.

Design goals:
- Submodels have their own system prompt + conversation (separate context window).
- The main model can spawn submodels and send them tasks to parallelize work.
- Submodels can optionally spawn their own submodels (bounded by depth/limits).
- Everything is serializable so it can be saved into `chat_history/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .chat_store import new_chat_id


JsonDict = dict[str, Any]


@dataclass
class SubmodelConfig:
    max_instances: int = 6
    max_depth: int = 1  # 0 = no submodels, 1 = main->submodels, 2 = recursive, etc.


@dataclass
class SubmodelSession:
    chat_id: str = field(default_factory=new_chat_id)
    title: str = "Submodel"
    model: str = ""
    system_prompt: str = "You are a helpful assistant."
    conversation: list[JsonDict] = field(default_factory=list)

    # Nested submodels (optional)
    submodels: "SubmodelManager | None" = None

    def to_record(self) -> JsonDict:
        return {
            "chat_id": self.chat_id,
            "title": self.title,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "conversation": list(self.conversation),
            "submodels": self.submodels.to_record() if self.submodels is not None else None,
        }

    @staticmethod
    def from_record(rec: JsonDict) -> "SubmodelSession":
        s = SubmodelSession()
        s.chat_id = str(rec.get("chat_id") or new_chat_id())
        s.title = str(rec.get("title") or "Submodel")
        s.model = str(rec.get("model") or "")
        s.system_prompt = str(rec.get("system_prompt") or "You are a helpful assistant.")
        conv = rec.get("conversation")
        s.conversation = conv if isinstance(conv, list) else []
        sub = rec.get("submodels")
        if isinstance(sub, dict):
            s.submodels = SubmodelManager.from_record(sub)
        return s


@dataclass
class SubmodelManager:
    cfg: SubmodelConfig = field(default_factory=SubmodelConfig)
    depth: int = 0
    sessions: dict[str, SubmodelSession] = field(default_factory=dict)

    def can_spawn(self) -> bool:
        if self.depth >= self.cfg.max_depth:
            return False
        return len(self.sessions) < self.cfg.max_instances

    def get(self, chat_id: str) -> Optional[SubmodelSession]:
        return self.sessions.get(str(chat_id))

    def list_meta(self) -> list[JsonDict]:
        out: list[JsonDict] = []
        for sid, s in self.sessions.items():
            out.append({"id": sid, "title": s.title, "model": s.model})
        return out

    def create(
        self,
        *,
        title: str,
        system_prompt: str,
        model: str,
        allow_nested: bool,
    ) -> SubmodelSession:
        if not self.can_spawn():
            raise RuntimeError("submodel limit reached (or depth limit)")
        s = SubmodelSession(
            title=title or "Submodel",
            system_prompt=system_prompt or "You are a helpful assistant.",
            model=model or "",
        )
        if allow_nested and (self.depth + 1) <= self.cfg.max_depth:
            s.submodels = SubmodelManager(cfg=self.cfg, depth=self.depth + 1)
        self.sessions[s.chat_id] = s
        return s

    def delete(self, chat_id: str) -> None:
        self.sessions.pop(str(chat_id), None)

    def to_record(self) -> JsonDict:
        return {
            "cfg": {"max_instances": int(self.cfg.max_instances), "max_depth": int(self.cfg.max_depth)},
            "depth": int(self.depth),
            "sessions": {sid: s.to_record() for sid, s in self.sessions.items()},
        }

    @staticmethod
    def from_record(rec: JsonDict) -> "SubmodelManager":
        cfg_in = rec.get("cfg") if isinstance(rec, dict) else {}
        cfg = SubmodelConfig(
            max_instances=int(cfg_in.get("max_instances", 6)) if isinstance(cfg_in, dict) else 6,
            max_depth=int(cfg_in.get("max_depth", 1)) if isinstance(cfg_in, dict) else 1,
        )
        depth = int(rec.get("depth", 0)) if isinstance(rec, dict) else 0
        mgr = SubmodelManager(cfg=cfg, depth=depth, sessions={})
        sess = rec.get("sessions")
        if isinstance(sess, dict):
            for sid, srec in sess.items():
                if isinstance(srec, dict):
                    s = SubmodelSession.from_record(srec)
                    # Keep mapping key consistent.
                    mgr.sessions[str(sid)] = s
        return mgr


SubmodelRunner = Callable[[SubmodelSession, str], str]


def run_in_submodel(
    *,
    session: SubmodelSession,
    user_input: str,
    runner: SubmodelRunner,
) -> str:
    """Append a user message, run the model, append assistant response."""

    session.conversation.append({"role": "user", "content": [{"type": "input_text", "text": str(user_input)}]})
    out = runner(session, str(user_input))
    session.conversation.append({"role": "assistant", "content": [{"type": "output_text", "text": str(out)}]})
    return str(out)

