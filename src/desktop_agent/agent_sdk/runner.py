"""desktop_agent.agent_sdk.runner

Small abstraction layer so the UI can swap between:
- legacy `ChatSession` (Responses API tool loop)
- `AgentsSdkSession` (OpenAI Agents SDK)

Both sessions already expose a similar interface, but this wrapper provides a
stable type/shape and a single factory to pick the right engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, runtime_checkable


JsonDict = dict[str, Any]


@runtime_checkable
class ChatEngine(Protocol):
    cfg: Any

    def send_stream(self, user_text: str) -> Iterator[JsonDict]: ...

    def estimate_tokens(self, text: str) -> int: ...

    def estimate_prompt_tokens(self, *, user_text: str) -> int: ...

    def usage_ratio_text(self) -> str: ...

    def to_record(self) -> JsonDict: ...

    def load_record(self, rec: JsonDict) -> None: ...

    def shutdown(self) -> None: ...


@dataclass(frozen=True)
class LegacyRunner:
    session: Any

    @property
    def cfg(self) -> Any:
        return getattr(self.session, "cfg", None)

    def send_stream(self, user_text: str) -> Iterator[JsonDict]:
        return self.session.send_stream(user_text)

    def estimate_tokens(self, text: str) -> int:
        return int(self.session.estimate_tokens(text))

    def estimate_prompt_tokens(self, *, user_text: str) -> int:
        return int(self.session.estimate_prompt_tokens(user_text=user_text))

    def usage_ratio_text(self) -> str:
        fn = getattr(self.session, "usage_ratio_text", None)
        if callable(fn):
            return str(fn())
        # Best-effort fallback.
        try:
            used = 0
            lu = getattr(self.session, "last_usage", None)
            if callable(lu):
                d = lu()
                if isinstance(d, dict):
                    used = int(d.get("total_tokens", 0) or d.get("input_tokens", 0) or 0)
            max_ctx = int(getattr(self.cfg, "context_window_tokens", 0) or 0)
            if max_ctx > 0:
                pct = (used / max_ctx) * 100.0
                return f"{used:,}/{max_ctx:,} tok ({pct:.1f}%)"
            return f"{used} tok"
        except Exception:
            return ""

    def to_record(self) -> JsonDict:
        return dict(self.session.to_record())

    def load_record(self, rec: JsonDict) -> None:
        self.session.load_record(rec)

    def shutdown(self) -> None:
        try:
            self.session.shutdown()
        except Exception:
            pass

    def restart_playwright(self) -> None:
        fn = getattr(self.session, "restart_playwright", None)
        if callable(fn):
            fn()


@dataclass(frozen=True)
class AgentsSdkRunner(LegacyRunner):
    """Same wrapper; named for clarity in logs/tests."""


def make_runner(session: Any) -> ChatEngine:
    # Avoid importing AgentsSdkSession here to prevent circular imports.
    mod = type(session).__module__
    name = type(session).__name__
    if mod.endswith("agent_sdk.session") and name == "AgentsSdkSession":
        return AgentsSdkRunner(session=session)
    return LegacyRunner(session=session)
