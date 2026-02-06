"""desktop_agent.chat_session

Stateful chat session over the OpenAI Responses API with tool calling.

This is separate from the planner UI in `desktop_agent.ui` and is intended for
interactive chat + tool use (including dynamic tool creation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from .config import DEFAULT_MODEL
from .tools import (
    ToolRegistry,
    create_and_register_python_tool_spec,
    make_create_and_register_python_tool_handler,
    make_read_file_handler,
    make_set_system_prompt_handler,
    self_tool_creator_handler,
    self_tool_creator_tool_spec,
    read_file_tool_spec,
    run_responses_with_function_tools,
    set_system_prompt_tool_spec,
)


JsonDict = dict[str, Any]


def _as_dict(x: Any) -> JsonDict:
    if isinstance(x, dict):
        return x
    if hasattr(x, "model_dump"):
        try:
            d = x.model_dump()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    if hasattr(x, "to_dict"):
        try:
            d = x.to_dict()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    # Best-effort fallback for SDK objects
    try:
        return json.loads(json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {"_repr": repr(x)}


def _input_text(role: str, text: str) -> JsonDict:
    return {"role": role, "content": [{"type": "input_text", "text": text}]}


@dataclass
class ChatConfig:
    model: str = DEFAULT_MODEL
    enable_web_search: bool = True
    web_search_context_size: str = "medium"  # low|medium|high
    tool_base_dir: Path = field(default_factory=lambda: Path.cwd())
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    allow_model_set_system_prompt: bool = True
    allow_model_create_tools: bool = True
    allow_model_propose_tools: bool = True
    allow_read_file: bool = True
    hide_think: bool = False


@dataclass
class ChatDelta:
    """New items appended to the conversation for a single user turn."""

    new_items: list[JsonDict]
    assistant_text: str


class ChatSession:
    def __init__(self, *, api_key: Optional[str] = None, config: Optional[ChatConfig] = None) -> None:
        self.cfg = config or ChatConfig()
        self._client = OpenAI(api_key=api_key)

        self._system_prompt: str = "You are a helpful assistant."
        self._conversation: list[JsonDict] = []
        self._reset_after_turn: bool = False

        self.registry = ToolRegistry(tools={}, handlers={})
        self._install_default_tools()

        self.reset(keep_system_prompt=True)

    # ---- system prompt / history ----

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def set_system_prompt(self, text: str) -> None:
        self._system_prompt = str(text)

    def reset(self, *, keep_system_prompt: bool = True) -> None:
        if not keep_system_prompt:
            self._system_prompt = "You are a helpful assistant."
        # Conversation excludes the system prompt; we pass it via `instructions=` so it can
        # be changed without rewriting the history.
        self._conversation = []

    # ---- tools ----

    def _install_default_tools(self) -> None:
        # read_file (repo-scoped)
        read_handler = make_read_file_handler(base_dir=self.cfg.tool_base_dir)

        def gated_read(args: JsonDict) -> str:
            if not self.cfg.allow_read_file:
                return json.dumps({"ok": False, "error": "read_file disabled"})
            return read_handler(args)

        self.registry.add(tool_spec=read_file_tool_spec(), handler=gated_read)

        # system prompt update tool (mutates this session)
        sys_handler = make_set_system_prompt_handler(
            set_prompt=self.set_system_prompt,
            reset_history=lambda do_reset: setattr(self, "_reset_after_turn", self._reset_after_turn or bool(do_reset)),
        )

        def gated_system(args: JsonDict) -> str:
            if not self.cfg.allow_model_set_system_prompt:
                return json.dumps({"ok": False, "error": "set_system_prompt disabled"})
            return sys_handler(args)

        self.registry.add(tool_spec=set_system_prompt_tool_spec(), handler=gated_system)

        # proposal-only tool creator (safe by default)
        def gated_propose(args: JsonDict) -> str:
            if not self.cfg.allow_model_propose_tools:
                return json.dumps({"ok": False, "error": "propose_function_tool disabled"})
            return self_tool_creator_handler(args)

        self.registry.add(tool_spec=self_tool_creator_tool_spec(), handler=gated_propose)

        # dynamic tool creation + registration
        create_handler = make_create_and_register_python_tool_handler(registry=self.registry)

        def gated_create(args: JsonDict) -> str:
            if not self.cfg.allow_model_create_tools:
                return json.dumps({"ok": False, "error": "create_and_register_python_tool disabled"})
            return create_handler(args)

        self.registry.add(tool_spec=create_and_register_python_tool_spec(), handler=gated_create)

    def _extra_tools(self) -> list[JsonDict]:
        if not self.cfg.enable_web_search:
            return []
        return [{"type": "web_search", "search_context_size": self.cfg.web_search_context_size}]

    # ---- send ----

    def send(self, user_text: str) -> ChatDelta:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be non-empty")

        start_len = len(self._conversation)
        self._conversation.append(_input_text("user", user_text.strip()))

        create_kwargs: dict[str, Any] = {}
        if self.cfg.temperature is not None:
            create_kwargs["temperature"] = float(self.cfg.temperature)
        if self.cfg.top_p is not None:
            create_kwargs["top_p"] = float(self.cfg.top_p)
        if self.cfg.max_output_tokens is not None:
            create_kwargs["max_output_tokens"] = int(self.cfg.max_output_tokens)
        if self.cfg.max_tool_calls is not None:
            create_kwargs["max_tool_calls"] = int(self.cfg.max_tool_calls)

        resp, updated = run_responses_with_function_tools(
            client=self._client,
            model=self.cfg.model,
            input_items=self._conversation,
            registry=self.registry,
            extra_tools=self._extra_tools(),
            instructions_provider=lambda: self._system_prompt,
            max_rounds=12,
            return_input_items=True,
            **create_kwargs,
        )

        assistant_text = getattr(resp, "output_text", "") or ""
        updated_dicts = [_as_dict(x) for x in updated]
        new_items = updated_dicts[start_len:]

        # Persist updated conversation for next turn.
        if self._reset_after_turn:
            self._conversation = []
            self._reset_after_turn = False
        else:
            self._conversation = updated_dicts
        return ChatDelta(new_items=new_items, assistant_text=str(assistant_text))
