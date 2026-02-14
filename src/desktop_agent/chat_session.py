"""desktop_agent.chat_session

Stateful chat session over the OpenAI Responses API with tool calling.

This is separate from the planner UI in `desktop_agent.ui` and is intended for
interactive chat + tool use (including dynamic tool creation and submodels).
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import os
import queue
import secrets
import subprocess
import sys
import threading
import time
import logging
from multiprocessing.connection import Listener

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError

from .config import DEFAULT_MODEL
from .chat_store import new_chat_id
from .tools import (
    ToolRegistry,
    add_to_system_prompt_tool_spec,
    append_file_tool_spec,
    create_and_register_python_tool_spec,
    get_system_prompt_tool_spec,
    make_append_file_handler,
    make_add_to_system_prompt_handler,
    make_create_and_register_python_tool_handler,
    make_get_system_prompt_handler,
    make_python_sandbox_handler,
    make_render_plot_handler,
    make_create_and_register_analysis_tool_handler,
    make_read_file_handler,
    make_set_system_prompt_handler,
    self_tool_creator_handler,
    self_tool_creator_tool_spec,
    python_sandbox_tool_spec,
    render_plot_tool_spec,
    create_and_register_analysis_tool_spec,
    read_file_tool_spec,
    run_responses_with_function_tools,
    set_system_prompt_tool_spec,
    make_write_file_handler,
    write_file_tool_spec,
    submodel_batch_tool_spec,
    submodel_close_tool_spec,
    submodel_list_tool_spec,
    run_responses_with_function_tools_stream,
    register_saved_analysis_tools,
)
from .model_caps import ModelCapsStore


JsonDict = dict[str, Any]


_SUBLOG = logging.getLogger("desktop_agent.submodels")
_LOG = logging.getLogger("desktop_agent.chat_session")


def _ensure_submodel_logging(*, base_dir: Path) -> None:
    """Log submodel activity to terminal + chat_history/chat_ui.log (append)."""

    if getattr(_SUBLOG, "_configured", False):
        return
    try:
        log_path = (base_dir / "chat_history" / "chat_ui.log").resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _SUBLOG.setLevel(logging.INFO)
        _SUBLOG.propagate = False
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        sh = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        _SUBLOG.handlers.clear()
        _SUBLOG.addHandler(fh)
        _SUBLOG.addHandler(sh)
        setattr(_SUBLOG, "_configured", True)
    except Exception:
        setattr(_SUBLOG, "_configured", True)


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


def _strip_status_keys(items: list[JsonDict]) -> list[JsonDict]:
    out: list[JsonDict] = []
    for it in items:
        if isinstance(it, dict) and "status" in it:
            d = dict(it)
            d.pop("status", None)
            out.append(d)
        else:
            out.append(it)
    return out


def _strip_reasoning_items(items: list[JsonDict]) -> list[JsonDict]:
    """Drop output-only reasoning items from stored conversation history.

    Some models emit `reasoning` items in the returned `input` list. Replaying those
    items back into the next `responses.create(input=...)` call can cause 400s.
    We still surface them to the UI via `new_items`, but we don't persist them.
    """

    out: list[JsonDict] = []
    for it in items:
        if isinstance(it, dict) and it.get("type") in {"reasoning", "reasoning_summary"}:
            continue
        out.append(it)
    return out


def _strip_hash_commands_text(text: str) -> str:
    """Remove `#commands` (no space) from user/assistant text.

    Keeps markdown headings like `# Heading` (hash followed by a space).
    """

    t = str(text or "")
    if "#" not in t:
        return t
    # Remove tokens like #skip / #foo_bar; keep markdown headings "# heading".
    t = re.sub(r"#(\S+)", "", t)
    # Collapse extra whitespace created by removals.
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _strip_hash_commands_in_messages(items: list[JsonDict]) -> list[JsonDict]:
    """Remove `#commands` from message content items (input_text/output_text)."""

    out: list[JsonDict] = []
    for it in items:
        if not isinstance(it, dict):
            out.append(it)
            continue
        role = it.get("role")
        if role not in {"user", "assistant", "system"}:
            out.append(it)
            continue
        content = it.get("content")
        if not isinstance(content, list) or not content:
            out.append(it)
            continue
        changed = False
        new_content: list[Any] = []
        for c in content:
            if not isinstance(c, dict):
                new_content.append(c)
                continue
            ctype = c.get("type")
            if ctype in {"input_text", "output_text"} and isinstance(c.get("text"), str):
                nt = _strip_hash_commands_text(str(c.get("text") or ""))
                if nt != c.get("text"):
                    cc = dict(c)
                    cc["text"] = nt
                    new_content.append(cc)
                    changed = True
                else:
                    new_content.append(c)
            else:
                new_content.append(c)
        if changed:
            ni = dict(it)
            ni["content"] = new_content
            out.append(ni)
        else:
            out.append(it)
    return out


def _extract_usage(resp: Any) -> dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def get_int(obj: Any, key: str) -> int:
        if isinstance(obj, dict):
            v = obj.get(key)
        else:
            v = getattr(obj, key, None)
        try:
            return int(v)
        except Exception:
            return 0

    input_tokens = get_int(usage, "input_tokens")
    output_tokens = get_int(usage, "output_tokens")
    total_tokens = get_int(usage, "total_tokens")
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def _input_text(role: str, text: str) -> JsonDict:
    return {"role": role, "content": [{"type": "input_text", "text": text}]}


@dataclass
class ChatConfig:
    model: str = DEFAULT_MODEL
    enable_web_search: bool = True
    web_search_context_size: str = "medium"  # low|medium|high
    enable_file_search: bool = False
    file_search_vector_store_ids: list[str] = field(default_factory=list)
    file_search_max_num_results: int | None = None
    include_file_search_results: bool = False
    tool_base_dir: Path = field(default_factory=lambda: Path.cwd())
    context_window_tokens: int = 128_000
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    allow_model_set_system_prompt: bool = True
    allow_model_create_tools: bool = True
    allow_model_propose_tools: bool = True
    allow_read_file: bool = True
    allow_write_files: bool = False
    hide_think: bool = False
    allow_python_sandbox: bool = True
    python_sandbox_timeout_s: float = 12.0
    allow_model_create_analysis_tools: bool = False
    allow_submodels: bool = True
    max_submodels: int = 6
    max_submodel_depth: int = 1
    submodel_depth: int = 0
    submodel_ping_s: float = 2.0


@dataclass
class ChatDelta:
    """New items appended to the conversation for a single user turn."""

    new_items: list[JsonDict]
    assistant_text: str


class ChatSession:
    def __init__(self, *, api_key: Optional[str] = None, config: Optional[ChatConfig] = None) -> None:
        self.cfg = config or ChatConfig()
        _ensure_submodel_logging(base_dir=self.cfg.tool_base_dir)
        self._api_key = api_key
        self._client = OpenAI(api_key=api_key)

        self._system_prompt: str = "You are a helpful assistant."
        self._conversation: list[JsonDict] = []
        self._reset_after_turn: bool = False
        self.chat_id: str = new_chat_id()
        self.title: str = "New chat"
        self._submodels: dict[str, "_SubprocessSubmodel"] = {}
        self._submodel_listener: Listener | None = None
        self._submodel_accept_thread: threading.Thread | None = None
        self._submodel_ping_thread: threading.Thread | None = None
        self._submodel_stop = threading.Event()

        self.registry = ToolRegistry(tools={}, handlers={})
        self._install_default_tools()

        # Per-model compatibility cache (avoid repeating 400s for unsupported params).
        caps_path = (self.cfg.tool_base_dir / "chat_history" / "model_caps.json").resolve()
        self._model_caps = ModelCapsStore(path=caps_path)

        self._start_submodel_threads()
        self.reset(keep_system_prompt=True)

        self._last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # ---- system prompt / history ----

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def set_system_prompt(self, text: str) -> None:
        self._system_prompt = str(text)

    def reset(self, *, keep_system_prompt: bool = True) -> None:
        if not keep_system_prompt:
            self._system_prompt = "You are a helpful assistant."
        # Conversation excludes the system prompt; we pass it via `instructions=` so it can
        # be changed without rewriting the history.
        self._conversation = []
        self.title = "New chat"
        self._submodels = {}
        self._stop_submodel_threads()
        self._start_submodel_threads()

    # ---- tools ----

    def _install_default_tools(self) -> None:
        # read_file (repo-scoped)
        read_handler = make_read_file_handler(base_dir=self.cfg.tool_base_dir)

        def gated_read(args: JsonDict) -> str:
            if not self.cfg.allow_read_file:
                return json.dumps({"ok": False, "error": "read_file disabled"})
            return read_handler(args)

        self.registry.add(tool_spec=read_file_tool_spec(), handler=gated_read)

        # write/append file tools (repo-scoped + allowlist)
        write_handler = make_write_file_handler(base_dir=self.cfg.tool_base_dir)
        append_handler = make_append_file_handler(base_dir=self.cfg.tool_base_dir)

        def gated_write(args: JsonDict) -> str:
            if not self.cfg.allow_write_files:
                return json.dumps({"ok": False, "error": "write_file disabled"})
            return write_handler(args)

        def gated_append(args: JsonDict) -> str:
            if not self.cfg.allow_write_files:
                return json.dumps({"ok": False, "error": "append_file disabled"})
            return append_handler(args)

        self.registry.add(tool_spec=write_file_tool_spec(), handler=gated_write)
        self.registry.add(tool_spec=append_file_tool_spec(), handler=gated_append)

        # python sandbox tool (restricted analysis runner)
        py_handler = make_python_sandbox_handler(base_dir=self.cfg.tool_base_dir, timeout_s=float(self.cfg.python_sandbox_timeout_s))

        def gated_py(args: JsonDict) -> str:
            if not self.cfg.allow_python_sandbox:
                return json.dumps({"ok": False, "error": "python_sandbox disabled"})
            return py_handler(args)

        self.registry.add(tool_spec=python_sandbox_tool_spec(), handler=gated_py)

        # Visualization helper (alias around python_sandbox for plots/graphs)
        render_plot = make_render_plot_handler(python_sandbox_handler=gated_py)
        self.registry.add(tool_spec=render_plot_tool_spec(), handler=render_plot)

        # Reusable analysis tools (scripts that run via python_sandbox and can be registered on the fly)
        scripts_root = Path(__file__).resolve().parents[2] / "ui" / "automated_calibration" / "analysis_tools"
        create_analysis = make_create_and_register_analysis_tool_handler(
            registry=self.registry, python_sandbox_handler=gated_py, scripts_root=scripts_root
        )

        def gated_create_analysis(args: JsonDict) -> str:
            if not self.cfg.allow_model_create_analysis_tools:
                return json.dumps({"ok": False, "error": "create_and_register_analysis_tool disabled"})
            if not self.cfg.allow_python_sandbox:
                return json.dumps({"ok": False, "error": "python_sandbox disabled"})
            return create_analysis(args)

        self.registry.add(tool_spec=create_and_register_analysis_tool_spec(), handler=gated_create_analysis)

        # Load any previously created analysis tools from disk so the model can reuse them.
        try:
            loaded = register_saved_analysis_tools(registry=self.registry, python_sandbox_handler=gated_py, scripts_root=scripts_root)
            if loaded:
                _SUBLOG.info("analysis_tools_loaded count=%d names=%s", len(loaded), ",".join(loaded[:50]))
        except Exception as e:  # noqa: BLE001
            _SUBLOG.info("analysis_tools_load_failed error=%s", str(e))

        # submodels (parallelizable helper sessions)
        self.registry.add(tool_spec=submodel_list_tool_spec(), handler=self._tool_submodel_list)
        self.registry.add(tool_spec=submodel_close_tool_spec(), handler=self._tool_submodel_close)
        self.registry.add(tool_spec=submodel_batch_tool_spec(), handler=self._tool_submodel_batch)

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

        get_sys = make_get_system_prompt_handler(get_prompt=self.get_system_prompt)

        def gated_get_sys(args: JsonDict) -> str:
            if not self.cfg.allow_model_set_system_prompt:
                return json.dumps({"ok": False, "error": "get_system_prompt disabled"})
            return get_sys(args)

        self.registry.add(tool_spec=get_system_prompt_tool_spec(), handler=gated_get_sys)

        add_sys = make_add_to_system_prompt_handler(
            get_prompt=self.get_system_prompt,
            set_prompt=self.set_system_prompt,
            reset_history=lambda do_reset: setattr(self, "_reset_after_turn", self._reset_after_turn or bool(do_reset)),
        )

        def gated_add_sys(args: JsonDict) -> str:
            if not self.cfg.allow_model_set_system_prompt:
                return json.dumps({"ok": False, "error": "add_to_system_prompt disabled"})
            return add_sys(args)

        self.registry.add(tool_spec=add_to_system_prompt_tool_spec(), handler=gated_add_sys)

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
        out: list[JsonDict] = []

        if self.cfg.enable_web_search:
            out.append({"type": "web_search", "search_context_size": self.cfg.web_search_context_size})

        if self.cfg.enable_file_search:
            ids = [str(x).strip() for x in (self.cfg.file_search_vector_store_ids or []) if str(x).strip()]
            if ids:
                fs: JsonDict = {"type": "file_search", "vector_store_ids": ids}
                if self.cfg.file_search_max_num_results is not None:
                    fs["max_num_results"] = int(self.cfg.file_search_max_num_results)
                out.append(fs)

        return out

    # ---- send ----

    def send(self, user_text: str) -> ChatDelta:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be non-empty")

        # Ensure history is replay-safe (no output-only items/keys/commands).
        self._conversation = _strip_hash_commands_in_messages(_strip_reasoning_items(_strip_status_keys(list(self._conversation))))

        start_len = len(self._conversation)
        self._conversation.append(_input_text("user", _strip_hash_commands_text(user_text.strip())))

        create_kwargs: dict[str, Any] = {}
        if self.cfg.temperature is not None:
            create_kwargs["temperature"] = float(self.cfg.temperature)
        if self.cfg.top_p is not None:
            create_kwargs["top_p"] = float(self.cfg.top_p)
        if self.cfg.max_output_tokens is not None:
            create_kwargs["max_output_tokens"] = int(self.cfg.max_output_tokens)
        if self.cfg.max_tool_calls is not None:
            mtc = int(self.cfg.max_tool_calls)
            if mtc >= 1:
                create_kwargs["max_tool_calls"] = mtc
        if self.cfg.enable_file_search and self.cfg.include_file_search_results:
            create_kwargs["include"] = ["file_search_call.results"]
        create_kwargs = self._model_caps.filter_create_kwargs(model=self.cfg.model, create_kwargs=create_kwargs)

        resp, updated = run_responses_with_function_tools(
            client=self._client,
            model=self.cfg.model,
            input_items=self._conversation,
            registry=self.registry,
            extra_tools=self._extra_tools(),
            instructions_provider=lambda: self._system_prompt,
            max_rounds=12,
            return_input_items=True,
            on_unsupported_param=lambda p: self._model_caps.mark_unsupported(model=self.cfg.model, param=p),
            **create_kwargs,
        )

        assistant_text = getattr(resp, "output_text", "") or ""
        self._last_usage = _extract_usage(resp)
        updated_dicts = [_as_dict(x) for x in updated]
        updated_dicts = _strip_status_keys(updated_dicts)
        new_items = updated_dicts[start_len:]
        updated_state = _strip_hash_commands_in_messages(_strip_reasoning_items(updated_dicts))

        # Update a simple title heuristic if still default.
        if self.title == "New chat":
            self.title = (user_text.strip().splitlines()[0][:60] or "New chat").strip()

        # Persist updated conversation for next turn.
        if self._reset_after_turn:
            self._conversation = []
            self._reset_after_turn = False
        else:
            self._conversation = updated_state
        return ChatDelta(new_items=new_items, assistant_text=str(assistant_text))

    def send_stream(
        self,
        user_text: str,
        *,
        on_item: Optional[callable] = None,
    ) -> "Iterator[dict[str, Any]]":
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be non-empty")

        # Ensure history is replay-safe (no output-only items/keys/commands).
        self._conversation = _strip_hash_commands_in_messages(_strip_reasoning_items(_strip_status_keys(list(self._conversation))))

        start_len = len(self._conversation)
        self._conversation.append(_input_text("user", _strip_hash_commands_text(user_text.strip())))

        create_kwargs: dict[str, Any] = {}
        if self.cfg.temperature is not None:
            create_kwargs["temperature"] = float(self.cfg.temperature)
        if self.cfg.top_p is not None:
            create_kwargs["top_p"] = float(self.cfg.top_p)
        if self.cfg.max_output_tokens is not None:
            create_kwargs["max_output_tokens"] = int(self.cfg.max_output_tokens)
        if self.cfg.max_tool_calls is not None:
            mtc = int(self.cfg.max_tool_calls)
            if mtc >= 1:
                create_kwargs["max_tool_calls"] = mtc
        if self.cfg.enable_file_search and self.cfg.include_file_search_results:
            create_kwargs["include"] = ["file_search_call.results"]
        create_kwargs = self._model_caps.filter_create_kwargs(model=self.cfg.model, create_kwargs=create_kwargs)

        final_resp_dict: dict[str, Any] | None = None
        updated_input: list[JsonDict] | None = None

        # Because `on_text_delta` is a callback, we collect events in a local list
        # and drain them as the stream progresses.
        events: list[dict[str, Any]] = []
        assistant_so_far: list[str] = []

        def _on_text_delta(d: str) -> None:
            assistant_so_far.append(d)
            events.append({"type": "assistant_delta", "delta": d})

        stream_iter = run_responses_with_function_tools_stream(
            client=self._client,
            model=self.cfg.model,
            input_items=self._conversation,
            registry=self.registry,
            extra_tools=self._extra_tools(),
            instructions_provider=lambda: self._system_prompt,
            max_rounds=12,
            return_input_items=True,
            on_item=on_item,
            on_text_delta=_on_text_delta,
            on_unsupported_param=lambda p: self._model_caps.mark_unsupported(model=self.cfg.model, param=p),
            **create_kwargs,
        )

        stream_error: Exception | None = None
        try:
            for ev in stream_iter:
                # Drain any buffered deltas first so UI updates feel immediate.
                while events:
                    yield events.pop(0)

                et = ev.get("type")
                if et == "done":
                    respd = ev.get("response")
                    final_resp_dict = respd if isinstance(respd, dict) else None
                elif et == "input_items":
                    inp = ev.get("input")
                    updated_input = inp if isinstance(inp, list) else None
                # Hide internal stream events from the UI; we only surface assistant deltas
                # (via the buffered callback) and the final `turn_done`.
        except Exception as e:  # noqa: BLE001
            stream_error = e

        while events:
            yield events.pop(0)

        if stream_error is not None:
            # Network hiccups (or server-side disconnects) can kill a long-running stream.
            # Treat this as a soft failure: preserve any partial assistant text in history
            # and surface an error event so UIs can show a useful message instead of
            # crashing the worker thread.
            partial = "".join(assistant_so_far).strip()

            # Detect common transient classes (best-effort; don't require httpx at import time).
            is_transient = isinstance(stream_error, (APIConnectionError, APITimeoutError))
            try:
                import httpx  # type: ignore

                is_transient = is_transient or isinstance(
                    stream_error,
                    (
                        httpx.RemoteProtocolError,
                        httpx.ReadError,
                        httpx.ConnectError,
                        httpx.WriteError,
                        httpx.TimeoutException,
                    ),
                )
            except Exception:
                pass

            msg = f"{type(stream_error).__name__}: {stream_error}"
            if is_transient:
                msg = f"(stream disconnected) {msg}"

            if partial:
                # Keep the partial answer, but annotate that it was interrupted.
                self._conversation.append(
                    _input_text(
                        "assistant",
                        f"{partial}\n\n[stream interrupted: {msg}]",
                    )
                )
            else:
                self._conversation.append(_input_text("assistant", f"[stream error: {msg}]"))

            updated_input = list(self._conversation)
            updated_dicts = [_as_dict(x) for x in updated_input]
            updated_dicts = _strip_status_keys(updated_dicts)
            new_items = updated_dicts[start_len:]
            updated_state = _strip_hash_commands_in_messages(_strip_reasoning_items(updated_dicts))
            self._conversation = updated_state

            _LOG.warning("Streaming turn failed; preserved partial assistant text. error=%s", msg)
            yield {"type": "error", "error": msg, "transient": bool(is_transient)}
            yield {"type": "turn_done", "new_items": new_items}
            return

        if final_resp_dict is not None:
            self._last_usage = _extract_usage(final_resp_dict)
        if updated_input is None:
            updated_input = self._conversation

        updated_dicts = [_as_dict(x) for x in updated_input]
        updated_dicts = _strip_status_keys(updated_dicts)
        new_items = updated_dicts[start_len:]
        updated_state = _strip_hash_commands_in_messages(_strip_reasoning_items(updated_dicts))

        # Update title heuristic if still default.
        if self.title == "New chat":
            self.title = (user_text.strip().splitlines()[0][:60] or "New chat").strip()

        if self._reset_after_turn:
            self._conversation = []
            self._reset_after_turn = False
        else:
            self._conversation = updated_state

        yield {"type": "turn_done", "new_items": new_items}

    def last_usage(self) -> dict[str, int]:
        return dict(self._last_usage)

    def usage_ratio_text(self) -> str:
        used = int(self._last_usage.get("total_tokens", 0))
        if used <= 0:
            used = int(self._last_usage.get("input_tokens", 0))
        max_ctx = int(self.cfg.context_window_tokens)
        if max_ctx <= 0:
            return f"{used} tok"
        pct = (used / max_ctx) * 100.0
        return f"{used:,}/{max_ctx:,} tok ({pct:.1f}%)"

    def estimate_tokens(self, text: str) -> int:
        """Best-effort token estimate for UI display while streaming.

        Tries `tiktoken` if available; otherwise falls back to a rough heuristic.
        """

        s = text or ""
        try:
            import tiktoken  # type: ignore

            try:
                enc = tiktoken.encoding_for_model(self.cfg.model)
            except Exception:
                # Reasonable defaults for modern OpenAI models.
                try:
                    enc = tiktoken.get_encoding("o200k_base")
                except Exception:
                    enc = tiktoken.get_encoding("cl100k_base")
            return int(len(enc.encode(s)))
        except Exception:
            # Heuristic: ~4 chars/token for English-ish text.
            return int(max(0, (len(s) + 3) // 4))

    def estimate_prompt_tokens(self, *, user_text: str) -> int:
        """Estimate prompt tokens (system + visible chat + new user msg)."""

        parts: list[str] = []
        if self._system_prompt:
            parts.append(self._system_prompt)

        for msg in self._conversation:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in {"user", "assistant", "system"}:
                continue
            content = msg.get("content")
            if not isinstance(content, list) or not content:
                continue
            t = content[0].get("text") if isinstance(content[0], dict) else None
            if isinstance(t, str) and t:
                parts.append(t)

        if user_text:
            parts.append(user_text)

        return self.estimate_tokens("\n".join(parts))

    # ---- persistence helpers ----

    def to_record(self) -> JsonDict:
        return {
            "chat_id": self.chat_id,
            "title": self.title,
            "model": self.cfg.model,
            "system_prompt": self._system_prompt,
            "enable_web_search": bool(self.cfg.enable_web_search),
            "web_search_context_size": str(self.cfg.web_search_context_size),
            "enable_file_search": bool(self.cfg.enable_file_search),
            "file_search_vector_store_ids": list(self.cfg.file_search_vector_store_ids),
            "file_search_max_num_results": self.cfg.file_search_max_num_results,
            "include_file_search_results": bool(self.cfg.include_file_search_results),
            "allow_python_sandbox": bool(self.cfg.allow_python_sandbox),
            "python_sandbox_timeout_s": float(self.cfg.python_sandbox_timeout_s),
            "conversation": list(_strip_status_keys(_strip_hash_commands_in_messages(_strip_reasoning_items(list(self._conversation))))),
            "submodels": {sid: sm.to_record() for sid, sm in self._submodels.items()},
            "submodel_depth": int(self.cfg.submodel_depth),
        }

    def load_record(self, rec: JsonDict) -> None:
        self.chat_id = str(rec.get("chat_id") or new_chat_id())
        self.title = str(rec.get("title") or "New chat")
        self.cfg.model = str(rec.get("model") or self.cfg.model)
        self.cfg.enable_web_search = bool(rec.get("enable_web_search", self.cfg.enable_web_search))
        self.cfg.web_search_context_size = str(rec.get("web_search_context_size") or self.cfg.web_search_context_size)
        self.cfg.enable_file_search = bool(rec.get("enable_file_search", self.cfg.enable_file_search))
        vs = rec.get("file_search_vector_store_ids")
        if isinstance(vs, list):
            self.cfg.file_search_vector_store_ids = [str(x) for x in vs if str(x).strip()]
        mn = rec.get("file_search_max_num_results")
        try:
            self.cfg.file_search_max_num_results = int(mn) if mn is not None else None
        except Exception:
            self.cfg.file_search_max_num_results = None
        self.cfg.include_file_search_results = bool(rec.get("include_file_search_results", self.cfg.include_file_search_results))
        self.cfg.allow_python_sandbox = bool(rec.get("allow_python_sandbox", self.cfg.allow_python_sandbox))
        try:
            self.cfg.python_sandbox_timeout_s = float(rec.get("python_sandbox_timeout_s", self.cfg.python_sandbox_timeout_s))
        except Exception:
            pass
        self._system_prompt = str(rec.get("system_prompt") or self._system_prompt)
        conv = rec.get("conversation")
        self._conversation = (
            _strip_hash_commands_in_messages(_strip_reasoning_items(_strip_status_keys(conv))) if isinstance(conv, list) else []
        )

        self._submodels = {}
        subs = rec.get("submodels")
        if isinstance(subs, dict):
            for sid, srec in subs.items():
                if not isinstance(srec, dict):
                    continue
                if int(self.cfg.submodel_depth) >= int(self.cfg.max_submodel_depth):
                    continue
                if len(self._submodels) >= int(self.cfg.max_submodels):
                    break
                self._submodels[str(sid)] = _SubprocessSubmodel.from_record(srec)

    # ---- submodels API (used by UI + tools) ----

    def list_submodels(self) -> list[JsonDict]:
        out: list[JsonDict] = []
        for sid, sm in self._submodels.items():
            out.append(
                {
                    "id": sid,
                    "title": sm.title,
                    "model": sm.model,
                    "depth": int(self.cfg.submodel_depth) + 1,
                    "state": sm.state,
                }
            )
        return out

    def get_submodel(self, submodel_id: str) -> Optional["_SubprocessSubmodel"]:
        return self._submodels.get(str(submodel_id))

    def close_submodel(self, submodel_id: str) -> bool:
        sid = str(submodel_id)
        existed = sid in self._submodels
        sm = self._submodels.pop(sid, None)
        if sm is not None:
            _SUBLOG.info("submodel_close submodel_id=%s title=%s model=%s", sm.submodel_id, sm.title, sm.model)
            sm.stop()
        return existed

    def _can_spawn_submodel(self) -> bool:
        if not self.cfg.allow_submodels:
            return False
        if int(self.cfg.submodel_depth) >= int(self.cfg.max_submodel_depth):
            return False
        return len(self._submodels) < int(self.cfg.max_submodels)

    def _start_submodel_threads(self) -> None:
        if self._submodel_listener is not None:
            return
        auth = secrets.token_bytes(16)
        self._submodel_listener = Listener(("127.0.0.1", 0), authkey=auth)
        self._submodel_auth_hex = auth.hex()
        self._submodel_stop.clear()
        try:
            host, port = self._submodel_listener.address
            _SUBLOG.info("submodel_listener_start host=%s port=%s", str(host), str(port))
        except Exception:
            pass

        def accept_loop() -> None:
            assert self._submodel_listener is not None
            while not self._submodel_stop.is_set():
                try:
                    conn = self._submodel_listener.accept()
                except Exception:
                    break
                try:
                    hello = conn.recv()
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    continue
                if isinstance(hello, dict) and hello.get("type") == "hello":
                    pass
                # Connection will be assigned by the spawn call via a pending queue.
                try:
                    pending = getattr(self, "_pending_submodel_conns", None)
                    if isinstance(pending, queue.Queue):
                        pending.put(conn, timeout=1)
                    else:
                        conn.close()
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass

        self._pending_submodel_conns: queue.Queue = queue.Queue()
        self._submodel_accept_thread = threading.Thread(target=accept_loop, daemon=True)
        self._submodel_accept_thread.start()

        def ping_loop() -> None:
            while not self._submodel_stop.is_set():
                try:
                    interval = float(self.cfg.submodel_ping_s)
                except Exception:
                    interval = 1.0
                interval = max(0.2, min(10.0, interval))
                for sm in list(self._submodels.values()):
                    try:
                        sm.ping()
                    except Exception:
                        _SUBLOG.info("submodel_ping_failed submodel_id=%s title=%s model=%s state=%s", sm.submodel_id, sm.title, sm.model, sm.state)
                        pass
                time.sleep(interval)

        self._submodel_ping_thread = threading.Thread(target=ping_loop, daemon=True)
        self._submodel_ping_thread.start()

    def _stop_submodel_threads(self) -> None:
        self._submodel_stop.set()
        for sm in list(self._submodels.values()):
            try:
                sm.stop()
            except Exception:
                pass
        self._submodels = {}
        try:
            if self._submodel_listener is not None:
                self._submodel_listener.close()
        except Exception:
            pass
        self._submodel_listener = None
        _SUBLOG.info("submodel_listener_stop")

    def shutdown(self) -> None:
        """Terminate submodel workers and stop background threads."""

        self._stop_submodel_threads()

    def _spawn_submodel(self, *, title: str, system_prompt: str, model: str, allow_nested: bool) -> "_SubprocessSubmodel":
        if not self._can_spawn_submodel():
            raise ValueError("submodel spawning disabled or limit reached")
        self._start_submodel_threads()
        assert self._submodel_listener is not None
        host, port = self._submodel_listener.address
        sub_id = new_chat_id()

        env = os.environ.copy()
        # Ensure the worker can import `desktop_agent` when launched from repo root.
        src_path = str((self.cfg.tool_base_dir / "src").resolve())
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [
            sys.executable,
            "-m",
            "desktop_agent.submodel_worker",
            "--connect-host",
            str(host),
            "--connect-port",
            str(port),
            "--authkey",
            str(getattr(self, "_submodel_auth_hex", "")),
            "--model",
            str(model or self.cfg.model),
            "--system-prompt",
            str(system_prompt or "You are a helpful assistant."),
            "--base-dir",
            str(self.cfg.tool_base_dir),
        ]
        if self.cfg.enable_web_search:
            cmd += ["--enable-web-search", "--web-search-context-size", str(self.cfg.web_search_context_size)]
        if self.cfg.enable_file_search:
            ids = [str(x).strip() for x in (self.cfg.file_search_vector_store_ids or []) if str(x).strip()]
            if ids:
                cmd += ["--enable-file-search", "--vector-store-ids", ",".join(ids)]
                if self.cfg.file_search_max_num_results is not None:
                    cmd += ["--file-search-max-num-results", str(int(self.cfg.file_search_max_num_results))]
                if self.cfg.include_file_search_results:
                    cmd += ["--include-file-search-results"]
        # Tool gates for worker
        if self.cfg.allow_read_file:
            cmd += ["--allow-read-file"]
        if self.cfg.allow_write_files:
            cmd += ["--allow-write-files"]
        if self.cfg.allow_model_set_system_prompt:
            cmd += ["--allow-set-system-prompt"]
        if self.cfg.allow_model_propose_tools:
            cmd += ["--allow-propose-tools"]
        if self.cfg.allow_model_create_tools:
            cmd += ["--allow-create-tools"]
        if self.cfg.allow_python_sandbox:
            cmd += ["--allow-python-sandbox", "--python-sandbox-timeout-s", str(float(self.cfg.python_sandbox_timeout_s))]

        # NOTE: nested submodel spawning inside workers is not implemented yet.
        _ = allow_nested

        proc = subprocess.Popen(cmd, env=env, cwd=str(self.cfg.tool_base_dir))
        try:
            conn = self._pending_submodel_conns.get(timeout=10)
        except Exception as e:
            proc.terminate()
            raise ValueError(f"failed to connect to submodel worker: {e}") from e

        sm = _SubprocessSubmodel(
            submodel_id=sub_id,
            title=title or "Submodel",
            model=str(model or self.cfg.model),
            system_prompt=str(system_prompt or "You are a helpful assistant."),
            process=proc,
            conn=conn,
        )
        sm.start_reader()
        self._submodels[sm.submodel_id] = sm
        _SUBLOG.info("submodel_spawned submodel_id=%s title=%s model=%s allow_nested=%s", sm.submodel_id, sm.title, sm.model, bool(allow_nested))
        return sm

    # ---- tool handlers ----

    def _tool_submodel_list(self, args: JsonDict) -> str:  # noqa: ARG001
        _SUBLOG.info("tool_submodel_list")
        return json.dumps({"ok": True, "submodels": self.list_submodels()})

    def _tool_submodel_close(self, args: JsonDict) -> str:
        sid = args.get("submodel_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError("submodel_id must be a string")
        _SUBLOG.info("tool_submodel_close submodel_id=%s", sid)
        existed = self.close_submodel(sid)
        return json.dumps({"ok": True, "submodel_id": sid, "existed": existed})

    def _tool_submodel_batch(self, args: JsonDict) -> str:
        if not self.cfg.allow_submodels:
            return json.dumps({"ok": False, "error": "submodels disabled"})
        tasks = args.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("tasks must be a list")
        parallel = bool(args.get("parallel", True))
        _SUBLOG.info("tool_submodel_batch tasks=%d parallel=%s", len(tasks), bool(parallel))

        # Build a stable list of (task_index, submodel, input, keep)
        prepared: list[tuple[int, _SubprocessSubmodel, str, bool]] = []
        created_ids: list[str] = []
        for i, t in enumerate(tasks):
            if not isinstance(t, dict):
                continue
            user_input = t.get("input")
            if not isinstance(user_input, str) or not user_input.strip():
                raise ValueError("each task.input must be a non-empty string")

            keep = bool(t.get("keep", True))
            sid = t.get("submodel_id")
            model = str(t.get("model") or "")
            allow_nested = bool(t.get("allow_nested", True))

            if isinstance(sid, str) and sid:
                sm = self._submodels.get(sid)
                if sm is None:
                    raise ValueError(f"unknown submodel_id: {sid}")
            else:
                if not self._can_spawn_submodel():
                    raise ValueError("submodel limit reached")
                title = str(t.get("title") or "Submodel")
                system_prompt = str(t.get("system_prompt") or "You are a helpful assistant.")
                sm = self._spawn_submodel(title=title, system_prompt=system_prompt, model=model, allow_nested=allow_nested)
                created_ids.append(sm.submodel_id)

            prepared.append((i, sm, user_input, keep))

        def run_one(sm: _SubprocessSubmodel, text: str) -> str:
            sm.run(text)
            return sm.wait_done(timeout_s=600)

        results: list[JsonDict] = []
        if parallel and len(prepared) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(prepared))) as ex:
                fut_map = {ex.submit(run_one, sm, inp): (idx, sm, keep) for idx, sm, inp, keep in prepared}
                for fut in as_completed(fut_map):
                    idx, sm, keep = fut_map[fut]
                    try:
                        out = fut.result()
                        results.append({"i": idx, "submodel_id": sm.submodel_id, "ok": True, "output": out})
                    except Exception as e:
                        results.append({"i": idx, "submodel_id": sm.submodel_id, "ok": False, "error": f"{type(e).__name__}: {e}"})
                    finally:
                        if not keep:
                            self.close_submodel(sm.submodel_id)
        else:
            for idx, sm, inp, keep in prepared:
                try:
                    out = run_one(sm, inp)
                    results.append({"i": idx, "submodel_id": sm.submodel_id, "ok": True, "output": out})
                except Exception as e:
                    results.append({"i": idx, "submodel_id": sm.submodel_id, "ok": False, "error": f"{type(e).__name__}: {e}"})
                finally:
                    if not keep:
                        self.close_submodel(sm.submodel_id)

        results.sort(key=lambda x: int(x.get("i", 0)))
        _SUBLOG.info("tool_submodel_batch_done created=%d results=%d", len(created_ids), len(results))
        return json.dumps({"ok": True, "created": created_ids, "results": results})


@dataclass
class _SubprocessSubmodel:
    submodel_id: str
    title: str
    model: str
    system_prompt: str
    process: subprocess.Popen | None
    conn: Any | None
    state: str = "idle"  # idle|running|error|stopped
    transcript: list[JsonDict] = field(default_factory=list)
    _done_q: "queue.Queue[str]" = field(default_factory=queue.Queue, repr=False)
    _reader: threading.Thread | None = field(default=None, repr=False)

    def to_record(self) -> JsonDict:
        return {
            "chat_id": self.submodel_id,
            "title": self.title,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "state": self.state,
            "transcript": list(self.transcript),
        }

    @staticmethod
    def from_record(rec: JsonDict) -> "_SubprocessSubmodel":
        sm = _SubprocessSubmodel(
            submodel_id=str(rec.get("chat_id") or new_chat_id()),
            title=str(rec.get("title") or "Submodel"),
            model=str(rec.get("model") or ""),
            system_prompt=str(rec.get("system_prompt") or ""),
            conn=None,
            process=None,
            state=str(rec.get("state") or "offline"),
            transcript=rec.get("transcript") if isinstance(rec.get("transcript"), list) else [],
        )
        return sm

    def start_reader(self) -> None:
        if self.conn is None or self._reader is not None:
            return

        def loop() -> None:
            while True:
                try:
                    msg = self.conn.recv()
                except Exception:
                    self.state = "stopped"
                    try:
                        _SUBLOG.info("submodel_reader_stopped submodel_id=%s title=%s model=%s", self.submodel_id, self.title, self.model)
                    except Exception:
                        pass
                    break
                if not isinstance(msg, dict):
                    continue
                mtype = msg.get("type")
                try:
                    if mtype == "assistant":
                        tprev = msg.get("text")
                        tprev = str(tprev) if isinstance(tprev, str) else ""
                        if len(tprev) > 140:
                            tprev = tprev[:140] + " …"
                        _SUBLOG.info("submodel_msg submodel_id=%s type=assistant text=%s", self.submodel_id, tprev)
                    elif mtype == "error":
                        _SUBLOG.info("submodel_msg submodel_id=%s type=error error=%s", self.submodel_id, str(msg.get("error") or ""))
                    elif mtype == "status":
                        _SUBLOG.info("submodel_msg submodel_id=%s type=status state=%s", self.submodel_id, str(msg.get("state") or ""))
                    else:
                        _SUBLOG.info("submodel_msg submodel_id=%s type=%s", self.submodel_id, str(mtype))
                except Exception:
                    pass
                if mtype == "status":
                    st = msg.get("state")
                    if isinstance(st, str):
                        self.state = st
                elif mtype == "assistant":
                    txt = msg.get("text")
                    if isinstance(txt, str):
                        self.transcript.append({"role": "assistant", "text": txt})
                elif mtype == "item":
                    self.transcript.append({"role": "tool", "item": msg.get("item")})
                elif mtype == "error":
                    self.state = "error"
                    self.transcript.append({"role": "error", "text": str(msg.get("error") or "")})
                elif mtype == "done":
                    self._done_q.put_nowait("done")
                elif mtype == "pong":
                    pass

        self._reader = threading.Thread(target=loop, daemon=True)
        self._reader.start()

    def ping(self) -> None:
        if self.conn is None:
            return
        try:
            self.conn.send({"op": "ping"})
        except Exception:
            self.state = "stopped"

    def run(self, user_input: str) -> None:
        if self.conn is None:
            raise ValueError("submodel is offline")
        self.transcript.append({"role": "main", "text": str(user_input)})
        self.conn.send({"op": "run", "input": str(user_input)})
        self.state = "running"

    def wait_done(self, *, timeout_s: float) -> str:
        try:
            self._done_q.get(timeout=float(timeout_s))
        except Exception as e:
            raise TimeoutError("submodel timed out") from e
        # Return last assistant message
        for m in reversed(self.transcript):
            if m.get("role") == "assistant" and isinstance(m.get("text"), str):
                return m["text"]
        return ""

    def stop(self) -> None:
        try:
            if self.conn is not None:
                self.conn.send({"op": "stop"})
        except Exception:
            pass
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass
        self.conn = None
        if self.process is not None:
            try:
                self.process.terminate()
            except Exception:
                pass
            self.process = None
        self.state = "stopped"
