"""desktop_agent.agent_sdk.session

Agents SDK-backed chat session with a sync generator streaming API that matches
the existing `ChatSession.send_stream()` shape used by the UI.

This is *experimental* and intentionally supports only a subset of tools.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from agents import Agent
from agents.run import Runner
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from agents.tool import FunctionTool, Tool, WebSearchTool

from ..chat_store import new_chat_id
from ..model_caps import ModelCapsStore
from ..tools import (
    ToolRegistry,
    ToolError,
    append_file_tool_spec,
    make_workspace_handlers,
    make_append_file_handler,
    make_python_sandbox_handler,
    make_read_file_handler,
    make_render_plot_handler,
    make_web_fetch_handler,
    make_write_file_handler,
    python_sandbox_tool_spec,
    read_file_tool_spec,
    render_plot_tool_spec,
    web_fetch_tool_spec,
    write_file_tool_spec,
    workspace_append_tool_spec,
    workspace_create_venv_tool_spec,
    workspace_http_server_start_tool_spec,
    workspace_http_server_stop_tool_spec,
    workspace_info_tool_spec,
    workspace_list_tool_spec,
    workspace_pip_install_tool_spec,
    workspace_read_tool_spec,
    workspace_run_python_tool_spec,
    workspace_write_tool_spec,
)
from ..tools import playwright_browser_tool_spec, make_playwright_browser_handler
from ..chat_session import ChatConfig  # reuse the shared config dataclass


JsonDict = dict[str, Any]


def _input_text(role: str, text: str) -> JsonDict:
    return {"role": role, "content": [{"type": "input_text", "text": text}]}


def _output_text(role: str, text: str) -> JsonDict:
    return {"role": role, "content": [{"type": "output_text", "text": text}]}


def _estimate_tokens_fallback(text: str) -> int:
    s = text or ""
    try:
        import tiktoken  # type: ignore

        try:
            enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return int(len(enc.encode(s)))
    except Exception:
        return int(max(0, (len(s) + 3) // 4))


@dataclass
class _TurnResult:
    assistant_text: str
    new_items: list[JsonDict]


class AgentsSdkSession:
    """A minimal Agents SDK-backed session compatible with Chat UI streaming."""

    def __init__(self, *, api_key: Optional[str], config: Optional[ChatConfig]) -> None:
        self.cfg = config or ChatConfig()
        self._api_key = api_key
        if api_key and not os.environ.get("OPENAI_API_KEY"):
            # Agents SDK uses the OpenAI client which reads env by default.
            os.environ["OPENAI_API_KEY"] = api_key

        # Compatibility cache for create-kwargs is still used by our legacy stack.
        # Here we keep it for parity / future migration (not heavily used yet).
        caps_path = (self.cfg.tool_base_dir / "chat_history" / "model_caps.json").resolve()
        self._model_caps = ModelCapsStore(path=caps_path)

        self.chat_id: str = new_chat_id()
        self.title: str = "New chat"
        self._system_prompt: str = "You are a helpful assistant."
        self._conversation: list[JsonDict] = []

        self.registry = ToolRegistry(tools={}, handlers={})
        self._install_default_tools()
        self._playwright_shutdown: Optional[callable] = None
        self._pip_approver: Callable[[str], bool] = lambda prompt: False

        self._last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # ---- system prompt / history ----

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def set_system_prompt(self, text: str) -> None:
        self._system_prompt = str(text or "")

    def reset(self, *, keep_system_prompt: bool = True) -> None:
        if not keep_system_prompt:
            self._system_prompt = "You are a helpful assistant."
        self._conversation = []
        self.title = "New chat"

    def set_pip_approver(self, fn: Callable[[str], bool]) -> None:
        self._pip_approver = fn

    # ---- token estimates ----

    def estimate_tokens(self, text: str) -> int:
        return _estimate_tokens_fallback(text or "")

    def estimate_prompt_tokens(self, *, user_text: str) -> int:
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

    def last_usage(self) -> dict[str, int]:
        return dict(self._last_usage)

    def usage_ratio_text(self) -> str:
        used = int(self._last_usage.get("total_tokens", 0))
        if used <= 0:
            used = int(self._last_usage.get("input_tokens", 0))
        max_ctx = int(getattr(self.cfg, "context_window_tokens", 0) or 0)
        if max_ctx <= 0:
            return f"{used} tok"
        pct = (used / max_ctx) * 100.0
        return f"{used:,}/{max_ctx:,} tok ({pct:.1f}%)"

    # ---- persistence ----

    def to_record(self) -> JsonDict:
        return {
            "chat_id": self.chat_id,
            "title": self.title,
            "model": self.cfg.model,
            "system_prompt": self._system_prompt,
            "enable_web_search": bool(self.cfg.enable_web_search),
            "web_search_context_size": str(self.cfg.web_search_context_size),
            "enable_web_fetch": bool(getattr(self.cfg, "enable_web_fetch", True)),
            "web_fetch_timeout_s": float(getattr(self.cfg, "web_fetch_timeout_s", 20.0)),
            "web_fetch_max_chars": int(getattr(self.cfg, "web_fetch_max_chars", 120_000)),
            "web_fetch_cache_ttl_s": float(getattr(self.cfg, "web_fetch_cache_ttl_s", 30 * 60.0)),
            "web_fetch_readability": bool(getattr(self.cfg, "web_fetch_readability", True)),
            "enable_playwright_browser": bool(getattr(self.cfg, "enable_playwright_browser", False)),
            "playwright_headless": bool(getattr(self.cfg, "playwright_headless", True)),
            "playwright_watch_mode": bool(getattr(self.cfg, "playwright_watch_mode", False)),
            "playwright_screenshot_full_page": bool(getattr(self.cfg, "playwright_screenshot_full_page", False)),
            "allow_playwright_eval": bool(getattr(self.cfg, "allow_playwright_eval", False)),
            "enable_telegram_bridge": bool(getattr(self.cfg, "enable_telegram_bridge", False)),
            "telegram_send_tool_events": bool(getattr(self.cfg, "telegram_send_tool_events", False)),
            "enable_dev_workspace": bool(getattr(self.cfg, "enable_dev_workspace", False)),
            "allow_dev_pip_install": bool(getattr(self.cfg, "allow_dev_pip_install", False)),
            "allow_dev_http_server": bool(getattr(self.cfg, "allow_dev_http_server", False)),
            "enable_file_search": bool(getattr(self.cfg, "enable_file_search", False)),
            "file_search_vector_store_ids": list(getattr(self.cfg, "file_search_vector_store_ids", []) or []),
            "file_search_max_num_results": getattr(self.cfg, "file_search_max_num_results", None),
            "include_file_search_results": bool(getattr(self.cfg, "include_file_search_results", False)),
            "allow_python_sandbox": bool(self.cfg.allow_python_sandbox),
            "python_sandbox_timeout_s": float(self.cfg.python_sandbox_timeout_s),
            "conversation": list(self._conversation),
            "agents_sdk": True,
            "use_agents_sdk": True,
        }

    def load_record(self, rec: JsonDict) -> None:
        self.chat_id = str(rec.get("chat_id") or new_chat_id())
        self.title = str(rec.get("title") or "New chat")
        self.cfg.model = str(rec.get("model") or self.cfg.model)
        self.cfg.enable_web_search = bool(rec.get("enable_web_search", self.cfg.enable_web_search))
        self.cfg.web_search_context_size = str(rec.get("web_search_context_size") or self.cfg.web_search_context_size)
        try:
            self.cfg.enable_web_fetch = bool(rec.get("enable_web_fetch", getattr(self.cfg, "enable_web_fetch", True)))
            self.cfg.web_fetch_timeout_s = float(rec.get("web_fetch_timeout_s", getattr(self.cfg, "web_fetch_timeout_s", 20.0)))
            self.cfg.web_fetch_max_chars = int(rec.get("web_fetch_max_chars", getattr(self.cfg, "web_fetch_max_chars", 120_000)))
            self.cfg.web_fetch_cache_ttl_s = float(
                rec.get("web_fetch_cache_ttl_s", getattr(self.cfg, "web_fetch_cache_ttl_s", 30 * 60.0))
            )
            self.cfg.web_fetch_readability = bool(rec.get("web_fetch_readability", getattr(self.cfg, "web_fetch_readability", True)))
        except Exception:
            pass
        try:
            self.cfg.enable_playwright_browser = bool(
                rec.get("enable_playwright_browser", getattr(self.cfg, "enable_playwright_browser", False))
            )
            self.cfg.playwright_headless = bool(rec.get("playwright_headless", getattr(self.cfg, "playwright_headless", True)))
            self.cfg.playwright_watch_mode = bool(rec.get("playwright_watch_mode", getattr(self.cfg, "playwright_watch_mode", False)))
            self.cfg.playwright_screenshot_full_page = bool(
                rec.get("playwright_screenshot_full_page", getattr(self.cfg, "playwright_screenshot_full_page", False))
            )
            self.cfg.allow_playwright_eval = bool(rec.get("allow_playwright_eval", getattr(self.cfg, "allow_playwright_eval", False)))
        except Exception:
            pass
        try:
            self.cfg.use_agents_sdk = bool(rec.get("use_agents_sdk", getattr(self.cfg, "use_agents_sdk", True)))
        except Exception:
            pass
        try:
            self.cfg.enable_telegram_bridge = bool(
                rec.get("enable_telegram_bridge", getattr(self.cfg, "enable_telegram_bridge", False))
            )
            self.cfg.telegram_send_tool_events = bool(
                rec.get("telegram_send_tool_events", getattr(self.cfg, "telegram_send_tool_events", False))
            )
        except Exception:
            pass
        try:
            self.cfg.enable_dev_workspace = bool(rec.get("enable_dev_workspace", getattr(self.cfg, "enable_dev_workspace", False)))
            self.cfg.allow_dev_pip_install = bool(
                rec.get("allow_dev_pip_install", getattr(self.cfg, "allow_dev_pip_install", False))
            )
            self.cfg.allow_dev_http_server = bool(
                rec.get("allow_dev_http_server", getattr(self.cfg, "allow_dev_http_server", False))
            )
        except Exception:
            pass

        self._system_prompt = str(rec.get("system_prompt") or self._system_prompt)
        conv = rec.get("conversation")
        self._conversation = [x for x in conv if isinstance(x, dict)] if isinstance(conv, list) else []

    # ---- tools ----

    def _install_default_tools(self) -> None:
        # read_file
        read_handler = make_read_file_handler(base_dir=self.cfg.tool_base_dir)

        def gated_read(args: JsonDict) -> str:
            if not self.cfg.allow_read_file:
                return json.dumps({"ok": False, "error": "read_file disabled"})
            return read_handler(args)

        self.registry.add(tool_spec=read_file_tool_spec(), handler=gated_read)

        # write/append
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

        # python sandbox
        py_handler = make_python_sandbox_handler(base_dir=self.cfg.tool_base_dir, timeout_s=float(self.cfg.python_sandbox_timeout_s))

        def gated_py(args: JsonDict) -> str:
            if not self.cfg.allow_python_sandbox:
                return json.dumps({"ok": False, "error": "python_sandbox disabled"})
            return py_handler(args)

        self.registry.add(tool_spec=python_sandbox_tool_spec(), handler=gated_py)
        self.registry.add(tool_spec=render_plot_tool_spec(), handler=make_render_plot_handler(python_sandbox_handler=gated_py))

        # web_fetch
        web_fetch_handler = make_web_fetch_handler(
            cache_dir=(self.cfg.tool_base_dir / "chat_history" / "web_cache"),
            default_timeout_s=float(getattr(self.cfg, "web_fetch_timeout_s", 20.0)),
            default_max_chars=int(getattr(self.cfg, "web_fetch_max_chars", 120_000)),
            default_cache_ttl_s=float(getattr(self.cfg, "web_fetch_cache_ttl_s", 30 * 60.0)),
            default_readability=bool(getattr(self.cfg, "web_fetch_readability", True)),
        )

        def gated_fetch(args: JsonDict) -> str:
            if not bool(getattr(self.cfg, "enable_web_fetch", True)) or not bool(getattr(self.cfg, "allow_web_fetch", True)):
                return json.dumps({"ok": False, "error": "web_fetch disabled"})
            return web_fetch_handler(args)

        self.registry.add(tool_spec=web_fetch_tool_spec(), handler=gated_fetch)

        # playwright_browser
        pw_cmd = ["cmd.exe", "/c", "npx", "-y", "@playwright/mcp@latest"]
        if bool(getattr(self.cfg, "playwright_headless", True)):
            pw_cmd += ["--headless"]
        img_dir = (self.cfg.tool_base_dir / "chat_history" / "browser").resolve()
        pw_handler, pw_shutdown = make_playwright_browser_handler(
            cmd=pw_cmd,
            repo_root=self.cfg.tool_base_dir,
            image_out_dir=img_dir,
            call_timeout_s=600.0,
        )
        self._playwright_shutdown = pw_shutdown

        def gated_pw(args: JsonDict) -> str:
            if not bool(getattr(self.cfg, "enable_playwright_browser", False)) or not bool(getattr(self.cfg, "allow_playwright_browser", False)):
                return json.dumps({"ok": False, "error": "playwright_browser disabled"})
            action = str(args.get("action") or "")
            if action in {"browser_evaluate", "browser_run_code"} and (not bool(getattr(self.cfg, "allow_playwright_eval", False))):
                return json.dumps({"ok": False, "error": f"{action} disabled"})
            merged = dict(args)
            merged.setdefault("watch", bool(getattr(self.cfg, "playwright_watch_mode", False)))
            merged.setdefault("screenshot_full_page", bool(getattr(self.cfg, "playwright_screenshot_full_page", False)))
            return pw_handler(merged)

        self.registry.add(tool_spec=playwright_browser_tool_spec(), handler=gated_pw)

        # ---- Dev workspace tools ----
        ws_handlers = make_workspace_handlers(
            repo_root=self.cfg.tool_base_dir,
            get_chat_id=lambda: str(self.chat_id),
            allow_workspace=lambda: bool(getattr(self.cfg, "enable_dev_workspace", False)),
            allow_pip=lambda: bool(getattr(self.cfg, "allow_dev_pip_install", False)),
            allow_http_server=lambda: bool(getattr(self.cfg, "allow_dev_http_server", False)),
            approve_pip=lambda prompt: bool(self._pip_approver(prompt)),
        )
        self.registry.add(tool_spec=workspace_info_tool_spec(), handler=ws_handlers["workspace_info"])
        self.registry.add(tool_spec=workspace_write_tool_spec(), handler=ws_handlers["workspace_write"])
        self.registry.add(tool_spec=workspace_append_tool_spec(), handler=ws_handlers["workspace_append"])
        self.registry.add(tool_spec=workspace_read_tool_spec(), handler=ws_handlers["workspace_read"])
        self.registry.add(tool_spec=workspace_list_tool_spec(), handler=ws_handlers["workspace_list"])
        self.registry.add(tool_spec=workspace_create_venv_tool_spec(), handler=ws_handlers["workspace_create_venv"])
        self.registry.add(tool_spec=workspace_pip_install_tool_spec(), handler=ws_handlers["workspace_pip_install"])
        self.registry.add(tool_spec=workspace_run_python_tool_spec(), handler=ws_handlers["workspace_run_python"])
        self.registry.add(tool_spec=workspace_http_server_start_tool_spec(), handler=ws_handlers["workspace_http_server_start"])
        self.registry.add(tool_spec=workspace_http_server_stop_tool_spec(), handler=ws_handlers["workspace_http_server_stop"])

    def restart_playwright(self) -> None:
        try:
            if self._playwright_shutdown is not None:
                self._playwright_shutdown()
        except Exception:
            pass

    def _agents_tools(self) -> list[Tool]:
        out: list[Tool] = []

        # Convert our function tool registry to Agents SDK FunctionTools.
        for name, spec in self.registry.tools.items():
            if not isinstance(spec, dict) or spec.get("type") != "function":
                continue
            params = spec.get("parameters")
            if not isinstance(params, dict) or params.get("type") != "object":
                continue
            desc = str(spec.get("description") or "")

            handler = self.registry.handlers.get(name)
            if handler is None:
                continue

            async def _invoke(ctx, args_json: str, *, _h=handler) -> Any:  # noqa: ANN001
                try:
                    args = json.loads(args_json or "{}")
                    if not isinstance(args, dict):
                        raise ToolError("tool args must be an object")
                except Exception as e:  # noqa: BLE001
                    return {"ok": False, "error": f"Invalid JSON args: {type(e).__name__}: {e}"}
                try:
                    out_s = _h(args)
                except Exception as e:  # noqa: BLE001
                    return {"ok": False, "error": f"{type(e).__name__}: {e}"}
                # If the handler returns JSON, return it as parsed dict so the model can use it.
                if isinstance(out_s, str) and out_s.strip().startswith("{"):
                    try:
                        d = json.loads(out_s)
                        return d
                    except Exception:
                        return out_s
                return out_s

            out.append(
                FunctionTool(
                    name=str(name),
                    description=desc,
                    params_json_schema=params,
                    on_invoke_tool=_invoke,
                    # Our tool specs are designed for the legacy Responses tool loop and
                    # may include fields that the Agents SDK strict schema validator rejects.
                    strict_json_schema=False,
                )
            )

        if bool(self.cfg.enable_web_search):
            out.append(WebSearchTool(search_context_size=str(self.cfg.web_search_context_size or "medium")))

        return out

    # ---- send ----

    def send_stream(self, user_text: str) -> Iterator[JsonDict]:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be non-empty")

        # Append user input (this mirrors legacy behavior).
        start_len = len(self._conversation)
        clean_user = user_text.strip()
        prompt_tok_est = self.estimate_prompt_tokens(user_text=clean_user)
        self._conversation.append(_input_text("user", clean_user))

        q: queue.Queue[JsonDict] = queue.Queue()
        done = threading.Event()

        def _run() -> None:
            try:
                tools = self._agents_tools()
                agent = Agent(
                    name="Main",
                    model=str(self.cfg.model),
                    instructions=str(self._system_prompt or ""),
                    tools=tools,
                )
                new_items: list[JsonDict] = []
                buf: list[str] = []

                async def _amain() -> _TurnResult:
                    rr = Runner.run_streamed(agent, input=list(self._conversation), max_turns=8)
                    async for ev in rr.stream_events():
                        if isinstance(ev, RawResponsesStreamEvent):
                            d = ev.data
                            if isinstance(d, dict):
                                et = d.get("type")
                                if et == "response.output_text.delta":
                                    delta = d.get("delta")
                                    if isinstance(delta, str) and delta:
                                        q.put({"type": "assistant_delta", "delta": delta})
                                        buf.append(delta)
                        elif isinstance(ev, RunItemStreamEvent):
                            if ev.name == "tool_called":
                                item = ev.item
                                raw = getattr(item, "raw_item", None)
                                nm = ""
                                args = ""
                                if isinstance(raw, dict):
                                    nm = str(raw.get("name") or raw.get("tool_name") or "")
                                    args = str(raw.get("arguments") or raw.get("args") or "")
                                else:
                                    nm = str(getattr(raw, "name", None) or getattr(raw, "tool_name", None) or "")
                                    args = str(getattr(raw, "arguments", None) or "")
                                if nm:
                                    new_items.append({"type": "function_call", "name": str(nm), "arguments": str(args)})
                            elif ev.name == "tool_output":
                                item = ev.item
                                out_val = getattr(item, "output", None)
                                try:
                                    out_s = json.dumps(out_val, ensure_ascii=False) if isinstance(out_val, (dict, list)) else str(out_val)
                                except Exception:
                                    out_s = str(out_val)
                                new_items.append({"type": "function_call_output", "output": out_s})
                    return _TurnResult(assistant_text="".join(buf), new_items=new_items)

                import asyncio

                tr = asyncio.run(_amain())
                full = tr.assistant_text
                out_tok_est = self.estimate_tokens(full)
                self._last_usage = {
                    "input_tokens": int(prompt_tok_est),
                    "output_tokens": int(out_tok_est),
                    "total_tokens": int(prompt_tok_est + out_tok_est),
                }
                # Append assistant message to local history.
                self._conversation.append(_output_text("assistant", full))
                # Title heuristic.
                if self.title == "New chat":
                    self.title = (clean_user.splitlines()[0][:60] or "New chat").strip()

                # Compute new_items in terms of our state change (best-effort).
                q.put({"type": "turn_done", "new_items": tr.new_items})
            except Exception as e:  # noqa: BLE001
                q.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
                # Roll back user append on failure to keep state sane.
                try:
                    self._conversation = self._conversation[:start_len]
                except Exception:
                    pass
            finally:
                done.set()

        threading.Thread(target=_run, daemon=True).start()

        while True:
            try:
                ev = q.get(timeout=0.1)
                yield ev
                if ev.get("type") == "turn_done":
                    return
            except queue.Empty:
                if done.is_set():
                    return

    def shutdown(self) -> None:
        try:
            self.restart_playwright()
        except Exception:
            pass
        return
