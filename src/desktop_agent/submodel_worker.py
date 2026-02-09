"""desktop_agent.submodel_worker

Subprocess worker for running a model session with tools enabled.

IPC: uses `multiprocessing.connection` (duplex) and exchanges JSON-serializable dicts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

from openai import OpenAI

from .chat_session import ChatConfig
from .tools import (
    ToolRegistry,
    add_to_system_prompt_tool_spec,
    append_file_tool_spec,
    create_and_register_python_tool_spec,
    get_system_prompt_tool_spec,
    make_python_sandbox_handler,
    make_append_file_handler,
    make_add_to_system_prompt_handler,
    make_create_and_register_python_tool_handler,
    make_get_system_prompt_handler,
    make_read_file_handler,
    make_set_system_prompt_handler,
    make_write_file_handler,
    python_sandbox_tool_spec,
    read_file_tool_spec,
    run_responses_with_function_tools,
    self_tool_creator_handler,
    self_tool_creator_tool_spec,
    set_system_prompt_tool_spec,
    write_file_tool_spec,
)


JsonDict = dict[str, Any]


def _input_text(role: str, text: str) -> JsonDict:
    return {"role": role, "content": [{"type": "input_text", "text": text}]}


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
    return {"_repr": repr(x)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Submodel worker process.")
    ap.add_argument("--connect-host", required=True)
    ap.add_argument("--connect-port", type=int, required=True)
    ap.add_argument("--authkey", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--base-dir", default="")
    ap.add_argument("--enable-web-search", action="store_true")
    ap.add_argument("--web-search-context-size", default="medium")
    ap.add_argument("--enable-file-search", action="store_true")
    ap.add_argument("--vector-store-ids", default="")
    ap.add_argument("--file-search-max-num-results", type=int, default=0)
    ap.add_argument("--include-file-search-results", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-output-tokens", type=int, default=1024)
    ap.add_argument("--max-tool-calls", type=int, default=8)
    ap.add_argument("--allow-read-file", action="store_true")
    ap.add_argument("--allow-write-files", action="store_true")
    ap.add_argument("--allow-set-system-prompt", action="store_true")
    ap.add_argument("--allow-propose-tools", action="store_true")
    ap.add_argument("--allow-create-tools", action="store_true")
    ap.add_argument("--allow-python-sandbox", action="store_true")
    ap.add_argument("--python-sandbox-timeout-s", type=float, default=12.0)
    args = ap.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        # Worker relies on env var; parent should set it.
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    # Parent passes authkey as hex; decode back to raw bytes for multiprocessing auth.
    try:
        auth = bytes.fromhex(str(args.authkey))
    except Exception as e:
        raise SystemExit(f"ERROR: invalid --authkey (expected hex): {e}") from e
    conn = Client((args.connect_host, int(args.connect_port)), authkey=auth)

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path.cwd().resolve()

    client = OpenAI()
    registry = ToolRegistry(tools={}, handlers={})

    system_prompt = str(args.system_prompt)
    conversation: list[JsonDict] = []
    reset_after_turn = False

    def set_prompt(p: str) -> None:
        nonlocal system_prompt
        system_prompt = str(p)

    def get_prompt() -> str:
        return system_prompt

    def reset_history(do_reset: bool) -> None:
        nonlocal reset_after_turn
        reset_after_turn = reset_after_turn or bool(do_reset)

    # Tool gating per args
    registry.add(tool_spec=read_file_tool_spec(), handler=make_read_file_handler(base_dir=base_dir))
    registry.add(tool_spec=write_file_tool_spec(), handler=make_write_file_handler(base_dir=base_dir))
    registry.add(tool_spec=append_file_tool_spec(), handler=make_append_file_handler(base_dir=base_dir))
    registry.add(
        tool_spec=python_sandbox_tool_spec(),
        handler=make_python_sandbox_handler(base_dir=base_dir, timeout_s=float(args.python_sandbox_timeout_s)),
    )

    # System prompt tools
    registry.add(tool_spec=set_system_prompt_tool_spec(), handler=make_set_system_prompt_handler(set_prompt=set_prompt, reset_history=reset_history))
    registry.add(tool_spec=get_system_prompt_tool_spec(), handler=make_get_system_prompt_handler(get_prompt=get_prompt))
    registry.add(
        tool_spec=add_to_system_prompt_tool_spec(),
        handler=make_add_to_system_prompt_handler(get_prompt=get_prompt, set_prompt=set_prompt, reset_history=reset_history),
    )

    # Tool creation tools
    registry.add(tool_spec=self_tool_creator_tool_spec(), handler=self_tool_creator_handler)
    registry.add(
        tool_spec=create_and_register_python_tool_spec(),
        handler=make_create_and_register_python_tool_handler(registry=registry),
    )

    # Apply gate switches by wrapping handlers (keeps tool specs stable).
    def wrap(name: str, allow: bool) -> None:
        h = registry.handlers.get(name)
        if h is None:
            return

        def gated(a: JsonDict, *, _h=h, _allow=allow, _name=name) -> str:
            if not _allow:
                return json.dumps({"ok": False, "error": f"{_name} disabled"})
            return _h(a)

        registry.handlers[name] = gated

    wrap("read_file", bool(args.allow_read_file))
    wrap("write_file", bool(args.allow_write_files))
    wrap("append_file", bool(args.allow_write_files))
    wrap("set_system_prompt", bool(args.allow_set_system_prompt))
    wrap("get_system_prompt", bool(args.allow_set_system_prompt))
    wrap("add_to_system_prompt", bool(args.allow_set_system_prompt))
    wrap("propose_function_tool", bool(args.allow_propose_tools))
    wrap("create_and_register_python_tool", bool(args.allow_create_tools))
    wrap("python_sandbox", bool(args.allow_python_sandbox))

    extra_tools: list[JsonDict] = []
    if args.enable_web_search:
        extra_tools.append({"type": "web_search", "search_context_size": str(args.web_search_context_size)})
    if args.enable_file_search:
        raw = str(args.vector_store_ids or "").strip()
        ids = [x.strip() for x in raw.split(",") if x.strip()]
        if ids:
            fs: JsonDict = {"type": "file_search", "vector_store_ids": ids}
            if int(args.file_search_max_num_results or 0) > 0:
                fs["max_num_results"] = int(args.file_search_max_num_results)
            extra_tools.append(fs)

    # Handshake
    conn.send({"type": "hello", "model": args.model})

    while True:
        try:
            msg = conn.recv()
        except (EOFError, ConnectionResetError, OSError):
            # Parent exited or closed the connection; shut down quietly.
            break
        if not isinstance(msg, dict):
            continue
        op = msg.get("op")
        if op == "stop":
            conn.send({"type": "stopped"})
            break
        if op == "ping":
            conn.send({"type": "pong", "ts": time.time()})
            continue
        if op == "run":
            user_input = msg.get("input")
            if not isinstance(user_input, str):
                conn.send({"type": "error", "error": "input must be string"})
                continue

            conn.send({"type": "status", "state": "running"})
            conversation.append(_input_text("user", user_input))

            try:
                create_kwargs: dict[str, Any] = {
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "max_output_tokens": int(args.max_output_tokens),
                    "max_tool_calls": int(args.max_tool_calls),
                }
                if args.enable_file_search and bool(args.include_file_search_results):
                    create_kwargs["include"] = ["file_search_call.results"]

                resp, updated = run_responses_with_function_tools(
                    client=client,
                    model=args.model,
                    input_items=conversation,
                    registry=registry,
                    extra_tools=extra_tools,
                    instructions_provider=lambda: system_prompt,
                    max_rounds=12,
                    return_input_items=True,
                    on_item=lambda it: conn.send({"type": "item", "item": _as_dict(it)}),
                    **create_kwargs,
                )

                out_text = getattr(resp, "output_text", "") or ""
                conn.send({"type": "assistant", "text": str(out_text)})

                # persist updated conversation
                updated_dicts = [_as_dict(x) for x in updated]
                if reset_after_turn:
                    conversation = []
                    reset_after_turn = False
                else:
                    conversation = updated_dicts

                conn.send({"type": "status", "state": "idle"})
                conn.send({"type": "done"})
            except Exception as e:
                conn.send({"type": "error", "error": f"{type(e).__name__}: {e}"})
                conn.send({"type": "status", "state": "idle"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
