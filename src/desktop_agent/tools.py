"""desktop_agent.tools

Utilities for OpenAI tool calling (Responses API) and a safe "self tool creator".

This module is intentionally conservative:
- It supports *proposing* new tools by writing a reviewed-on-disk proposal.
- It does NOT auto-import or execute newly generated code.

Why:
Tool definitions are passed to the model via the Responses API `tools` parameter.
When the model returns `function_call` items, the application is responsible for
executing the function and sending back a `function_call_output` item.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from .dev_workspace import (
    WorkspaceError,
    append_text as ws_append_text,
    create_venv as ws_create_venv,
    ensure_workspace as ws_ensure_workspace,
    http_server_start as ws_http_server_start,
    http_server_stop as ws_http_server_stop,
    list_dir as ws_list_dir,
    pip_install as ws_pip_install,
    read_text as ws_read_text,
    run_venv_python as ws_run_venv_python,
    venv_exists as ws_venv_exists,
    venv_python_path as ws_venv_python_path,
    write_text as ws_write_text,
)


JsonDict = dict[str, Any]
FunctionHandler = Callable[[JsonDict], str]

_LOG = logging.getLogger(__name__)


def _drop_unsupported_param_from_error(e: Exception, create_kwargs: dict[str, Any]) -> str | None:
    """If OpenAI returns 'Unsupported parameter', drop it and return the param name."""

    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            param = err.get("param")
            msg = err.get("message")
            if isinstance(param, str) and param in create_kwargs and isinstance(msg, str) and "Unsupported parameter" in msg:
                create_kwargs.pop(param, None)
                return param

    # Fallback: string-match on common error shapes.
    msg = str(e)
    if "Unsupported parameter" in msg:
        for p in list(create_kwargs.keys()):
            if f"'{p}'" in msg or f"\"{p}\"" in msg:
                create_kwargs.pop(p, None)
                return p
    return None


def _strip_status_field_from_input_items(items: list[JsonDict]) -> list[JsonDict]:
    """Sanitize input items before echoing them back into `responses.create(input=...)`.

    - Some SDK outputs include a top-level `status` field; the Responses API rejects it.
    - Some models return `reasoning` items; those must not be replayed as input (they're
      output-only and can cause 400s when echoed back).
    - Some message items can carry references to missing reasoning items (e.g. `reasoning`
      / `reasoning_id`). If we strip reasoning items, we must also strip those references,
      otherwise the API can reject the input due to missing required items.
    """

    def deep_pop(obj: Any, *, keys: set[str]) -> Any:
        if isinstance(obj, dict):
            d = {}
            for k, v in obj.items():
                if k in keys:
                    continue
                d[k] = deep_pop(v, keys=keys)
            return d
        if isinstance(obj, list):
            return [deep_pop(x, keys=keys) for x in obj]
        return obj

    out: list[JsonDict] = []
    for it in items:
        if isinstance(it, dict) and it.get("type") in {"reasoning", "reasoning_summary"}:
            continue
        if isinstance(it, dict):
            # Remove unsupported fields and reasoning references (top-level and nested).
            d = dict(it)
            d.pop("status", None)
            d = deep_pop(d, keys={"reasoning", "reasoning_id"})
            out.append(d)
        else:
            out.append(it)  # type: ignore[arg-type]
    return out


class ToolError(RuntimeError):
    pass


def _utc_ts_compact() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _validate_tool_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ToolError("tool_name must be a non-empty string")
    if not _TOOL_NAME_RE.match(name):
        raise ToolError("tool_name must match ^[a-z][a-z0-9_]{0,63}$")
    return name


def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ToolError(f"Refusing to overwrite existing file: {path}")
    path.write_text(text, encoding="utf-8")


@dataclass
class ToolRegistry:
    """Mutable tool registry that can change while an agent loop runs."""

    tools: dict[str, JsonDict]
    handlers: dict[str, FunctionHandler]

    def tool_list(self) -> list[JsonDict]:
        return list(self.tools.values())

    def add(self, *, tool_spec: JsonDict, handler: FunctionHandler) -> None:
        name = tool_spec.get("name")
        if not isinstance(name, str) or not name:
            raise ToolError("tool_spec missing name")
        if name in self.tools:
            raise ToolError(f"Tool already exists: {name}")
        self.tools[name] = tool_spec
        self.handlers[name] = handler


@dataclass(frozen=True)
class ToolProposal:
    tool_name: str
    tool_spec_path: Path
    python_stub_path: Path


def propose_function_tool(
    *,
    tool_name: str,
    description: str,
    parameters_schema: JsonDict,
    base_dir: Optional[Path] = None,
) -> ToolProposal:
    """Create an on-disk proposal for a new function tool.

    Writes:
    - a JSON tool definition (Responses API `tools` entry)
    - a Python stub with a handler skeleton
    """

    name = _validate_tool_name(tool_name)
    if not isinstance(description, str) or not description.strip():
        raise ToolError("description must be a non-empty string")
    if not isinstance(parameters_schema, dict) or parameters_schema.get("type") != "object":
        raise ToolError("parameters_schema must be a JSON schema object with type='object'")

    out_root = base_dir or (Path.cwd() / "tool_proposals")
    out_dir = out_root / name / _utc_ts_compact()

    tool_spec: JsonDict = {
        "type": "function",
        "name": name,
        "description": description.strip(),
        "parameters": parameters_schema,
    }

    tool_spec_path = out_dir / "tool.json"
    python_stub_path = out_dir / f"{name}.py"

    _safe_write_text(tool_spec_path, json.dumps(tool_spec, indent=2) + "\n")

    stub = f"""\
\"\"\"Tool stub: {name}

Review + implement this handler, then register it in your tool runner.
\"\"\"

from __future__ import annotations

import json
from typing import Any


def {name}(args: dict[str, Any]) -> str:
    \"\"\"Handler for the `{name}` function tool.

    Args:
        args: Parsed JSON arguments from the model's function_call.arguments.

    Returns:
        A string (often JSON) to send back as function_call_output.output.
    \"\"\"
    # TODO: implement tool logic.
    return json.dumps({{"ok": False, "error": "not implemented"}})
"""
    _safe_write_text(python_stub_path, stub)

    return ToolProposal(tool_name=name, tool_spec_path=tool_spec_path, python_stub_path=python_stub_path)


def self_tool_creator_tool_spec() -> JsonDict:
    """Tool definition to let the model propose new function tools."""

    return {
        "type": "function",
        "name": "propose_function_tool",
        "description": (
            "Create an on-disk proposal for a new function tool (JSON schema + Python stub). "
            "This does not execute code or enable the tool automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "snake_case tool name (letters/numbers/underscore), e.g. fetch_jira_ticket",
                },
                "description": {"type": "string", "description": "What the tool does."},
                "parameters_schema": {
                    "type": "object",
                    "description": "JSON Schema for the tool's arguments (must have type='object').",
                },
            },
            "required": ["tool_name", "description", "parameters_schema"],
            "additionalProperties": False,
        },
    }


def self_tool_creator_handler(args: JsonDict) -> str:
    proposal = propose_function_tool(
        tool_name=str(args.get("tool_name", "")),
        description=str(args.get("description", "")),
        parameters_schema=args.get("parameters_schema") if isinstance(args.get("parameters_schema"), dict) else {},
    )
    return json.dumps(
        {
            "ok": True,
            "tool_name": proposal.tool_name,
            "tool_spec_path": str(proposal.tool_spec_path),
            "python_stub_path": str(proposal.python_stub_path),
            "next": "Review/edit the stub and then wire it into your tool runner.",
        }
    )


def run_responses_with_function_tools(
    *,
    client: Any,
    model: str,
    input_items: list[JsonDict],
    tools: Optional[Sequence[JsonDict]] = None,
    handlers: Optional[dict[str, FunctionHandler]] = None,
    registry: Optional[ToolRegistry] = None,
    extra_tools: Optional[Sequence[JsonDict]] = None,
    instructions: Optional[str] = None,
    instructions_provider: Optional[Callable[[], Optional[str]]] = None,
    on_item: Optional[Callable[[JsonDict], None]] = None,
    on_unsupported_param: Optional[Callable[[str], None]] = None,
    max_rounds: int = 8,
    return_input_items: bool = False,
    **create_kwargs: Any,
) -> Any:
    """Run a Responses API call and handle `function_call` items.

    This implements the standard tool-calling flow:
    1) create response with tools
    2) execute each `function_call`
    3) append `function_call_output`
    4) call again until no more tool calls

    The response `output` contains `function_call` items with `call_id`, `name`,
    and JSON-encoded `arguments`.
    """

    if max_rounds < 1:
        raise ToolError("max_rounds must be >= 1")

    if registry is None:
        if tools is None or handlers is None:
            raise ToolError("Provide either (tools, handlers) or registry")
    input_list = list(input_items)
    cw = dict(create_kwargs)

    for _ in range(max_rounds):
        if registry is not None:
            round_tools = registry.tool_list() + list(extra_tools or [])
        else:
            round_tools = list(tools or [])
        round_handlers = registry.handlers if registry is not None else (handlers or {})

        cur_instructions = instructions_provider() if instructions_provider is not None else instructions
        last_err: Exception | None = None
        for _drop_try in range(8):
            try:
                resp = client.responses.create(
                    model=model,
                    input=_strip_status_field_from_input_items(input_list),
                    tools=list(round_tools),
                    instructions=cur_instructions,
                    **cw,
                )
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                dropped = _drop_unsupported_param_from_error(e, cw)
                if dropped is None:
                    raise
                if on_unsupported_param is not None:
                    try:
                        on_unsupported_param(dropped)
                    except Exception:
                        pass
        if last_err is not None:
            raise last_err

        output = getattr(resp, "output", None)
        if not isinstance(output, list):
            return (resp, input_list) if return_input_items else resp

        input_list += output
        if on_item is not None:
            for item in output:
                try:
                    on_item(item if isinstance(item, dict) else {"_repr": repr(item)})
                except Exception:
                    pass

        any_calls = False
        for item in output:
            item_type = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
            if item_type != "function_call":
                continue

            any_calls = True
            call_id = getattr(item, "call_id", None) if not isinstance(item, dict) else item.get("call_id")
            name = getattr(item, "name", None) if not isinstance(item, dict) else item.get("name")
            arguments = getattr(item, "arguments", None) if not isinstance(item, dict) else item.get("arguments")

            if not isinstance(call_id, str) or not call_id:
                raise ToolError("Malformed function_call: missing call_id")
            if not isinstance(name, str) or not name:
                raise ToolError("Malformed function_call: missing name")
            if not isinstance(arguments, str):
                raise ToolError("Malformed function_call: missing arguments")

            handler = round_handlers.get(name)
            if handler is None:
                out = json.dumps({"ok": False, "error": f"unknown tool: {name}"})
            else:
                try:
                    parsed = json.loads(arguments) if arguments else {}
                    if not isinstance(parsed, dict):
                        raise ToolError("tool arguments must decode to an object")
                    out = handler(parsed)
                except Exception as e:  # noqa: BLE001
                    out = json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})

            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": out,
                }
            )
            if on_item is not None:
                try:
                    on_item({"type": "function_call_output", "call_id": call_id, "output": out})
                except Exception:
                    pass

        if not any_calls:
            return (resp, input_list) if return_input_items else resp

    raise ToolError(f"Exceeded max_rounds={max_rounds} while handling tool calls")


def _as_event_dict(evt: Any) -> JsonDict:
    if isinstance(evt, dict):
        return evt
    if hasattr(evt, "model_dump"):
        try:
            d = evt.model_dump()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {"_repr": repr(evt)}


def run_responses_with_function_tools_stream(
    *,
    client: Any,
    model: str,
    input_items: list[JsonDict],
    tools: Optional[Sequence[JsonDict]] = None,
    handlers: Optional[dict[str, FunctionHandler]] = None,
    registry: Optional[ToolRegistry] = None,
    extra_tools: Optional[Sequence[JsonDict]] = None,
    instructions: Optional[str] = None,
    instructions_provider: Optional[Callable[[], Optional[str]]] = None,
    on_item: Optional[Callable[[JsonDict], None]] = None,
    on_text_delta: Optional[Callable[[str], None]] = None,
    on_unsupported_param: Optional[Callable[[str], None]] = None,
    max_rounds: int = 8,
    return_input_items: bool = False,
    **create_kwargs: Any,
) -> Iterator[JsonDict]:
    """Streaming variant of `run_responses_with_function_tools`.

    Yields events:
    - {"type": "text_delta", "delta": "..."} (best-effort)
    - {"type": "round_completed"} with the completed response dict
    - {"type": "done"} with the final response dict
    """

    if max_rounds < 1:
        raise ToolError("max_rounds must be >= 1")
    if registry is None and (tools is None or handlers is None):
        raise ToolError("Provide either (tools, handlers) or registry")

    input_list = list(input_items)
    cw = dict(create_kwargs)
    last_resp: Any = None

    for _ in range(max_rounds):
        if registry is not None:
            round_tools = registry.tool_list() + list(extra_tools or [])
        else:
            round_tools = list(tools or [])
        round_handlers = registry.handlers if registry is not None else (handlers or {})

        cur_instructions = instructions_provider() if instructions_provider is not None else instructions

        completed_resp: Any = None
        for _drop_try in range(8):
            yielded_any = False
            completed_resp = None
            # Create the stream (may raise immediately).
            try:
                stream = client.responses.create(
                    model=model,
                    input=_strip_status_field_from_input_items(input_list),
                    tools=list(round_tools),
                    instructions=cur_instructions,
                    stream=True,
                    **cw,
                )
            except Exception as e:  # noqa: BLE001
                dropped = _drop_unsupported_param_from_error(e, cw)
                if dropped is None:
                    raise
                if on_unsupported_param is not None:
                    try:
                        on_unsupported_param(dropped)
                    except Exception:
                        pass
                continue

            try:
                for evt in stream:
                    ed = _as_event_dict(evt)
                    etype = str(ed.get("type") or "")

                    # Best-effort text streaming: handle common delta shapes.
                    delta = ed.get("delta")
                    if isinstance(delta, str) and ("output_text" in etype or etype.endswith(".delta") or "text" in etype):
                        yielded_any = True
                        if on_text_delta is not None:
                            try:
                                on_text_delta(delta)
                            except Exception:
                                pass
                        yield {"type": "text_delta", "delta": delta}

                    # Completed response is usually embedded in the event.
                    if "response" in ed and isinstance(ed.get("response"), dict):
                        # Some events include the response payload repeatedly; take the latest.
                        if etype.endswith("completed") or etype.endswith("done") or etype.endswith("response.completed"):
                            completed_resp = ed["response"]
                        else:
                            completed_resp = ed["response"]

                    # Some SDKs use `response` attribute, but not exposed in dict.
                    if completed_resp is None and hasattr(evt, "response"):
                        try:
                            r = getattr(evt, "response")
                            if r is not None:
                                completed_resp = r
                        except Exception:
                            pass
                break
            except Exception as e:  # noqa: BLE001
                dropped = _drop_unsupported_param_from_error(e, cw)
                if dropped is None or yielded_any:
                    raise
                if on_unsupported_param is not None:
                    try:
                        on_unsupported_param(dropped)
                    except Exception:
                        pass
                continue

        # Fallback: if stream didn't surface a response object, try to use the last event.
        resp = completed_resp if completed_resp is not None else last_resp
        last_resp = resp

        output = None
        if isinstance(resp, dict):
            output = resp.get("output")
        else:
            output = getattr(resp, "output", None)

        if not isinstance(output, list):
            yield {"type": "done", "response": _as_event_dict(resp)}
            if return_input_items:
                yield {"type": "input_items", "input": input_list}
            return

        input_list += output
        if on_item is not None:
            for item in output:
                try:
                    on_item(item if isinstance(item, dict) else {"_repr": repr(item)})
                except Exception:
                    pass

        any_calls = False
        for item in output:
            item_type = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
            if item_type != "function_call":
                continue

            any_calls = True
            call_id = getattr(item, "call_id", None) if not isinstance(item, dict) else item.get("call_id")
            name = getattr(item, "name", None) if not isinstance(item, dict) else item.get("name")
            arguments = getattr(item, "arguments", None) if not isinstance(item, dict) else item.get("arguments")

            if not isinstance(call_id, str) or not call_id:
                raise ToolError("Malformed function_call: missing call_id")
            if not isinstance(name, str) or not name:
                raise ToolError("Malformed function_call: missing name")
            if not isinstance(arguments, str):
                raise ToolError("Malformed function_call: missing arguments")

            handler = round_handlers.get(name)
            if handler is None:
                out = json.dumps({"ok": False, "error": f"unknown tool: {name}"})
            else:
                try:
                    parsed = json.loads(arguments) if arguments else {}
                    if not isinstance(parsed, dict):
                        raise ToolError("tool arguments must decode to an object")
                    out = handler(parsed)
                except Exception as e:  # noqa: BLE001
                    out = json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})

            input_list.append({"type": "function_call_output", "call_id": call_id, "output": out})
            if on_item is not None:
                try:
                    on_item({"type": "function_call_output", "call_id": call_id, "output": out})
                except Exception:
                    pass

        yield {"type": "round_completed", "response": _as_event_dict(resp)}

        if not any_calls:
            yield {"type": "done", "response": _as_event_dict(resp)}
            if return_input_items:
                yield {"type": "input_items", "input": input_list}
            return

    raise ToolError(f"Exceeded max_rounds={max_rounds} while handling tool calls")


def default_self_tool_creator() -> tuple[list[JsonDict], dict[str, FunctionHandler]]:
    """Convenience: tool spec + handler mapping for the self tool creator."""

    return ([self_tool_creator_tool_spec()], {"propose_function_tool": self_tool_creator_handler})


# ---- Built-in tools ----

def playwright_browser_tool_spec() -> JsonDict:
    """Tool spec for controlling a Playwright MCP browser session."""

    return {
        "type": "function",
        "name": "playwright_browser",
        "description": (
            "Control a real browser via the Playwright MCP server. "
            "Use this for interactive web automation (navigate, click, type, snapshot, screenshot). "
            "Common actions: browser_install (first time), browser_tabs, browser_navigate, "
            "browser_snapshot, browser_take_screenshot, browser_click, browser_type, browser_press_key."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "A Playwright MCP tool name, e.g. browser_install, browser_navigate, browser_click, "
                        "browser_type, browser_snapshot, browser_take_screenshot."
                    ),
                },
                "params": {"type": "object", "description": "Arguments for the action.", "default": {}},
                "restart": {"type": "boolean", "description": "If true, restart the browser session first.", "default": False},
                "watch": {
                    "type": "boolean",
                    "description": "If true, automatically take a screenshot after the action (best-effort).",
                    "default": False,
                },
                "screenshot_full_page": {
                    "type": "boolean",
                    "description": "When watch=true (or action is screenshot), request a full-page screenshot.",
                    "default": False,
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    }


def make_playwright_browser_handler(
    *,
    cmd: list[str],
    repo_root: Path,
    image_out_dir: Path,
    startup_timeout_s: float = 60.0,
    call_timeout_s: float = 120.0,
    auto_install: bool = True,
    fixed_image_name: str | None = None,
) -> tuple[FunctionHandler, Callable[[], None]]:
    """Create a handler for `playwright_browser` backed by a long-lived MCP server.

    Returns:
        (handler, shutdown)
    """

    repo_root = Path(repo_root).resolve()
    image_out_dir = Path(image_out_dir).resolve()
    image_out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import so app still runs without this optional dependency.
    try:
        from agents.mcp import MCPServerStdio, MCPServerStdioParams  # type: ignore
    except Exception as e:  # noqa: BLE001

        def _missing(_: JsonDict) -> str:
            return json.dumps({"ok": False, "error": f"Playwright MCP unavailable: {type(e).__name__}: {e}"})

        return _missing, (lambda: None)

    import asyncio
    import threading

    lock = threading.Lock()
    loop: asyncio.AbstractEventLoop | None = None
    server: MCPServerStdio | None = None
    ready = threading.Event()
    stop_ev: asyncio.Event | None = None
    thread: threading.Thread | None = None
    did_auto_install = False

    def _start_thread() -> None:
        nonlocal loop, server, stop_ev, thread
        if thread is not None and thread.is_alive():
            return

        ready.clear()

        def _runner() -> None:
            nonlocal loop, server, stop_ev
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _main() -> None:
                nonlocal server, stop_ev
                stop_ev = asyncio.Event()
                params = MCPServerStdioParams(command=cmd[0], args=cmd[1:])
                srv = MCPServerStdio(params, client_session_timeout_seconds=startup_timeout_s)
                try:
                    async with srv:
                        server = srv
                        ready.set()
                        await stop_ev.wait()
                finally:
                    server = None

            try:
                loop.run_until_complete(_main())
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        if not ready.wait(timeout=startup_timeout_s + 5.0):
            raise ToolError("Playwright MCP failed to start (timeout)")

    def _shutdown() -> None:
        nonlocal loop, server, stop_ev, thread
        with lock:
            if loop is not None and stop_ev is not None:
                try:
                    loop.call_soon_threadsafe(stop_ev.set)
                except Exception:
                    pass
            loop = None
            server = None
            stop_ev = None
            t = thread
            thread = None
        if t is not None and t.is_alive():
            try:
                t.join(timeout=2.0)
            except Exception:
                pass

    def _restart() -> None:
        _shutdown()
        _start_thread()

    def _call_tool(action: str, params: dict[str, Any]) -> Any:
        if loop is None or server is None:
            _start_thread()
        assert loop is not None
        assert server is not None

        async def _do() -> Any:
            return await server.call_tool(action, params)

        fut = asyncio.run_coroutine_threadsafe(_do(), loop)
        return fut.result(timeout=call_timeout_s)

    def _result_to_json(res: Any) -> JsonDict:
        out_text_parts: list[str] = []
        image_paths: list[str] = []
        contents = getattr(res, "content", None)
        if isinstance(contents, list):
            for c in contents:
                ctype = getattr(c, "type", None)
                if ctype == "text":
                    txt = getattr(c, "text", "")
                    if isinstance(txt, str) and txt:
                        out_text_parts.append(txt)
                elif ctype == "image":
                    data = getattr(c, "data", None)
                    mime = getattr(c, "mimeType", None) or getattr(c, "mime_type", None)
                    if isinstance(data, str) and data:
                        ext = "png"
                        if isinstance(mime, str) and "jpeg" in mime.lower():
                            ext = "jpg"
                        if isinstance(fixed_image_name, str) and fixed_image_name.strip():
                            # Single-file mode (used by browser-first UIs): overwrite a stable filename.
                            base = fixed_image_name.strip()
                            if "." in base:
                                name = base
                            else:
                                name = f"{base}.{ext}"
                        else:
                            name = f"pw_{uuid.uuid4().hex[:10]}.{ext}"
                        abs_path = (image_out_dir / name).resolve()
                        try:
                            abs_path.write_bytes(base64.b64decode(data))
                            rel = abs_path.relative_to(repo_root)
                            image_paths.append(rel.as_posix())
                        except Exception:
                            continue
        return {"text": "\n".join(out_text_parts).strip(), "image_paths": image_paths}

    def _handler(args: JsonDict) -> str:
        nonlocal did_auto_install
        action = args.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ToolError("action must be a non-empty string")
        action = action.strip()
        params = args.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ToolError("params must be an object")
        restart = bool(args.get("restart", False))
        watch = bool(args.get("watch", False))
        full_page = bool(args.get("screenshot_full_page", False))

        with lock:
            if restart:
                _restart()
            if (thread is None) or (not thread.is_alive()) or (server is None) or (loop is None):
                _start_thread()

        # Best-effort: ensure the Playwright browser binaries exist.
        # Without this, the first "real" navigation can appear to do nothing while Playwright installs.
        # We do it once per handler lifecycle.
        if auto_install and (not did_auto_install) and action != "browser_install":
            try:
                _LOG.info("playwright_mcp_auto_install_start")
                _call_tool("browser_install", {})
                did_auto_install = True
                _LOG.info("playwright_mcp_auto_install_done")
            except Exception as e:  # noqa: BLE001
                # Don't fail the user's action just because install failed.
                _LOG.warning("playwright_mcp_auto_install_failed error=%s", f"{type(e).__name__}: {e}")

        def do_call() -> Any:
            return _call_tool(action, params)

        try:
            _LOG.info("playwright_mcp_call action=%s", action)
            res = do_call()
        except Exception as e:  # noqa: BLE001
            # Best-effort auto-restart on common transport crashes.
            msg = str(e)
            retryable = any(x in msg.lower() for x in ("epipe", "broken pipe", "disconnected", "cannot switch to a different thread"))
            if (not restart) and retryable:
                try:
                    with lock:
                        _restart()
                    res = do_call()
                except Exception as e2:  # noqa: BLE001
                    _LOG.error(
                        "playwright_mcp_call_failed action=%s error=%s", action, f"{type(e2).__name__}: {e2}"
                    )
                    return json.dumps(
                        {"ok": False, "error": f"{type(e2).__name__}: {e2}", "action": action, "hint": "try restart=true"},
                        ensure_ascii=False,
                    )
            else:
                _LOG.error("playwright_mcp_call_failed action=%s error=%s", action, f"{type(e).__name__}: {e}")
                return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "action": action}, ensure_ascii=False)

        payload = _result_to_json(res)

        image_paths: list[str] = []
        if isinstance(payload.get("image_paths"), list):
            image_paths.extend([str(x) for x in payload.get("image_paths") if isinstance(x, str)])

        # Watch mode: take a screenshot after any non-screenshot action.
        if watch and action not in {"browser_take_screenshot"}:
            try:
                res2 = _call_tool("browser_take_screenshot", {"fullPage": bool(full_page)})
                p2 = _result_to_json(res2)
                if isinstance(p2.get("image_paths"), list):
                    for p in p2["image_paths"]:
                        if isinstance(p, str) and p and p not in image_paths:
                            image_paths.append(p)
            except Exception:
                pass

        out: JsonDict = {"ok": True, "action": action, "text": str(payload.get("text") or "").strip(), "image_paths": image_paths}
        return json.dumps(out, ensure_ascii=False)

    return _handler, _shutdown


def web_fetch_tool_spec() -> JsonDict:
    """Tool spec for fetching a web page (HTTP GET) with optional readability extraction."""

    return {
        "type": "function",
        "name": "web_fetch",
        "description": (
            "Fetch a URL over HTTP(S) and return its text content (optionally extracting main content). "
            "Use this when web_search results need deeper reading."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
                "timeout_s": {"type": "number", "description": "Request timeout in seconds (optional)."},
                "max_chars": {"type": "integer", "description": "Maximum characters to return (optional)."},
                "readability": {"type": "boolean", "description": "If true, attempt to extract main article content."},
                "cache_ttl_s": {"type": "number", "description": "Cache TTL in seconds (optional)."},
                "no_cache": {"type": "boolean", "description": "If true, bypass cache (optional)."},
                "user_agent": {"type": "string", "description": "Override user-agent (optional)."},
                "max_redirects": {"type": "integer", "description": "Maximum redirects to follow (optional)."},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    }


def _safe_cache_key(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()
    return h[:32]


def _html_to_text_fallback(html_text: str, *, max_chars: int) -> str:
    # Very small, dependency-free HTML->text fallback.
    t = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_text or "")
    t = re.sub(r"(?is)<style.*?>.*?</style>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    try:
        import html as _html  # stdlib

        t = _html.unescape(t)
    except Exception:
        pass
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    if len(t) > max_chars:
        t = t[:max_chars]
    return t


def _readability_extract(html_text: str, *, max_chars: int) -> str:
    # Prefer readability-lxml when available, else fallback.
    try:
        from readability import Document  # type: ignore[import-not-found]
        import lxml.html  # type: ignore[import-not-found]

        doc = Document(html_text or "")
        summary_html = doc.summary(html_partial=True) or ""
        root = lxml.html.fromstring(summary_html)
        txt = root.text_content() or ""
        txt = re.sub(r"[ \t\r\f\v]+", " ", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        txt = txt.strip()
        if len(txt) > max_chars:
            txt = txt[:max_chars]
        return txt
    except Exception:
        return _html_to_text_fallback(html_text or "", max_chars=max_chars)


def make_web_fetch_handler(
    *,
    cache_dir: Path,
    default_timeout_s: float = 20.0,
    default_max_chars: int = 120_000,
    default_cache_ttl_s: float = 30 * 60.0,
    default_readability: bool = True,
) -> FunctionHandler:
    """Create a handler for `web_fetch`."""

    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _handler(args: JsonDict) -> str:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ToolError("url must be a non-empty string")
        url = url.strip()

        timeout_s = float(args.get("timeout_s", default_timeout_s))
        max_chars = int(args.get("max_chars", default_max_chars))
        readability = bool(args.get("readability", default_readability))
        cache_ttl_s = float(args.get("cache_ttl_s", default_cache_ttl_s))
        no_cache = bool(args.get("no_cache", False))
        user_agent = args.get("user_agent")
        max_redirects = int(args.get("max_redirects", 8))

        if not (1.0 <= timeout_s <= 120.0):
            raise ToolError("timeout_s must be between 1 and 120")
        if not (1_000 <= max_chars <= 1_000_000):
            raise ToolError("max_chars must be between 1000 and 1000000")
        if not (0.0 <= cache_ttl_s <= 24 * 3600.0):
            raise ToolError("cache_ttl_s must be between 0 and 86400")
        if not (0 <= max_redirects <= 16):
            raise ToolError("max_redirects must be between 0 and 16")
        if user_agent is not None and not isinstance(user_agent, str):
            raise ToolError("user_agent must be a string")

        cache_key = _safe_cache_key(url)
        cache_path = cache_dir / f"{cache_key}.json"
        now = time.time()
        if (not no_cache) and cache_ttl_s > 0 and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                ts = float(cached.get("ts", 0.0))
                if (now - ts) <= cache_ttl_s and isinstance(cached.get("result"), dict):
                    res = dict(cached["result"])
                    res["cached"] = True
                    return json.dumps(res, ensure_ascii=False)
            except Exception:
                pass

        try:
            import httpx  # type: ignore[import-not-found]

            headers = {
                "User-Agent": (user_agent.strip() if isinstance(user_agent, str) and user_agent.strip() else "desktop-agent/1.0"),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
            }
            with httpx.Client(
                timeout=timeout_s,
                follow_redirects=True,
                headers=headers,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                max_redirects=max_redirects,
            ) as client:
                r = client.get(url)
                status = int(r.status_code)
                final_url = str(r.url)
                ctype = r.headers.get("content-type", "")
                text = r.text if isinstance(r.text, str) else str(r.content.decode("utf-8", errors="replace"))
        except Exception as e:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "url": url}, ensure_ascii=False)

        is_html = "text/html" in (ctype or "").lower()
        if readability and is_html:
            out_text = _readability_extract(text, max_chars=max_chars)
        else:
            out_text = text
            if len(out_text) > max_chars:
                out_text = out_text[:max_chars]

        res: JsonDict = {
            "ok": True,
            "url": url,
            "final_url": final_url,
            "status_code": status,
            "content_type": ctype,
            "text": out_text,
            "cached": False,
            "truncated": len(out_text) >= max_chars,
        }

        if (not no_cache) and cache_ttl_s > 0:
            try:
                cache_path.write_text(json.dumps({"ts": now, "result": res}, ensure_ascii=False) + "\n", encoding="utf-8")
            except Exception:
                pass

        return json.dumps(res, ensure_ascii=False)

    return _handler


def read_file_tool_spec() -> JsonDict:
    """Tool spec for reading a text file from disk (repo-scoped by default)."""

    return {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file and return a snippet by line range. "
            "This tool is scoped to a configured base directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (relative to the tool base dir)."},
                "start_line": {"type": "integer", "description": "1-based start line (default: 1)."},
                "max_lines": {"type": "integer", "description": "Maximum number of lines to return (default: 200)."},
                "max_chars": {"type": "integer", "description": "Maximum characters to return (default: 20000)."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    }


def make_read_file_handler(*, base_dir: Optional[Path] = None) -> FunctionHandler:
    """Create a handler for `read_file`.

    Safety:
    - Restricts reads to within `base_dir` (defaults to current working directory).
    - Reads text as UTF-8 with replacement for errors.
    - Enforces max lines/chars to prevent huge outputs.
    """

    root = (base_dir or Path.cwd()).resolve()

    def _handler(args: JsonDict) -> str:
        raw_path = args.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path must be a non-empty string")

        start_line = int(args.get("start_line", 1))
        max_lines = int(args.get("max_lines", 200))
        max_chars = int(args.get("max_chars", 20_000))

        if start_line < 1:
            raise ToolError("start_line must be >= 1")
        if not (1 <= max_lines <= 2000):
            raise ToolError("max_lines must be between 1 and 2000")
        if not (1 <= max_chars <= 200_000):
            raise ToolError("max_chars must be between 1 and 200000")

        # Only allow paths relative to root.
        rel = Path(raw_path)
        if rel.is_absolute():
            raise ToolError("absolute paths are not allowed; pass a path relative to the repo root")

        path = (root / rel).resolve()
        try:
            if not path.is_relative_to(root):
                raise ToolError("path escapes base_dir")
        except AttributeError:
            # Python <3.9 fallback (not expected in this repo)
            if str(path).lower().find(str(root).lower()) != 0:
                raise ToolError("path escapes base_dir")

        if not path.exists() or not path.is_file():
            raise ToolError(f"file not found: {rel.as_posix()}")

        st = path.stat()
        end_line = start_line + max_lines - 1

        lines: list[str] = []
        total_chars = 0
        truncated = False
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            for i, line in enumerate(f, start=1):
                if i < start_line:
                    continue
                if i > end_line:
                    break
                if total_chars + len(line) > max_chars:
                    remaining = max_chars - total_chars
                    if remaining > 0:
                        lines.append(line[:remaining])
                    truncated = True
                    break
                lines.append(line)
                total_chars += len(line)

        text = "".join(lines)
        return json.dumps(
            {
                "ok": True,
                "path": str(rel).replace("\\", "/"),
                "start_line": start_line,
                "end_line": min(end_line, start_line + len(lines) - 1) if lines else start_line - 1,
                "file_size_bytes": int(st.st_size),
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                "truncated": truncated,
                "text": text,
            }
        )

    return _handler


def _resolve_repo_relative_path(*, base_dir: Path, raw_path: str) -> tuple[Path, Path]:
    root = base_dir.resolve()
    rel = Path(raw_path)
    if rel.is_absolute():
        raise ToolError("absolute paths are not allowed; pass a path relative to the repo root")
    abs_path = (root / rel).resolve()
    if not abs_path.exists():
        # keep error message in terms of relative path
        raise ToolError(f"file not found: {rel.as_posix()}")
    try:
        if not abs_path.is_relative_to(root):
            raise ToolError("path escapes base_dir")
    except AttributeError:
        if str(abs_path).lower().find(str(root).lower()) != 0:
            raise ToolError("path escapes base_dir")
    return root, abs_path


def _resolve_repo_relative_path_allow_missing(*, base_dir: Path, raw_path: str) -> tuple[Path, Path, Path]:
    root = base_dir.resolve()
    rel = Path(raw_path)
    if rel.is_absolute():
        raise ToolError("absolute paths are not allowed; pass a path relative to the repo root")
    abs_path = (root / rel).resolve()
    try:
        if not abs_path.is_relative_to(root):
            raise ToolError("path escapes base_dir")
    except AttributeError:
        if str(abs_path).lower().find(str(root).lower()) != 0:
            raise ToolError("path escapes base_dir")
    return root, rel, abs_path


def _is_allowed_relpath(*, rel: Path, allowed_globs: Sequence[str]) -> bool:
    posix = rel.as_posix().lstrip("./")
    for g in allowed_globs:
        if Path(posix).match(g):
            return True
    return False


def write_file_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "write_file",
        "description": "Write (create or overwrite) a UTF-8 text file (repo-scoped and allow-listed).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path (must be allow-listed)."},
                "content": {"type": "string", "description": "UTF-8 text content to write."},
                "max_chars": {"type": "integer", "description": "Refuse content larger than this (default: 200000)."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    }


def append_file_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "append_file",
        "description": "Append UTF-8 text to a file (repo-scoped and allow-listed). Creates the file if missing.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path (must be allow-listed)."},
                "content": {"type": "string", "description": "UTF-8 text content to append."},
                "ensure_newline": {
                    "type": "boolean",
                    "description": "If true, ensures content ends with a newline (default: true).",
                },
                "max_chars": {"type": "integer", "description": "Refuse content larger than this (default: 200000)."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    }


def make_write_file_handler(*, base_dir: Optional[Path] = None, allowed_globs: Optional[Sequence[str]] = None) -> FunctionHandler:
    root = (base_dir or Path.cwd()).resolve()
    allow = list(allowed_globs or ["memory.md", "chat_history/*.json", "chat_history/*.md"])

    def _handler(args: JsonDict) -> str:
        raw_path = args.get("path")
        content = args.get("content")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path must be a non-empty string")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        max_chars = int(args.get("max_chars", 200_000))
        if not (1 <= max_chars <= 2_000_000):
            raise ToolError("max_chars must be between 1 and 2000000")
        if len(content) > max_chars:
            raise ToolError(f"content too large ({len(content)} chars > {max_chars})")

        _, rel, abs_path = _resolve_repo_relative_path_allow_missing(base_dir=root, raw_path=raw_path)
        if not _is_allowed_relpath(rel=rel, allowed_globs=allow):
            raise ToolError(f"path not allow-listed: {rel.as_posix()}")

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        return json.dumps({"ok": True, "path": rel.as_posix(), "bytes": abs_path.stat().st_size})

    return _handler


def make_append_file_handler(
    *,
    base_dir: Optional[Path] = None,
    allowed_globs: Optional[Sequence[str]] = None,
) -> FunctionHandler:
    root = (base_dir or Path.cwd()).resolve()
    allow = list(allowed_globs or ["memory.md", "chat_history/*.json", "chat_history/*.md"])

    def _handler(args: JsonDict) -> str:
        raw_path = args.get("path")
        content = args.get("content")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path must be a non-empty string")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        ensure_newline = bool(args.get("ensure_newline", True))
        max_chars = int(args.get("max_chars", 200_000))
        if not (1 <= max_chars <= 2_000_000):
            raise ToolError("max_chars must be between 1 and 2000000")
        if len(content) > max_chars:
            raise ToolError(f"content too large ({len(content)} chars > {max_chars})")

        _, rel, abs_path = _resolve_repo_relative_path_allow_missing(base_dir=root, raw_path=raw_path)
        if not _is_allowed_relpath(rel=rel, allowed_globs=allow):
            raise ToolError(f"path not allow-listed: {rel.as_posix()}")

        if ensure_newline and content and not content.endswith("\n"):
            content_to_write = content + "\n"
        else:
            content_to_write = content

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        with abs_path.open("a", encoding="utf-8", newline="") as f:
            f.write(content_to_write)
        return json.dumps({"ok": True, "path": rel.as_posix(), "bytes": abs_path.stat().st_size})

    return _handler


def python_sandbox_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "python_sandbox",
        "description": (
            "Run a restricted Python script in an isolated working directory (sandbox) for data analysis. "
            "Uses the current venv (numpy/pandas/matplotlib available if installed). "
            "File access is restricted to inside the sandbox directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
                "input_files": {
                    "type": "object",
                    "description": "Optional mapping of relative file paths to UTF-8 text contents to create inside the sandbox.",
                    "additionalProperties": {"type": "string"},
                },
                "copy_from_repo": {
                    "type": "array",
                    "description": (
                        "Optional list of repo-relative file paths OR directories to copy into the sandbox before running. "
                        "Directories are copied recursively (size-limited)."
                    ),
                    "items": {"type": "string"},
                },
                "copy_globs": {
                    "type": "array",
                    "description": "Optional list of repo-relative glob patterns (e.g. 'logs/*.jsonl') to copy into the sandbox.",
                    "items": {"type": "string"},
                },
                "timeout_s": {"type": "number", "description": "Optional per-run timeout in seconds."},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    }


def render_plot_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "render_plot",
        "description": (
            "Run Python (via python_sandbox) to generate plots/graphs and return paths to created image files. "
            "Save figures using matplotlib (Agg) to PNG/SVG inside the sandbox; the UI can display them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute; should save one or more image files."},
                "copy_from_repo": {"type": "array", "items": {"type": "string"}},
                "copy_globs": {"type": "array", "items": {"type": "string"}},
                "timeout_s": {"type": "number"},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    }


def create_peer_agent_tool_spec() -> JsonDict:
    """Tool spec for creating a same-level peer agent in the Chat UI.

    Note: this tool is UI-owned (it requires an AgentHub). The handler is expected
    to enforce additional gating (e.g., only allow if the user explicitly asked).
    """

    return {
        "type": "function",
        "name": "create_peer_agent",
        "description": (
            "Create a new same-level peer agent (friend) in the current chat session. "
            "Use only when the user explicitly asked you to create/generate a new agent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique display name for the new agent."},
                "model": {"type": "string", "description": "Model id for the agent, e.g. gpt-4.1-mini."},
                "system_prompt": {"type": "string", "description": "System prompt for the new agent."},
                "memory_path": {
                    "type": ["string", "null"],
                    "description": "Optional repo-relative path to a memory markdown file.",
                },
            },
            "required": ["name", "model", "system_prompt"],
            "additionalProperties": False,
        },
    }


def make_create_peer_agent_handler(
    *,
    is_armed: Callable[[], bool],
    disarm: Callable[[], None],
    create_peer: Callable[[str, str, str, Optional[str]], JsonDict],
) -> FunctionHandler:
    """Create handler for `create_peer_agent` with UI-defined callbacks.

    Args:
        is_armed: Returns True if the user explicitly requested agent creation.
        disarm: Called after a successful creation attempt (best-effort).
        create_peer: Callback that actually creates the agent and returns a JSON-ish dict.
    """

    def _handler(args: JsonDict) -> str:
        if not is_armed():
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        "create_peer_agent is gated. Ask the user to explicitly say "
                        "'generate agent' or 'create an agent' in chat before calling this tool."
                    ),
                }
            )

        name = args.get("name")
        model = args.get("model")
        system_prompt = args.get("system_prompt")
        memory_path = args.get("memory_path", None)

        if not isinstance(name, str) or not name.strip():
            raise ToolError("name must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ToolError("model must be a non-empty string")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ToolError("system_prompt must be a non-empty string")
        if memory_path is not None and not isinstance(memory_path, str):
            raise ToolError("memory_path must be a string or null")

        try:
            out = create_peer(name.strip(), model.strip(), system_prompt.strip(), memory_path.strip() if isinstance(memory_path, str) else None)
        finally:
            # Always disarm after an attempt so the user must explicitly re-request.
            try:
                disarm()
            except Exception:
                pass

        return json.dumps({"ok": True, **(out or {})})

    return _handler


def make_render_plot_handler(*, python_sandbox_handler: FunctionHandler) -> FunctionHandler:
    """Thin wrapper around python_sandbox that highlights image outputs."""

    def handler(args: JsonDict) -> str:
        payload: JsonDict = {"code": args.get("code")}
        for k in ("copy_from_repo", "copy_globs", "timeout_s"):
            v = args.get(k)
            if v is not None:
                payload[k] = v

        out_s = python_sandbox_handler(payload)
        try:
            d = json.loads(out_s)
        except Exception:
            return out_s
        if not isinstance(d, dict):
            return out_s

        imgs = d.get("image_paths")
        if not isinstance(imgs, list):
            imgs = []
        d2 = dict(d)
        d2["image_paths"] = [x for x in imgs if isinstance(x, str) and x.strip()][:20]
        return json.dumps(d2, ensure_ascii=False)

    return handler


_PY_SANDBOX_BANNED_IMPORTS: set[str] = {
    "asyncio",
    "ctypes",
    "ftplib",
    "http",
    "importlib",
    "inspect",
    "multiprocessing",
    "os",
    "pickle",
    "pkgutil",
    "pty",
    "requests",
    "resource",
    "shlex",
    "signal",
    "site",
    "socket",
    "subprocess",
    "sys",
    "telnetlib",
    "threading",
    "types",
    "urllib",
    "webbrowser",
}


_PY_SANDBOX_BANNED_CALLS: set[str] = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "setattr",
    "vars",
}


def _validate_python_sandbox_code(code: str) -> list[str]:
    try:
        tree = ast.parse(code or "", mode="exec")
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    errors: list[str] = []

    class V(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
            for a in node.names:
                top = (a.name or "").split(".", 1)[0]
                if top in _PY_SANDBOX_BANNED_IMPORTS:
                    errors.append(f"import of '{top}' is not allowed")
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
            mod = node.module or ""
            top = mod.split(".", 1)[0] if mod else ""
            if top in _PY_SANDBOX_BANNED_IMPORTS:
                errors.append(f"import of '{top}' is not allowed")
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
            if isinstance(node.attr, str) and node.attr.startswith("__"):
                errors.append("dunder attribute access is not allowed")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _PY_SANDBOX_BANNED_CALLS:
                errors.append(f"call to '{fn.id}' is not allowed")
            self.generic_visit(node)

    V().visit(tree)
    return errors


def make_python_sandbox_handler(
    *,
    base_dir: Optional[Path] = None,
    timeout_s: float = 12.0,
    max_output_chars: int = 20000,
    max_copy_bytes: int = 5_000_000,
    max_copy_files: int = 2000,
) -> FunctionHandler:
    """Run Python in a restricted subprocess with sandboxed filesystem access.

    Uses the current interpreter (`sys.executable`), so it runs inside the same venv.
    This is not a perfect security boundary; treat it as "safer by default".
    """

    root = (base_dir or Path.cwd()).resolve()

    runner_src = """\
import builtins
import io
import os
import pathlib
import runpy
import sys

ROOT = pathlib.Path(os.environ["PY_SANDBOX_ROOT"]).resolve()
USER = ROOT / "user_code.py"
ALLOW_ABS = [pathlib.Path(p).resolve() for p in os.environ.get("PY_SANDBOX_ALLOW_ABS_PREFIXES", "").split(os.pathsep) if p]

_orig_open = builtins.open

def _resolve_write(p):
    p = pathlib.Path(p)
    if p.is_absolute():
        raise PermissionError("absolute paths are not allowed")
    rp = (ROOT / p).resolve()
    if not str(rp).startswith(str(ROOT)):
        raise PermissionError("path escapes sandbox")
    return rp

def _resolve_read(p):
    p = pathlib.Path(p)
    if p.is_absolute():
        rp = p.resolve()
        for ap in ALLOW_ABS:
            try:
                if str(rp).startswith(str(ap)):
                    return rp
            except Exception:
                pass
        raise PermissionError("absolute paths are not allowed")
    rp = (ROOT / p).resolve()
    if not str(rp).startswith(str(ROOT)):
        raise PermissionError("path escapes sandbox")
    return rp

def safe_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    m = str(mode or "r")
    is_write = any(x in m for x in ("w", "a", "+", "x"))
    rp = _resolve_write(file) if is_write else _resolve_read(file)
    return _orig_open(rp, mode, buffering, encoding=encoding, errors=errors, newline=newline, closefd=closefd, opener=opener)

builtins.open = safe_open
io.open = safe_open

try:
    pathlib.Path.open = lambda self, *a, **k: safe_open(str(self), *a, **k)  # type: ignore[assignment]
except Exception:
    pass

import os as _os
_orig_remove = _os.remove
_orig_unlink = _os.unlink
_orig_rename = _os.rename
_orig_replace = _os.replace
_orig_mkdir = _os.mkdir
_orig_makedirs = _os.makedirs
_orig_rmdir = _os.rmdir

def _wrap1(fn):
    def inner(p, *a, **k):
        rp = _resolve_write(p)
        return fn(str(rp), *a, **k)
    return inner

def _wrap2(fn):
    def inner(p1, p2, *a, **k):
        rp1 = _resolve_write(p1)
        rp2 = _resolve_write(p2)
        return fn(str(rp1), str(rp2), *a, **k)
    return inner

_os.remove = _wrap1(_orig_remove)
_os.unlink = _wrap1(_orig_unlink)
_os.rmdir = _wrap1(_orig_rmdir)
_os.mkdir = _wrap1(_orig_mkdir)
_os.makedirs = _wrap1(_orig_makedirs)
_os.rename = _wrap2(_orig_rename)
_os.replace = _wrap2(_orig_replace)

try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

os.chdir(ROOT)
runpy.run_path(str(USER), run_name="__main__")
"""

    def handler(args: JsonDict) -> str:
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ToolError("code must be a non-empty string")

        errs = _validate_python_sandbox_code(code)
        if errs:
            return json.dumps({"ok": False, "error": "UnsafeCode", "details": errs})

        t_s = args.get("timeout_s")
        try:
            tlim = float(t_s) if t_s is not None else float(timeout_s)
        except Exception:
            tlim = float(timeout_s)
        tlim = max(0.5, min(120.0, tlim))

        run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        run_dir = (root / "python_sandbox_runs" / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "runner.py").write_text(runner_src, encoding="utf-8")
        (run_dir / "user_code.py").write_text(code, encoding="utf-8")

        input_files = args.get("input_files") or {}
        if isinstance(input_files, dict):
            for rel, content in input_files.items():
                if not isinstance(rel, str) or not rel.strip():
                    continue
                if not isinstance(content, str):
                    continue
                p = Path(rel)
                if p.is_absolute():
                    continue
                dest = (run_dir / p).resolve()
                if not str(dest).startswith(str(run_dir)):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")

        copied: list[str] = []
        copied_bytes = 0
        copied_files = 0

        def _copy_path(rp: Path) -> None:
            nonlocal copied_bytes, copied_files
            if copied_files >= max_copy_files:
                return
            src = (root / rp).resolve()
            if not str(src).startswith(str(root)):
                return
            if not src.exists():
                return

            if src.is_dir():
                for fp in src.rglob("*"):
                    if copied_files >= max_copy_files:
                        return
                    if not fp.is_file():
                        continue
                    try:
                        sz = int(fp.stat().st_size)
                    except Exception:
                        continue
                    if sz <= 0:
                        continue
                    if sz > max_copy_bytes:
                        continue
                    if copied_bytes + sz > max_copy_bytes:
                        return
                    relp = fp.relative_to(root)
                    dest = (run_dir / relp).resolve()
                    if not str(dest).startswith(str(run_dir)):
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(fp.read_bytes())
                    copied.append(relp.as_posix())
                    copied_bytes += sz
                    copied_files += 1
                return

            if not src.is_file():
                return
            try:
                sz = int(src.stat().st_size)
            except Exception:
                return
            if sz <= 0:
                return
            if sz > max_copy_bytes:
                return
            if copied_bytes + sz > max_copy_bytes:
                return
            dest = (run_dir / rp).resolve()
            if not str(dest).startswith(str(run_dir)):
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
            copied.append(rp.as_posix())
            copied_bytes += sz
            copied_files += 1

        copy_from_repo = args.get("copy_from_repo") or []
        if isinstance(copy_from_repo, list):
            for rel in copy_from_repo:
                if not isinstance(rel, str) or not rel.strip():
                    continue
                rp = Path(rel)
                if rp.is_absolute():
                    continue
                _copy_path(rp)

        copy_globs = args.get("copy_globs") or []
        if isinstance(copy_globs, list):
            for pat in copy_globs:
                if copied_files >= max_copy_files:
                    break
                if not isinstance(pat, str) or not pat.strip():
                    continue
                gp = pat.replace("\\", "/").lstrip("./")
                # Only allow simple repo-relative patterns.
                for fp in root.glob(gp):
                    if copied_files >= max_copy_files:
                        break
                    if not fp.exists() or not fp.is_file():
                        continue
                    try:
                        relp = fp.relative_to(root)
                    except Exception:
                        continue
                    _copy_path(relp)

        env = os.environ.copy()
        env["PY_SANDBOX_ROOT"] = str(run_dir)
        # Allow read-only imports/resources from the running interpreter's environment (venv + stdlib).
        # Writes are still restricted to the sandbox directory.
        allow_abs = [str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve())]
        env["PY_SANDBOX_ALLOW_ABS_PREFIXES"] = os.pathsep.join(dict.fromkeys(allow_abs))
        env.setdefault("MPLBACKEND", "Agg")

        try:
            p = subprocess.run(
                [sys.executable, str(run_dir / "runner.py")],
                cwd=str(run_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=tlim,
            )
            stdout = p.stdout or ""
            stderr = p.stderr or ""
            exit_code = int(p.returncode)
            timed_out = False
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
            stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
            exit_code = -1
            timed_out = True

        def cap(s: str) -> str:
            if not isinstance(s, str):
                s = str(s)
            if len(s) > max_output_chars:
                return s[:max_output_chars] + "\\n…(truncated)…"
            return s

        created_files: list[str] = []
        try:
            for fp in run_dir.rglob("*"):
                if not fp.is_file():
                    continue
                relp = fp.relative_to(run_dir).as_posix()
                if relp in {"runner.py", "user_code.py"}:
                    continue
                created_files.append(relp)
        except Exception:
            pass

        image_exts = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")
        image_paths: list[str] = []
        try:
            run_dir_rel = str(run_dir.relative_to(root)).replace("\\", "/")
            for relp in created_files:
                if isinstance(relp, str) and relp.lower().endswith(image_exts):
                    image_paths.append(f"{run_dir_rel}/{relp}".replace("\\", "/"))
        except Exception:
            pass
        image_paths = image_paths[:20]

        return json.dumps(
            {
                "ok": True,
                "run_dir": str(run_dir.relative_to(root)).replace("\\", "/"),
                "copied_from_repo": copied,
                "copied_files": int(copied_files),
                "copied_bytes": int(copied_bytes),
                "exit_code": exit_code,
                "timed_out": timed_out,
                "stdout": cap(stdout),
                "stderr": cap(stderr),
                "created_files": created_files[:200],
                "image_paths": image_paths,
            }
        )

    return handler


def create_and_register_analysis_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "create_and_register_analysis_tool",
        "description": (
            "Create and register a reusable analysis tool backed by a python_sandbox script. "
            "Use this when you wrote a script you will likely reuse on future log analyses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "Name for the new tool (snake_case)."},
                "description": {"type": "string", "description": "What the tool does and when to use it."},
                "python_code": {"type": "string", "description": "Python script body to run in python_sandbox."},
            },
            "required": ["tool_name", "description", "python_code"],
            "additionalProperties": False,
        },
    }


def make_create_and_register_analysis_tool_handler(
    *,
    registry: ToolRegistry,
    python_sandbox_handler: FunctionHandler,
    scripts_root: Path,
) -> FunctionHandler:
    root = scripts_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def handler(args: JsonDict) -> str:
        tool_name = _validate_tool_name(str(args.get("tool_name") or ""))
        desc = args.get("description")
        code = args.get("python_code")
        if not isinstance(desc, str) or not desc.strip():
            raise ToolError("description must be a non-empty string")
        if not isinstance(code, str) or not code.strip():
            raise ToolError("python_code must be a non-empty string")

        errs = _validate_python_sandbox_code(code)
        if errs:
            return json.dumps({"ok": False, "error": "UnsafeCode", "details": errs})

        ts = _utc_ts_compact()
        out_dir = (root / tool_name / ts).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        script_path = out_dir / f"{tool_name}.py"
        meta_path = out_dir / "tool.json"

        script_path.write_text(code, encoding="utf-8")
        meta_path.write_text(
            json.dumps({"tool_name": tool_name, "description": desc, "created_at": ts}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Register a tool that runs this script via python_sandbox.
        tool_spec: JsonDict = {
            "type": "function",
            "name": tool_name,
            "description": str(desc),
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {"type": "object", "description": "Arbitrary JSON passed to the script as `args`."},
                    "copy_from_repo": {"type": "array", "items": {"type": "string"}},
                    "copy_globs": {"type": "array", "items": {"type": "string"}},
                    "timeout_s": {"type": "number"},
                },
                "additionalProperties": False,
            },
        }

        def run_tool(call_args: JsonDict, *, _script=script_path) -> str:
            tool_args = call_args.get("args")
            if tool_args is None:
                tool_args = {}
            if not isinstance(tool_args, dict):
                raise ToolError("args must be an object")

            input_files = {"tool_args.json": json.dumps(tool_args, ensure_ascii=False)}
            payload: JsonDict = {
                "code": (
                    "import json\n"
                    "from pathlib import Path\n"
                    "args = json.loads(Path('tool_args.json').read_text(encoding='utf-8'))\n"
                    f"# --- user script: {_script.name} ---\n"
                    + _script.read_text(encoding="utf-8", errors="replace")
                ),
                "input_files": input_files,
            }
            for k in ("copy_from_repo", "copy_globs", "timeout_s"):
                v = call_args.get(k)
                if v is not None:
                    payload[k] = v
            return python_sandbox_handler(payload)

        # Analysis tools are meant to be iterated on; allow overwriting an existing tool_name
        # within the same session (e.g., after loading saved tools at startup).
        registry.tools[tool_name] = tool_spec
        registry.handlers[tool_name] = run_tool
        return json.dumps({"ok": True, "tool_name": tool_name, "saved_to": str(out_dir)})

    return handler


def _load_saved_analysis_tools(
    *,
    scripts_root: Path,
) -> list[tuple[str, str, Path]]:
    """Return a list of (tool_name, description, script_path) for the newest saved version of each tool.

    Expected on-disk layout (created by create_and_register_analysis_tool):
      scripts_root/<tool_name>/<timestamp>/<tool_name>.py
      scripts_root/<tool_name>/<timestamp>/tool.json
    """

    root = Path(scripts_root).resolve()
    if not root.exists() or not root.is_dir():
        return []

    latest: dict[str, tuple[str, str, Path]] = {}
    for tool_dir in root.iterdir():
        if not tool_dir.is_dir():
            continue
        tool_name = tool_dir.name
        try:
            _validate_tool_name(tool_name)
        except Exception:
            continue

        # Pick the newest timestamp directory (lexicographic is fine with UTC ts format).
        versions = [p for p in tool_dir.iterdir() if p.is_dir()]
        if not versions:
            continue
        versions.sort(key=lambda p: p.name, reverse=True)
        vdir = versions[0]
        meta = vdir / "tool.json"
        script = vdir / f"{tool_name}.py"
        if not meta.exists() or not script.exists():
            continue

        try:
            md = json.loads(meta.read_text(encoding="utf-8"))
            if not isinstance(md, dict):
                continue
            desc = md.get("description")
            if not isinstance(desc, str) or not desc.strip():
                continue
        except Exception:
            continue

        latest[tool_name] = (tool_name, desc.strip(), script)

    return list(latest.values())


def list_saved_analysis_tool_names(*, scripts_root: Path) -> list[str]:
    """List saved analysis tool names on disk (newest version per tool_name)."""

    names = [t[0] for t in _load_saved_analysis_tools(scripts_root=scripts_root)]
    names.sort()
    return names


def register_saved_analysis_tools(
    *,
    registry: ToolRegistry,
    python_sandbox_handler: FunctionHandler,
    scripts_root: Path,
) -> list[str]:
    """Register previously saved analysis tools into the provided registry.

    Returns the list of tool names registered (newest version per tool_name).
    """

    loaded = _load_saved_analysis_tools(scripts_root=scripts_root)
    names: list[str] = []

    for tool_name, desc, script_path in loaded:
        # Register a tool that runs this script via python_sandbox.
        tool_spec: JsonDict = {
            "type": "function",
            "name": tool_name,
            "description": str(desc),
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {"type": "object", "description": "Arbitrary JSON passed to the script as `args`."},
                    "copy_from_repo": {"type": "array", "items": {"type": "string"}},
                    "copy_globs": {"type": "array", "items": {"type": "string"}},
                    "timeout_s": {"type": "number"},
                },
                "additionalProperties": False,
            },
        }

        def run_tool(call_args: JsonDict, *, _script=script_path) -> str:
            tool_args = call_args.get("args")
            if tool_args is None:
                tool_args = {}
            if not isinstance(tool_args, dict):
                raise ToolError("args must be an object")

            input_files = {"tool_args.json": json.dumps(tool_args, ensure_ascii=False)}
            payload: JsonDict = {
                "code": (
                    "import json\n"
                    "from pathlib import Path\n"
                    "args = json.loads(Path('tool_args.json').read_text(encoding='utf-8'))\n"
                    f"# --- user script: {_script.name} ---\n"
                    + _script.read_text(encoding="utf-8", errors="replace")
                ),
                "input_files": input_files,
            }
            for k in ("copy_from_repo", "copy_globs", "timeout_s"):
                v = call_args.get(k)
                if v is not None:
                    payload[k] = v
            return python_sandbox_handler(payload)

        # Overwrite if already present (e.g., if user loaded tools earlier in the session).
        registry.tools[tool_name] = tool_spec
        registry.handlers[tool_name] = run_tool
        names.append(tool_name)

    names.sort()
    return names


def set_system_prompt_tool_spec() -> JsonDict:
    """Tool spec to update the live system prompt without resetting the conversation by default."""

    return {
        "type": "function",
        "name": "set_system_prompt",
        "description": (
            "Update the system prompt used for subsequent responses without resetting the conversation by default. "
            "Optionally reset history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "system_prompt": {"type": "string", "description": "The new system prompt to use."},
                "reset_history": {
                    "type": "boolean",
                    "description": "If true, clears prior chat history.",
                    "default": False,
                },
                "note": {"type": "string", "description": "Optional short note (for UI/logging)."},
            },
            "required": ["system_prompt"],
            "additionalProperties": False,
        },
    }


def make_set_system_prompt_handler(
    *,
    set_prompt: Callable[[str], None],
    reset_history: Callable[[bool], None],
) -> FunctionHandler:
    """Create a handler that updates in-memory state owned by the UI/session."""

    def _handler(args: JsonDict) -> str:
        sp = args.get("system_prompt")
        if not isinstance(sp, str) or not sp.strip():
            raise ToolError("system_prompt must be a non-empty string")
        rh = bool(args.get("reset_history", False))
        set_prompt(sp)
        reset_history(rh)
        note = args.get("note")
        return json.dumps({"ok": True, "reset_history": rh, "note": note if isinstance(note, str) else ""})

    return _handler


def get_system_prompt_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "get_system_prompt",
        "description": "Return the current system prompt used for subsequent responses.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }


def make_get_system_prompt_handler(*, get_prompt: Callable[[], str]) -> FunctionHandler:
    def _handler(args: JsonDict) -> str:  # noqa: ARG001
        return json.dumps({"ok": True, "system_prompt": str(get_prompt())})

    return _handler


def add_to_system_prompt_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "add_to_system_prompt",
        "description": (
            "Append text to the current system prompt for subsequent responses. "
            "Optionally reset history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to append to the system prompt."},
                "separator": {
                    "type": "string",
                    "description": "Separator inserted between old prompt and appended text (default: \\n\\n).",
                },
                "reset_history": {
                    "type": "boolean",
                    "description": "If true, clears prior chat history.",
                    "default": False,
                },
                "note": {"type": "string", "description": "Optional short note (for UI/logging)."},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    }


def make_add_to_system_prompt_handler(
    *,
    get_prompt: Callable[[], str],
    set_prompt: Callable[[str], None],
    reset_history: Callable[[bool], None],
) -> FunctionHandler:
    def _handler(args: JsonDict) -> str:
        extra = args.get("text")
        if not isinstance(extra, str) or not extra.strip():
            raise ToolError("text must be a non-empty string")
        sep = args.get("separator", "\n\n")
        if not isinstance(sep, str):
            sep = "\n\n"
        rh = bool(args.get("reset_history", False))

        cur = str(get_prompt() or "")
        new_prompt = (cur + (sep if cur and extra else "") + extra).strip()
        set_prompt(new_prompt)
        reset_history(rh)
        note = args.get("note")
        return json.dumps({"ok": True, "reset_history": rh, "note": note if isinstance(note, str) else ""})

    return _handler


# ---- Submodels / parallelization tools ----


def submodel_batch_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "submodel_batch",
        "description": (
            "Create and/or run multiple sub-model instances in parallel. "
            "Each submodel has its own system prompt and conversation context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "submodel_id": {
                                "type": "string",
                                "description": "If provided, reuse an existing submodel; otherwise create a new one.",
                            },
                            "title": {"type": "string"},
                            "system_prompt": {"type": "string", "description": "Used when creating a new submodel."},
                            "input": {"type": "string", "description": "User input sent to the submodel."},
                            "model": {"type": "string", "description": "Optional model override for the submodel."},
                            "keep": {
                                "type": "boolean",
                                "description": "If false, terminate the submodel after producing output (default: true).",
                            },
                            "allow_nested": {
                                "type": "boolean",
                                "description": "If true, allow this submodel to spawn its own submodels (bounded).",
                            },
                        },
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                },
                "parallel": {"type": "boolean", "description": "Run tasks concurrently (default: true)."},
            },
            "required": ["tasks"],
            "additionalProperties": False,
        },
    }


def submodel_list_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "submodel_list",
        "description": "List active submodels (id/title/model).",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }


def submodel_close_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "submodel_close",
        "description": "Terminate a submodel instance by id.",
        "parameters": {
            "type": "object",
            "properties": {"submodel_id": {"type": "string"}},
            "required": ["submodel_id"],
            "additionalProperties": False,
        },
    }

# ---- Dynamic tool creation (codegen) ----


_ALLOWED_IMPORTS: set[str] = {
    "json",
    "math",
    "re",
    "statistics",
    "datetime",
    "decimal",
    "fractions",
    "itertools",
    "functools",
    "collections",
    "typing",
}

_FORBIDDEN_BUILTINS: set[str] = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
}


def _static_check_python_tool(code: str) -> None:
    if not isinstance(code, str) or not code.strip():
        raise ToolError("python_code must be a non-empty string")
    if len(code.encode("utf-8")) > 40_000:
        raise ToolError("python_code too large (limit 40KB)")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ToolError(f"python_code syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = (alias.name or "").split(".", 1)[0]
                if mod not in _ALLOWED_IMPORTS:
                    raise ToolError(f"Disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".", 1)[0]
            if mod not in _ALLOWED_IMPORTS:
                raise ToolError(f"Disallowed import-from: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_BUILTINS:
                raise ToolError(f"Disallowed builtin call: {node.func.id}")


def _verify_in_subprocess(*, module_path: Path, func_name: str, tests: list[JsonDict], timeout_s: float) -> JsonDict:
    runner = r"""
import importlib.util, json, sys, traceback

module_path = sys.argv[1]
func_name = sys.argv[2]
tests = json.loads(sys.argv[3])

def die(msg):
    print(json.dumps({"ok": False, "error": msg}))
    sys.exit(0)

spec = importlib.util.spec_from_file_location("generated_tool", module_path)
if spec is None or spec.loader is None:
    die("failed to load module spec")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    die("import failed: " + repr(e))

fn = getattr(mod, func_name, None)
if fn is None or not callable(fn):
    die("missing callable: " + func_name)

results = []
for i, t in enumerate(tests):
    args = t.get("args", {})
    if not isinstance(args, dict):
        results.append({"i": i, "ok": False, "error": "args must be an object"})
        continue
    try:
        out = fn(args)
        if not isinstance(out, str):
            results.append({"i": i, "ok": False, "error": "handler must return str"})
            continue
        exp = t.get("expect_contains")
        if exp is not None and isinstance(exp, str) and exp not in out:
            results.append({"i": i, "ok": False, "error": "missing expect_contains"})
            continue
        results.append({"i": i, "ok": True, "output": out[:2000]})
    except Exception as e:
        results.append({"i": i, "ok": False, "error": repr(e)})

ok = all(r.get("ok") for r in results) if results else True
print(json.dumps({"ok": ok, "results": results}))
"""

    if not module_path.exists():
        raise ToolError(f"module_path not found: {module_path}")
    if not isinstance(tests, list):
        raise ToolError("tests must be a list")

    payload = json.dumps(tests, ensure_ascii=False)
    cmd = [sys.executable, "-I", "-c", runner, str(module_path), func_name, payload]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)

    out = (proc.stdout or "").strip()
    if not out:
        return {"ok": False, "error": "verifier produced no output", "stderr": proc.stderr[-4000:]}
    try:
        report = json.loads(out.splitlines()[-1])
    except json.JSONDecodeError:
        return {"ok": False, "error": "verifier output not JSON", "stdout": out[-4000:], "stderr": proc.stderr[-4000:]}
    if not isinstance(report, dict):
        return {"ok": False, "error": "verifier report not an object"}
    report["returncode"] = proc.returncode
    if proc.stderr:
        report["stderr_tail"] = proc.stderr[-2000:]
    return report


def create_and_register_python_tool_spec() -> JsonDict:
    """Tool spec allowing the model to propose Python tool code + quick tests."""

    return {
        "type": "function",
        "name": "create_and_register_python_tool",
        "description": (
            "Create a new Python-backed function tool, verify it with self-tests, and register it for immediate use "
            "in subsequent tool-calling rounds. Tool code is restricted (imports limited; dangerous builtins blocked)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "description": {"type": "string"},
                "parameters_schema": {"type": "object"},
                "python_code": {"type": "string", "description": "Python module source; must define def <tool_name>(args: dict)->str"},
                "tests": {
                    "type": "array",
                    "description": "Self-tests run by verifier.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "args": {"type": "object"},
                            "expect_contains": {"type": "string", "description": "Optional substring expected in output"},
                        },
                        "required": ["args"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["tool_name", "description", "parameters_schema", "python_code", "tests"],
            "additionalProperties": False,
        },
    }


def make_create_and_register_python_tool_handler(
    *,
    registry: ToolRegistry,
    base_dir: Optional[Path] = None,
    verify_timeout_s: float = 5.0,
) -> FunctionHandler:
    """Factory for a handler that mutates `registry` to add the created tool."""

    out_root = base_dir or (Path.cwd() / "generated_tools")

    def _handler(args: JsonDict) -> str:
        name = _validate_tool_name(str(args.get("tool_name", "")))
        description = str(args.get("description", "")).strip()
        params = args.get("parameters_schema")
        code = args.get("python_code")
        tests = args.get("tests")

        if not description:
            raise ToolError("description must be non-empty")
        if not isinstance(params, dict) or params.get("type") != "object":
            raise ToolError("parameters_schema must be a JSON schema object with type='object'")
        if not isinstance(code, str):
            raise ToolError("python_code must be a string")
        if not isinstance(tests, list):
            raise ToolError("tests must be a list")

        _static_check_python_tool(code)

        out_dir = out_root / name / _utc_ts_compact()
        mod_path = out_dir / f"{name}.py"
        _safe_write_text(mod_path, code if code.endswith("\n") else code + "\n")

        report = _verify_in_subprocess(module_path=mod_path, func_name=name, tests=tests, timeout_s=verify_timeout_s)
        if not report.get("ok"):
            return json.dumps({"ok": False, "error": "verification_failed", "report": report, "module_path": str(mod_path)})

        tool_spec: JsonDict = {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": params,
        }

        def _runtime_handler(call_args: JsonDict) -> str:
            # Execute in a subprocess each time to avoid importing arbitrary code into the main process.
            single = [{"args": call_args}]
            rep = _verify_in_subprocess(module_path=mod_path, func_name=name, tests=single, timeout_s=verify_timeout_s)
            if not rep.get("ok"):
                return json.dumps({"ok": False, "error": "tool_runtime_failed", "report": rep})
            res0 = (rep.get("results") or [{}])[0]
            out = res0.get("output")
            return out if isinstance(out, str) else json.dumps({"ok": False, "error": "bad_tool_output"})

        registry.add(tool_spec=tool_spec, handler=_runtime_handler)

        return json.dumps(
            {
                "ok": True,
                "tool_name": name,
                "tool_spec": tool_spec,
                "module_path": str(mod_path),
                "verification_report": report,
            }
        )

    return _handler


# ---- Dev workspace tools (Python-only venv + pip + web preview) ----


def workspace_info_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "workspace_info",
        "description": "Get the current dev workspace root for this chat and whether its venv exists.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    }


def workspace_write_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "workspace_write",
        "description": "Write a UTF-8 text file inside the dev workspace (workspace-relative path only).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path, e.g. index.html or src/app.py"},
                "content": {"type": "string", "description": "Full file contents (UTF-8)."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    }


def workspace_append_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "workspace_append",
        "description": "Append UTF-8 text to a file inside the dev workspace (workspace-relative path only).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path."},
                "text": {"type": "string", "description": "Text to append."},
            },
            "required": ["path", "text"],
            "additionalProperties": False,
        },
    }


def workspace_read_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "workspace_read",
        "description": "Read a UTF-8 text file inside the dev workspace (workspace-relative path only).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "default": 1},
                "max_lines": {"type": "integer", "default": 200},
                "max_chars": {"type": "integer", "default": 20000},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    }


def workspace_list_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "workspace_list",
        "description": "List files under a workspace directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "dir": {"type": "string", "default": ".", "description": "Workspace-relative directory, default '.'"},
                "max_entries": {"type": "integer", "default": 200},
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def workspace_create_venv_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "workspace_create_venv",
        "description": "Create a Python venv at workspace/.venv (no-op if it already exists).",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    }


def workspace_pip_install_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "workspace_pip_install",
        "description": "Install Python packages into the workspace venv using pip (requires user approval).",
        "parameters": {
            "type": "object",
            "properties": {
                "packages": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['numpy==2.2.2']"},
                "timeout_s": {"type": "number", "default": 900},
            },
            "required": ["packages"],
            "additionalProperties": False,
        },
    }


def workspace_run_python_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "workspace_run_python",
        "description": "Run a Python script using the workspace venv interpreter.",
        "parameters": {
            "type": "object",
            "properties": {
                "script_path": {"type": "string", "description": "Workspace-relative script path."},
                "args": {"type": "array", "items": {"type": "string"}, "default": []},
                "timeout_s": {"type": "number", "default": 60},
            },
            "required": ["script_path"],
            "additionalProperties": False,
        },
    }


def workspace_http_server_start_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "workspace_http_server_start",
        "description": "Start a static HTTP server for the workspace (python -m http.server).",
        "parameters": {
            "type": "object",
            "properties": {
                "root": {"type": "string", "default": ".", "description": "Workspace-relative dir to serve."},
                "port": {"type": "integer", "default": 8000},
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def workspace_http_server_stop_tool_spec() -> JsonDict:
    return {
        "type": "function",
        "name": "workspace_http_server_stop",
        "description": "Stop the workspace static HTTP server started previously.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    }


def make_workspace_handlers(
    *,
    repo_root: Path,
    get_chat_id: Callable[[], str],
    allow_workspace: Callable[[], bool],
    allow_pip: Callable[[], bool],
    allow_http_server: Callable[[], bool],
    approve_pip: Callable[[str], bool],
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, FunctionHandler]:
    def _ws() -> Path:
        return ws_ensure_workspace(repo_root=repo_root, chat_id=get_chat_id())

    def _log(msg: str) -> None:
        try:
            if log is not None:
                log(msg)
        except Exception:
            pass

    def _disabled(name: str) -> str:
        return json.dumps({"ok": False, "error": f"{name} disabled"})

    def info(_: JsonDict) -> str:
        if not allow_workspace():
            return _disabled("workspace")
        ws = _ws()
        return json.dumps(
            {
                "ok": True,
                "workspace_root": str(ws),
                "venv_exists": bool(ws_venv_exists(ws)),
                "venv_python": str(ws_venv_python_path(ws)),
            }
        )

    def write(args: JsonDict) -> str:
        if not allow_workspace():
            return _disabled("workspace")
        try:
            p = ws_write_text(ws_root=_ws(), rel_path=str(args.get("path") or ""), content=str(args.get("content") or ""))
            return json.dumps({"ok": True, "path": str(p)})
        except WorkspaceError as e:
            return json.dumps({"ok": False, "error": f"{e}"})

    def append(args: JsonDict) -> str:
        if not allow_workspace():
            return _disabled("workspace")
        try:
            p = ws_append_text(ws_root=_ws(), rel_path=str(args.get("path") or ""), text=str(args.get("text") or ""))
            return json.dumps({"ok": True, "path": str(p)})
        except WorkspaceError as e:
            return json.dumps({"ok": False, "error": f"{e}"})

    def read(args: JsonDict) -> str:
        if not allow_workspace():
            return _disabled("workspace")
        try:
            txt = ws_read_text(
                ws_root=_ws(),
                rel_path=str(args.get("path") or ""),
                start_line=int(args.get("start_line") or 1),
                max_lines=int(args.get("max_lines") or 200),
                max_chars=int(args.get("max_chars") or 20000),
            )
            return json.dumps({"ok": True, "text": txt})
        except WorkspaceError as e:
            return json.dumps({"ok": False, "error": f"{e}"})

    def ls(args: JsonDict) -> str:
        if not allow_workspace():
            return _disabled("workspace")
        try:
            items = ws_list_dir(ws_root=_ws(), rel_dir=str(args.get("dir") or "."), max_entries=int(args.get("max_entries") or 200))
            return json.dumps({"ok": True, "items": items})
        except WorkspaceError as e:
            return json.dumps({"ok": False, "error": f"{e}"})

    def mkvenv(_: JsonDict) -> str:
        if not allow_workspace():
            return _disabled("workspace")
        try:
            py = ws_create_venv(ws_root=_ws())
            return json.dumps({"ok": True, "venv_python": str(py)})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})

    def pip_install(args: JsonDict) -> str:
        if not allow_workspace():
            return _disabled("workspace")
        if not allow_pip():
            return json.dumps({"ok": False, "error": "pip installs disabled"})
        packages = args.get("packages")
        if not isinstance(packages, list) or not packages:
            return json.dumps({"ok": False, "error": "packages must be a non-empty array"})
        pkgs = [str(x) for x in packages if str(x).strip()]
        prompt = "Allow pip install into dev workspace venv?\n\n" + "\n".join(f"- {p}" for p in pkgs)
        if not bool(approve_pip(prompt)):
            return json.dumps({"ok": False, "error": "User did not approve pip install"})
        timeout_s = float(args.get("timeout_s") or 900.0)
        ws = _ws()
        _log(f"pip_install_start packages={pkgs}")
        try:
            cp = ws_pip_install(ws_root=ws, packages=pkgs, timeout_s=timeout_s, log_path=(ws / "pip_install.log"))
            out = (cp.stdout or "").strip()
            if len(out) > 12000:
                out = out[-12000:]
            _log(f"pip_install_done code={cp.returncode}")
            return json.dumps(
                {
                    "ok": cp.returncode == 0,
                    "returncode": int(cp.returncode),
                    "output": out,
                    "log_path": str((ws / "pip_install.log").resolve()),
                }
            )
        except Exception as e:  # noqa: BLE001
            _log(f"pip_install_error error={type(e).__name__}: {e}")
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})

    def run_py(args: JsonDict) -> str:
        if not allow_workspace():
            return _disabled("workspace")
        script_path = str(args.get("script_path") or "")
        if not script_path:
            return json.dumps({"ok": False, "error": "script_path required"})
        argv = args.get("args")
        if argv is None:
            argv = []
        if not isinstance(argv, list):
            return json.dumps({"ok": False, "error": "args must be an array"})
        timeout_s = float(args.get("timeout_s") or 60.0)
        ws = _ws()
        try:
            # Ensure script exists inside workspace.
            rp = (ws / Path(script_path)).resolve()
            if not str(rp).startswith(str(ws)) or (not rp.exists()):
                return json.dumps({"ok": False, "error": "script_path must exist inside workspace"})
            cp = ws_run_venv_python(ws_root=ws, args=[str(rp), *[str(x) for x in argv]], timeout_s=timeout_s)
            out = (cp.stdout or "").strip()
            if len(out) > 12000:
                out = out[-12000:]
            return json.dumps({"ok": cp.returncode == 0, "returncode": int(cp.returncode), "output": out})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})

    def server_start(args: JsonDict) -> str:
        if not allow_workspace():
            return _disabled("workspace")
        if not allow_http_server():
            return json.dumps({"ok": False, "error": "http server disabled"})
        ws = _ws()
        try:
            root = str(args.get("root") or ".")
            port = int(args.get("port") or 8000)
            st = ws_http_server_start(ws_root=ws, port=port, root=root, use_venv=True)
            return json.dumps(
                {"ok": True, "port": int(st.port), "pid": int(st.pid), "url": f"http://127.0.0.1:{int(st.port)}/"}
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})

    def server_stop(_: JsonDict) -> str:
        if not allow_workspace():
            return _disabled("workspace")
        if not allow_http_server():
            return json.dumps({"ok": False, "error": "http server disabled"})
        try:
            ok = bool(ws_http_server_stop(ws_root=_ws()))
            return json.dumps({"ok": True, "stopped": ok})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})

    return {
        "workspace_info": info,
        "workspace_write": write,
        "workspace_append": append,
        "workspace_read": read,
        "workspace_list": ls,
        "workspace_create_venv": mkvenv,
        "workspace_pip_install": pip_install,
        "workspace_run_python": run_py,
        "workspace_http_server_start": server_start,
        "workspace_http_server_stop": server_stop,
    }
