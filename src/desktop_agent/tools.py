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
import json
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


JsonDict = dict[str, Any]
FunctionHandler = Callable[[JsonDict], str]


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

    for _ in range(max_rounds):
        if registry is not None:
            round_tools = registry.tool_list() + list(extra_tools or [])
        else:
            round_tools = list(tools or [])
        round_handlers = registry.handlers if registry is not None else (handlers or {})

        cur_instructions = instructions_provider() if instructions_provider is not None else instructions
        resp = client.responses.create(
            model=model,
            input=input_list,
            tools=list(round_tools),
            instructions=cur_instructions,
            **create_kwargs,
        )

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
    last_resp: Any = None

    for _ in range(max_rounds):
        if registry is not None:
            round_tools = registry.tool_list() + list(extra_tools or [])
        else:
            round_tools = list(tools or [])
        round_handlers = registry.handlers if registry is not None else (handlers or {})

        cur_instructions = instructions_provider() if instructions_provider is not None else instructions

        stream = client.responses.create(
            model=model,
            input=input_list,
            tools=list(round_tools),
            instructions=cur_instructions,
            stream=True,
            **create_kwargs,
        )

        completed_resp: Any = None

        for evt in stream:
            ed = _as_event_dict(evt)
            etype = str(ed.get("type") or "")

            # Best-effort text streaming: handle common delta shapes.
            delta = ed.get("delta")
            if isinstance(delta, str) and ("output_text" in etype or etype.endswith(".delta") or "text" in etype):
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

ROOT = pathlib.Path(os.environ["PY_SANDBOX_ROOT"]).resolve()
USER = ROOT / "user_code.py"

_orig_open = builtins.open

def _resolve(p):
    p = pathlib.Path(p)
    if p.is_absolute():
        raise PermissionError("absolute paths are not allowed")
    rp = (ROOT / p).resolve()
    if not str(rp).startswith(str(ROOT)):
        raise PermissionError("path escapes sandbox")
    return rp

def safe_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    rp = _resolve(file)
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
        rp = _resolve(p)
        return fn(str(rp), *a, **k)
    return inner

def _wrap2(fn):
    def inner(p1, p2, *a, **k):
        rp1 = _resolve(p1)
        rp2 = _resolve(p2)
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

        registry.add(tool_spec=tool_spec, handler=run_tool)
        return json.dumps({"ok": True, "tool_name": tool_name, "saved_to": str(out_dir)})

    return handler


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
