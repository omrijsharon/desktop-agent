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
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


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

        if not any_calls:
            return (resp, input_list) if return_input_items else resp

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
