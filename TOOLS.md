# Tools in this repo

This repo uses OpenAI’s Responses API “tools” mechanism to let a model request that *your program* run some function and return the result back to the model.

The important point: **the model does not execute code directly**. It emits tool-call “messages”, and the application decides what to run and what data to return.

## Concepts

### Tool specs (what the model can call)

A tool is described by a JSON object like:

- `{"type": "web_search"}` (built-in tool type)
- `{"type": "function", "name": "...", "description": "...", "parameters": { ...json schema... }}` (your function tool)

In this repo, function-tool specs are represented as plain `dict`s (`JsonDict`) in `src/desktop_agent/tools.py`.

### Handlers (what your app runs)

When the model outputs a `function_call` item, the app:

1. Looks up the `name`
2. Parses `arguments` (JSON string) into a dict
3. Calls a Python handler: `handler(args_dict) -> str`
4. Appends a `function_call_output` item containing the handler’s returned string
5. Calls the model again with the updated conversation

## Where this is implemented

### Core runner

- `src/desktop_agent/tools.py`:
  - `run_responses_with_function_tools(...)`: minimal loop to run `responses.create(...)` and satisfy `function_call`s.
  - `ToolRegistry`: a mutable `{tools, handlers}` registry. This is what makes “dynamic tools” possible.

You can pass either:

- A fixed `tools=[...]` and `handlers={...}` mapping, or
- A live `registry=ToolRegistry(...)` so tools can be added while the loop is running.

### Built-in tools shipped with the repo

In `src/desktop_agent/tools.py`:

- `read_file`:
  - Spec: `read_file_tool_spec()`
  - Handler factory: `make_read_file_handler(base_dir=...)`
  - Safety: restricted to the configured base directory (by default, repo root / CWD) and capped by `max_lines` + `max_chars`.
- `write_file` / `append_file`:
  - Specs: `write_file_tool_spec()`, `append_file_tool_spec()`
  - Handler factories: `make_write_file_handler(...)`, `make_append_file_handler(...)`
  - Safety: repo-scoped + allow-listed paths (defaults include `memory.md` and `chat_history/*`), and content size limits.
- `set_system_prompt`:
  - Spec: `set_system_prompt_tool_spec()`
  - Handler factory: `make_set_system_prompt_handler(...)`
  - Purpose: lets the model update the live system prompt for future turns without resetting the conversation by default.
- `get_system_prompt`:
  - Spec: `get_system_prompt_tool_spec()`
  - Handler factory: `make_get_system_prompt_handler(...)`
  - Purpose: lets the model retrieve the current system prompt.
- `add_to_system_prompt`:
  - Spec: `add_to_system_prompt_tool_spec()`
  - Handler factory: `make_add_to_system_prompt_handler(...)`
  - Purpose: appends text to the system prompt instead of replacing it.

### “Self-extending tools” (model creates a new tool)

There are two related mechanisms:

1) **Proposal-only (safe by default)**
- Tool: `propose_function_tool`
- Behavior: writes a proposal folder under `tool_proposals/<tool_name>/<timestamp>/` containing:
  - `tool.json` (the tool spec)
  - `<tool_name>.py` (a stub handler)
- This does *not* register or execute the new tool automatically.

2) **Create + verify + register (dynamic)**
- Tool: `create_and_register_python_tool`
- Behavior:
  1. Receives: `tool_name`, tool schema, `python_code`, and `tests`
  2. Runs a static safety check on the code (allow-listed imports; blocks some dangerous builtins)
  3. Writes the module under `generated_tools/<tool_name>/<timestamp>/<tool_name>.py`
  4. Runs the tests in a subprocess (`python -I ...`) via an internal verifier
  5. If tests pass, registers the new tool into the live `ToolRegistry`

Once registered, subsequent rounds in the same running loop include the new tool in the model’s `tools=[...]`.

Important safety design choice:
- Generated tool code is executed in a **subprocess per call**, not imported into the main agent process.

### Built-in web search

The OpenAI `web_search` tool is a built-in tool type. When enabled, you include it in the `tools=[...]` list (no local handler required). The platform runs the search and provides results back to the model as tool output items.

### Built-in file search (vector stores)

The OpenAI `file_search` tool is also a built-in tool type. When enabled, it is included in the `tools=[...]` list with your `vector_store_ids`. The platform performs retrieval against those stores and provides results back to the model as tool output items.

Optional: you can request the tool to include results payloads in the response by setting `include=["file_search_call.results"]` on the Responses API call (this can increase token usage).

### Local python sandbox (data analysis)

This repo also ships a local function tool:

- `python_sandbox`: runs user-provided Python in a subprocess using the current venv (so `numpy`, `pandas`, `matplotlib` work if installed).

Safety notes (best-effort, not bulletproof):
- Static validation blocks obvious dangerous imports/calls (e.g. `os`, `subprocess`, `socket`, `eval`, `exec`).
- Runtime guards restrict file access to a per-run sandbox directory under `python_sandbox_runs/`.
- Use the UI Controls to enable/disable it and set a timeout.

### Reusable analysis tools (python sandbox scripts)

In the calibration analysis window, the model can create reusable tools via:

- `create_and_register_analysis_tool`

This writes the script under `ui/automated_calibration/analysis_tools/<tool_name>/<timestamp>/` and registers a new function tool named `<tool_name>`. When called, that tool runs the saved script inside `python_sandbox` (so it can use numpy/pandas/matplotlib) and passes your provided JSON as `args`.

## Submodels (parallel helper instances)

In the chat UI, the main model can spawn "submodels" via function tools:
- `submodel_batch`: create/reuse submodels and run tasks (optionally in parallel)
- `submodel_list`: list active submodels
- `submodel_close`: terminate a submodel

Implementation detail:
- Submodels run in a separate Python process (`desktop_agent.submodel_worker`) and communicate over a local IPC channel.
- The main process sends periodic `ping` messages (configurable in Controls).

## Demo

Minimal end-to-end demo:

- `scripts/self_tooling_demo.py`

Example:

```powershell
python scripts/self_tooling_demo.py "Create a tool that converts Celsius to Fahrenheit and then convert 20C"
```

## How to use this in your own loop

Typical pattern:

1. Create a `ToolRegistry`
2. Register your built-in tools (e.g. `read_file`)
3. Register `create_and_register_python_tool` (so the model can extend itself)
4. Use `run_responses_with_function_tools(..., registry=registry)` for your agent loop

If you want stronger guarantees, add an extra gate before registering:
- Run more tests
- Add a “review model” step that must approve the tool code + verification report before `registry.add(...)`
