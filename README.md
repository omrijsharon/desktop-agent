# desktop-agent

A Windows desktop coworking agent that can observe the screen, plan actions, and execute mouse/keyboard input while keeping the user informed via a small chat UI.

## Quickstart

### 1) Create + activate the virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Install dev/test dependencies (optional)

```powershell
pip install -r requirements-dev.txt
```

## Configure API key (local dev)

This repo reads the OpenAI key from the `OPENAI_API_KEY` environment variable.

Recommended workflow:

1) Copy the example file:
   - `Copy-Item .env.example .env`
2) Edit `.env` and set `OPENAI_API_KEY`.

Notes:

- `.env` is **git-ignored** and must never be committed.
- `.env.example` is safe to commit and should contain only placeholders.

## Running tests

```powershell
pytest -q
```

## Run UIs

### Browser Helper UI (Playwright web automation)

From repo root (PowerShell):

- `.\.venv\Scripts\python.exe -m desktop_agent.browser_helper_ui`
- or `.\.venv\Scripts\python.exe scripts\browser_helper_ui.py`

## Tests

### PowerShell (recommended)

Run tests without any PowerShell scripts (works even when `.ps1` execution is disabled):

- From repo root:
  - `.\\.venv\\Scripts\\python -m pytest -q`

- From anywhere:
  - `& "C:\\Users\\tamipinhasi\\Documents\\repos\\desktop-agent\\.venv\\Scripts\\python.exe" -m pytest -q`

Note: `cd /d ...` is **cmd.exe** syntax and will error in PowerShell. If you need to change directories in PowerShell, use:
`Set-Location "C:\\Users\\tamipinhasi\\Documents\\repos\\desktop-agent"`

### Optional helper script

`scripts/pytest.ps1` exists, but it requires PowerShell script execution to be enabled on your system.

## Controller (manual testing)

The repo currently includes a Windows input controller that reads **one JSON object per line** from `stdin` and writes **one JSON response per line** to `stdout`.

### Run (interactive)

```powershell
python -m desktop_agent.controller
```

### Send a single command

PowerShell example (prints a JSON line into the process):

```powershell
'{"op":"move","x":300,"y":300}' | python -m desktop_agent.controller
```

### Stop

- Press **ESC** for emergency stop (releases held inputs)
- Or send `{"op":"stop"}`

## Run (MVP UI)

From repo root (PowerShell):

- `./.venv/Scripts/python -m desktop_agent.main`

Environment variables:

- `OPENAI_API_KEY` (required to get real plans)
- `OPENAI_MODEL` (optional; default `gpt-4o-mini`)
- `DESKTOP_AGENT_FAKE_LLM` (optional; `1` forces fake mode even if an API key exists)
- `STEP_MODE` (optional; `1` enables step mode by default)
- `ALWAYS_ON_TOP` (optional; `1` keeps the window above others)

## Docs

- `project.md` – product requirements and architecture notes
- `short_term_plan.md` – immediate tasks / cleanup checklist
- `docs/gpt-5-2_prompting_guide.md` – local copy of the OpenAI GPT‑5.2 prompting guide (converted from HTML)

## Experimental: self-extending tools (local dev)

This repo includes an experimental mechanism for letting a model *create and register* a new function tool while an agent loop is already running.

Key files:
- `src/desktop_agent/tools.py` – tool registry + function-tool runner + tool creation/verification helpers
- `scripts/self_tooling_demo.py` – minimal demo loop

How it works:
1) The model calls `create_and_register_python_tool` with a tool schema, Python code, and a few self-tests.
2) The handler runs a verifier (in a subprocess) and only registers the tool if the tests pass.
3) Subsequent model calls in the same loop include the new tool in `tools=[...]`.

Run the demo:
- `python scripts/self_tooling_demo.py "Create a tool that converts Celsius to Fahrenheit and then convert 20C"`

Safety notes:
- Generated tool code is restricted (imports are allow-listed; some dangerous builtins are blocked).
- Generated tools run in a subprocess per-call (no auto-import into the main process).
