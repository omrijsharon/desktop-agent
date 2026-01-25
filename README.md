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
