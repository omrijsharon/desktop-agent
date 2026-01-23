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

## Running tests

```powershell
pytest -q
```

## Controller (manual testing)

The repo currently includes a Windows input controller that reads **one JSON object per line** from `stdin` and writes **one JSON response per line** to `stdout`.

### Run (interactive)

```powershell
python .\controller.py
```

### Send a single command

PowerShell example (prints a JSON line into the process):

```powershell
'{"op":"move","x":300,"y":300}' | python .\controller.py
```

### Stop

- Press **ESC** for emergency stop (releases held inputs)
- Or send `{"op":"stop"}`

## Docs

- `project.md` – product requirements and architecture notes
- `short_term_plan.md` – immediate tasks / cleanup checklist
