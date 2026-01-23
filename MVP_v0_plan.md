# MVP v0 plan

This plan is derived from `project.md` and reflects what is **already implemented** vs what is **still needed** to reach **MVP v0**.

## MVP v0 definition (from `project.md`)

A minimal usable loop:

1. Small desktop chat window (right-docked by default, draggable)
2. Planner loop:
   1) capture screenshot
   2) ask LLM for next action(s)
   3) execute actions via controller
   4) post progress back to chat

Supported MVP capabilities:

- Mouse move/click/drag
- Keyboard typing/shortcuts
- Simple browser-app UI tasks
- Step-mode approvals
- Safety: ESC emergency stop must always work

---

## Current implementation status

### Implemented

- **Project structure / packaging**
  - `src/` layout with `pyproject.toml` and editable install support.

- **Controller** (`src/desktop_agent/controller.py`)
  - Windows SendInput backend for mouse/keyboard.
  - JSON stdin/stdout command server (`ControlsServer`).
  - Global ESC emergency stop (poll-based).
  - Step-mode approval gating via `{"op":"approve"}`.

- **Protocol validation (partial)** (`src/desktop_agent/protocol.py`)
  - TypedDict action definitions.
  - `validate_action()` / `validate_actions()`.

- **Tests & CI**
  - Unit tests for controller server behavior and protocol validation.
  - GitHub Actions workflow runs unit tests + a smoke test.

### Not implemented yet (core MVP v0 work)

- `vision.py` screenshot capture + utilities
- `llm.py` multimodal call wrapper + strict JSON parsing/retry
- `executor.py` safe execution layer (rate limits, bounds checks, timeouts, pause/resume, step mode integration)
- `planner.py` agent loop (observe → plan → act → observe)
- `ui.py` desktop chat UI (docked right, draggable, always-on-top option, thread-safe updates)
- `main.py` wiring (UI thread + agent worker thread + lifecycle)
- `config.py` central settings (model config, step-mode, hotkeys, etc.)

---

## MVP v0 implementation tasks

### 0) Establish module/package skeleton

- [ ] Create stubs (with minimal docstrings) for:
  - `src/desktop_agent/vision.py`
  - `src/desktop_agent/llm.py`
  - `src/desktop_agent/executor.py`
  - `src/desktop_agent/planner.py`
  - `src/desktop_agent/ui.py`
  - `src/desktop_agent/main.py`
  - `src/desktop_agent/config.py`
- [ ] Add/update tests where pure logic exists (protocol/validation/state machines).

### 1) Vision subsystem (`vision.py`)

Goal: fast screenshot capture + conversion helpers.

- [ ] Add dependencies:
  - `mss` (capture)
  - `Pillow` (image conversion/scaling)
- [ ] Implement `ScreenCapture` service:
  - capture full virtual screen
  - return raw bytes (PNG/JPEG) and optionally a PIL image / numpy array
  - return width/height and monitor bounds
- [ ] Implement coordinate conversion utilities:
  - real screen ↔ scaled preview mapping
- [ ] Unit tests:
  - pure conversion helpers
  - image encoding utilities (no real screen required)

### 2) LLM subsystem (`llm.py`)

Goal: given goal + screenshot(s), return **strict structured JSON**: `high_level`, `actions`, `notes`.

- [ ] Decide provider for MVP (OpenAI first per `project.md`).
- [ ] Add dependency (likely `openai`).
- [ ] Implement a small adapter:
  - build messages (system + user)
  - optionally include screenshot as image input
  - enforce JSON-only reply
  - parse response strictly, retry with stricter prompt on failure
- [ ] Validate returned `actions` using `desktop_agent.protocol.validate_actions()`.
- [ ] Unit tests:
  - parsing/validation of mock LLM responses
  - retry logic when JSON parsing fails

### 3) Executor (`executor.py`)

Goal: safely run protocol actions against the controller.

MVP design choice:
- For now, integrate **directly** with `WindowsControls` (in-process) to keep complexity down.
- Later, optionally add an out-of-process controller server mode.

Tasks:

- [ ] Implement `Executor` with:
  - `execute(actions)`
  - configurable rate limit (`max_actions_per_second`)
  - optional coordinate bounds checks using virtual screen metrics
  - per-batch timeout
  - `release_all()` on exception
- [ ] Implement **pause/resume/stop**:
  - pause = block execution until resumed
  - stop = immediately `release_all()` and abort
- [ ] Implement Step Mode integration:
  - action-level approval via callback/hook: `requires_approval(action_batch) → bool`
  - executor should be able to block waiting for approval without blocking UI thread
- [ ] Unit tests:
  - use a fake controls backend to ensure executor sends expected calls
  - test pause/stop behavior

### 4) Planner (`planner.py`)

Goal: the main agent loop and safe incremental planning.

- [ ] Implement state machine (as described in `project.md`):
  - `IDLE → PLANNING → EXECUTING → WAITING_FOR_USER → DONE/ERROR`
- [ ] Planner loop:
  - capture screenshot
  - call LLM for a small batch of actions
  - validate actions
  - hand to executor
  - post high-level updates continuously
  - re-observe often (avoid long chains)
- [ ] Keep history:
  - last N actions + outcomes
  - last screenshot metadata (not necessarily bytes)
- [ ] Unit tests:
  - state transitions
  - ensures planner never emits unvalidated actions

### 5) UI (`ui.py`)

Goal: a small, always-on-top (optional) chat UI that is docked right by default and draggable.

- [ ] Choose UI toolkit (**PySide6** recommended in `project.md`).
- [ ] Implement window:
  - chat history
  - user input + send
  - status indicator (RUNNING/PAUSED/STOPPED)
  - high-level plan feed (live)
  - optional low-level action feed
- [ ] Implement docking behavior:
  - initial right-dock
  - drag anywhere
  - optional snap-to-edge threshold
- [ ] Thread-safe update API:
  - agent thread can call something like `ui.post_message(...)` safely

### 6) Wiring / lifecycle (`main.py` + `config.py`)

Goal: start UI, start agent thread, handle shutdown.

- [ ] `config.py`
  - load env vars / config for model key, step-mode, always-on-top, etc.
- [ ] `main.py`
  - start UI thread
  - start agent worker thread
  - connect queues/signals for UI updates
  - ensure clean stop:
    - stop planner/executor
    - `controller.release_all()`

### 7) MVP v0 acceptance test script

Goal: reproduce `project.md` success criteria.

- [ ] Provide a minimal run path:
  - `python -m desktop_agent.main` (or similar)
- [ ] Manual test checklist:
  - UI appears docked right
  - user enters: “Open EasyEDA and search for CM4 composite out pin”
  - agent outputs a high-level plan
  - agent executes visible actions
  - agent keeps updating intent
  - ESC stops immediately

---

## Suggested development order (practical)

1) `vision.py` (capture + utilities)
2) `executor.py` (safe execution layer)
3) `llm.py` (structured planning output)
4) `planner.py` (loop + state management)
5) `ui.py` (chat window + thread-safe updates)
6) `main.py` (wire everything)

This order keeps each next component testable with fakes before integrating the full UI.
