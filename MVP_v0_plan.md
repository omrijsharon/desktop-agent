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

- [x] Create stubs (with minimal docstrings) for:
  - `src/desktop_agent/vision.py` ✅ (implemented)
  - `src/desktop_agent/llm.py` ✅ (implemented)
  - `src/desktop_agent/executor.py` ✅ (implemented)
  - `src/desktop_agent/planner.py` ✅ (implemented)
  - `src/desktop_agent/ui.py` ✅ (implemented)
  - `src/desktop_agent/main.py` ✅ (implemented)
  - `src/desktop_agent/config.py` ✅ (implemented)
- [x] Add/update tests where pure logic exists (protocol/validation/state machines).

### 1) Vision subsystem (`vision.py`)

Goal: fast screenshot capture + conversion helpers.

- [x] Add dependencies:
  - `mss` (capture)
  - `Pillow` (image conversion/scaling)
- [x] Implement `ScreenCapture` service:
  - capture full virtual screen
  - return raw bytes (PNG/JPEG) and optionally a PIL image / numpy array
  - return width/height and monitor bounds
- [x] Implement coordinate conversion utilities:
  - real screen ↔ scaled preview mapping
- [x] Unit tests:
  - pure conversion helpers
  - image encoding utilities (no real screen required)

### 2) LLM subsystem (`llm.py`)

Goal: given goal + screenshot(s), return **strict structured JSON**: `high_level`, `actions`, `notes`.

- [x] Decide provider for MVP (OpenAI first per `project.md`).
- [x] Add dependency (likely `openai`).
- [x] Implement a small adapter:
  - build messages (system + user)
  - optionally include screenshot as image input
  - enforce JSON-only reply
  - parse response strictly, retry with stricter prompt on failure
- [x] Validate returned `actions` using `desktop_agent.protocol.validate_actions()`.
- [x] Unit tests:
  - parsing/validation of mock LLM responses
  - retry logic when JSON parsing fails

### 3) Executor (`executor.py`)

Goal: safely run protocol actions against the controller.

MVP design choice:
- For now, integrate **directly** with `WindowsControls` (in-process) to keep complexity down.
- Later, optionally add an out-of-process controller server mode.

Tasks:

- [x] Implement `Executor` with:
  - `execute(actions)`
  - configurable rate limit (`max_actions_per_second`)
  - optional coordinate bounds checks using virtual screen metrics
  - per-batch timeout
  - `release_all()` on exception
- [x] Implement **pause/resume/stop**:
  - pause = block execution until resumed
  - stop = immediately `release_all()` and abort
- [x] Implement Step Mode integration:
  - action-level approval via callback/hook: `requires_approval(action_batch) → bool`
  - executor should be able to block waiting for approval without blocking UI thread
- [x] Unit tests:
  - use a fake controls backend to ensure executor sends expected calls
  - test pause/stop behavior

### 4) Planner (`planner.py`)

Goal: the main agent loop and safe incremental planning.

- [x] Implement state machine (as described in `project.md`):
  - `IDLE → PLANNING → EXECUTING → WAITING_FOR_USER → DONE/ERROR`
- [x] Planner loop:
  - capture screenshot
  - call LLM for a small batch of actions
  - validate actions
  - hand to executor
  - post high-level updates continuously
  - re-observe often (avoid long chains)
- [x] Keep history:
  - last N actions + outcomes
  - last screenshot metadata (not necessarily bytes)
- [x] Unit tests:
  - state transitions
  - ensures planner never emits unvalidated actions

### 5) UI (`ui.py`)

Goal: a small, always-on-top (optional) chat UI that is docked right by default and draggable.

- [x] Choose UI toolkit (**PySide6** recommended in `project.md`).
- [x] Implement window:
  - chat history
  - user input + send
  - status indicator (RUNNING/PAUSED/STOPPED)
  - high-level plan feed (live)
  - optional low-level action feed
- [x] Implement docking behavior:
  - initial right-dock
  - drag anywhere
  - optional snap-to-edge threshold
- [x] Thread-safe update API:
  - agent thread can call something like `ui.post_message(...)` safely

### 6) Wiring / lifecycle (`main.py` + `config.py`)

Goal: start UI, start agent thread, handle shutdown.

- [x] `config.py`
  - load env vars / config for model key, step-mode, always-on-top, etc.
- [x] `main.py`
  - start UI thread
  - start agent worker thread
  - connect queues/signals for UI updates
  - ensure clean stop:
    - stop planner/executor
    - `controller.release_all()`

### 7) MVP v0 acceptance test script

Goal: reproduce `project.md` success criteria.

- [x] Provide a minimal run path:
  - `python -m desktop_agent.main` (or similar)
- [ ] Manual test checklist:
  - UI appears docked right
  - user enters: “Open EasyEDA and search for CM4 composite out pin”
  - agent outputs a high-level plan
  - agent executes visible actions
  - agent keeps updating intent
  - ESC stops immediately

---

## Nice-to-have / contingency: two-stage planning (planner → action translator)

If the single-model approach struggles (e.g., fragile coordinate selection, overly long action chains, or repeated invalid actions), consider splitting planning into **two stages**:

1) **High-level planner** (intent): decides *what* to do next in human terms (small, incremental step).
2) **Low-level action translator** (execution plan): converts that intent into the strict `desktop_agent.protocol.Action` list for the controller/executor.

### Why we might need this

- **Reliability:** a “translator” can be prompted/validated to be extremely strict and conservative, reducing malformed actions.
- **Safety:** translator can refuse when targets are uncertain (return `actions: []`) and force step-mode confirmation for destructive operations.
- **Maintainability:** prompts and logic for “what to do” vs “how to click/type” evolve independently.
- **Model flexibility:** the translator can be a cheaper model (or even deterministic code) while the planner uses a stronger model.

### Tradeoffs / costs

- **Latency + cost:** potentially one extra model call per loop.
- **More plumbing:** additional types/schemas, logs, tests, and debugging surface.
- **Failure modes:** planner/translator mismatch (planner expects capabilities translator doesn’t have).

### Implementation approach (tasks)

A recommended path is to try **deterministic translation first**, and only add a second LLM translator if needed.

#### A) Deterministic translator (preferred first step)

- [ ] Define an intermediate “intent schema” (separate from low-level actions), e.g.:
  - click a UI element by *description* ("Click the Search box")
  - type text into a target field ("Type 'hello' into the active input")
  - open start menu, focus window, etc. (keep the list small)
- [ ] Add `desktop_agent/intent.py`:
  - TypedDicts + validation for the intent schema
  - mapping intent → `protocol.Action[]` using heuristics (focus-first, small moves)
- [ ] Extend `vision.py` (optional):
  - lightweight target helpers (e.g., click center of screen quadrant, or provide a “cursor hint”)
  - keep it heuristic-only in MVP (no OCR dependency unless necessary)
- [ ] Update planner loop:
  - LLM returns intent JSON
  - deterministic translator converts intent → actions
  - executor runs actions as usual
- [ ] Tests:
  - validation tests for intent schema
  - unit tests for intent → action mapping (pure logic)

#### B) Second LLM translator (fallback if A is insufficient)

- [ ] Add `TranslatorLLM` (parallel to `PlannerLLM`) with its own strict prompt:
  - Input: screenshot + high-level intent + constraints
  - Output: `high_level/actions/notes` with actions **only** (schema locked)
  - Must be conservative: if uncertain, output `actions: []` and explain.
- [ ] Add config switches:
  - enable/disable translator stage
  - separate model selection for translator (default to smaller model)
- [ ] Logging + observability:
  - persist planner-intent, translator-actions, and validation failures for debugging
- [ ] Tests:
  - mock translator responses and ensure strict validation + retry-on-parse-failure works
  - ensure step-mode gating still works correctly end-to-end

---

## Planned enhancement: iterative action batches + post-action re-observation + self-evaluation (recommended)

Current single-step demo (`scripts/llm_screenshot_demo.py`) can execute a one-shot action list. In practice, for safety and reliability, the agent should be explicitly instructed and wired to behave conservatively:

- After **each action** (or small batch), capture a **new screenshot** and plan again.
- When using the mouse, prefer **move → re-observe → click** (avoid clicking based on a stale image).
- Add an explicit **self-evaluation** field in the model response so we can see whether it believes the task succeeded, needs more steps, failed-but-wants-to-retry, or failed-and-wants-to-quit.

Important: the self-evaluation data is **for logging/UI only**. The controller/executor must **ignore** it.

### Instruction/prompt updates (LLM behavior)

- [ ] Update `desktop_agent.prompts.system_prompt()` to include a rule like:
  - “You will receive a new screenshot after actions are executed. When using the mouse: move first, request/expect a new screenshot, and only click when you are confident the cursor is over the intended target.”
- [ ] Add explicit “small batches only” guidance:
  - Prefer 1–3 actions per cycle.
  - Avoid multi-click sequences without intermediate re-observation.

### Response schema updates (self-evaluation)

- [ ] Extend the LLM response schema (planner output) to include e.g.:
  - `self_eval`: { `status`: "success" | "continue" | "retry" | "give_up", `reason`: string }
- [ ] Ensure parsing/validation in `desktop_agent.llm` tolerates absence of `self_eval` (backward compatible).
- [ ] Ensure planner/UI logs `self_eval` but executor/controller do not act on it.

### Loop behavior updates (demo + core planner)

- [ ] Update `scripts/llm_screenshot_demo.py` to run as a mini-loop:
  - capture screenshot → ask LLM → validate → execute
  - after **move** actions, always capture a fresh screenshot before any click
  - exit only when:
    - the model returns `self_eval.status == "success"`, OR
    - a max-iterations limit is hit, OR
    - user aborts / ESC stop
- [ ] Add “verification step” for the Chrome-open task:
  - after a click intended to open Chrome, capture a screenshot and ask the model whether Chrome is open.
  - only exit after the model indicates success (via `self_eval`).

### Safety/UX tasks

- [ ] Keep explicit user confirmation before executing each batch in the demo script.
- [ ] Add a max loops / max actions cap to prevent runaway behavior.
- [ ] Document that ESC is the emergency stop and should be used if misbehavior occurs.

### Tests

- [ ] Add unit tests for:
  - parsing `self_eval` when present/absent
  - planner loop respecting `self_eval.status` to stop/continue
  - demo loop logic (can be tested with FakeLLMClient + stub controls)

---

## Two-agent approach (recommended next iteration): Narrator → Translator

The current prompt-only “self-check” approach can be brittle. Instead, split the system into **two LLM roles**:

1) **Narrator agent (high-level intent)**
   - Output: a single plain-language next-step command such as:
     - “Move the cursor a tiny bit to the right toward the Chrome icon.”
     - “Scroll down slightly to reveal more results.”
     - “Re-observe to confirm the cursor is positioned over the target before clicking.”
   - No low-level coordinates, no protocol actions.

2) **Translator agent (low-level compiler)**
   - Input: the narrator’s single intent sentence + the latest screenshot.
   - Output: strict `desktop_agent.protocol.Action[]` (via the existing JSON plan schema).
   - Must be conservative; if it cannot confidently translate, return `actions: []` and explain in `notes`.

**Initial implementation:** use the **same model** for both agents, but keep them as separate prompts and separate calls so they can be tuned independently.

### Tasks

#### Prompts / schemas

- [ ] In `src/desktop_agent/prompts.py`, add two prompts:
  - `narrator_prompt()`: plain text output, exactly one intent sentence.
  - `translator_prompt()` (or `compiler_prompt()`): strict JSON output (existing plan schema), includes coordinate system rules.
- [ ] Ensure the translator prompt explicitly states:
  - x axis is horizontal, y axis is vertical.
  - `move_delta` negative values move opposite direction.
  - “move → re-observe → click” policy.

#### LLM wiring (same model, two calls)

- [ ] In `src/desktop_agent/llm.py`, implement a two-stage call path:
  - Call narrator with (goal + screenshot) → `intent: str`
  - Call translator with (intent + screenshot) → `LLMPlan` (strict JSON)
- [ ] Add logging hooks so the UI/demo can display:
  - narrator intent
  - translator actions
  - validation failures (if any)

#### Planner loop integration

- [ ] In `src/desktop_agent/planner.py`, store and surface both:
  - narrator intent (for “what we’re doing next”)
  - translator-produced `high_level` + actions (for executor)
- [ ] Keep step-mode approvals at the executor boundary (unchanged).

#### Demo script integration

- [ ] Update `scripts/llm_screenshot_demo.py` to use the two-agent flow:
  - print the narrator intent before showing actions
  - still require explicit `yes` before executing actions
  - keep conservative move-only sub-batch behavior before any click

#### Tests

- [ ] Add/update tests in `tests/test_llm.py` covering:
  - narrator output cleaning (single sentence)
  - translator JSON parsing/validation
  - retry behavior per-stage (narrator retry vs translator retry)

#### Config / UX

- [ ] Add config toggles (default ON for experiments):
  - `DESKTOP_AGENT_TWO_AGENT=1` to enable narrator→translator
  - later: separate model IDs per stage (optional)

---
