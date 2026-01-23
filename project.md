# desktop-agent

## What this project is

`desktop-agent` is a **full desktop coworking agent** that can:

- Show a **small chat UI** (snapped to the right edge by default, movable)
- Let the user chat with an LLM about what to do
- **Observe the desktop** (screenshots)
- **Plan actions** (high-level “intent” steps)
- **Execute actions** (low-level mouse/keyboard) using the OS controller
- Continuously **explain in chat** what it’s doing at a high level while it performs low-level steps

This repo must implement **everything end-to-end** (UI + vision + planner + executor + safety).

Primary target OS: **Windows**.

---

## Product goals

### UX goals

1. **Chat window**
   - Always-on-top small window
   - Snaps/docks to the right side by default
   - Can be dragged anywhere
   - Contains:
     - User chat input
     - LLM messages
     - “Now doing:” high-level action feed (live)
     - Status indicators (**RUNNING** / **PAUSED** / **STOPPED**)

2. **Transparency while acting**
   - The agent must continuously post:
     - The high-level plan step it’s attempting (abstract actions)
     - The current low-level action being executed (optional but useful)
   - Example high-level messages:
     - “Opening EasyEDA project”
     - “Routing net to CM4 composite pin”
     - “Placing JST-SH footprint and aligning”

3. **Human control**
   - Global emergency stop: **ESC** stops execution immediately and releases held inputs
   - Optional **Step Mode**: require user approval per action or per plan step
   - Must be able to pause/resume

### Technical goals

- Cross-cutting: stable action protocol between planner and executor
- Reliable screen capture
- Deterministic low-level execution with robust safety (timeouts, bounds, sanity checks)

---

## MVP scope (what to build first)

### MVP v0 (minimal usable loop)

- A desktop chat window (right-docked, draggable)
- A planner loop that:
  1. Captures a screenshot
  2. Asks the LLM for next action(s)
  3. Executes them via controller
  4. Posts progress to chat

Supported capabilities:

- Mouse move/click/drag
- Keyboard typing/shortcuts
- Simple “click this UI element” tasks in browser apps
- Step-mode approvals

---

## Architecture (modules)

Recommended structure:

```
desktop-agent/
├── controller.py  # OS-level input (already implemented)
├── vision.py       # screenshot capture + preprocessing
├── llm.py          # model adapter (OpenAI first, configurable)
├── planner.py      # produces high-level + low-level actions
├── executor.py     # runs low-level actions via controller safely
├── protocol.py     # action schemas and validation
├── ui.py           # chat window UI, docking, logs
├── main.py         # wiring, event loop, threads
├── config.py       # settings (model, hotkeys, step mode, etc.)
└── project.md      # this file
```

### Threading model (suggested)

- **UI thread**: renders chat, user input, status
- **Agent thread**: capture → plan → execute loop
- **Hotkey watcher**: ESC stop (may already be inside controller; keep global)

---

## Vision subsystem (`vision.py`)

### Requirements

- Capture full-screen screenshots at ~1–5 FPS (MVP)
- Optional region capture (to reduce cost)
- Provide:
  - Raw screenshot (PNG/JPEG bytes)
  - Scaled-down preview for UI
  - Basic utilities:
    - Coordinate conversion between scaled image and real screen pixels

Implementation suggestions (Windows):

- Use `mss` for fast capture
- Convert to PIL / numpy for scaling/compressing

---

## LLM subsystem (`llm.py`)

### Requirements

- Configurable model provider:
  - Start with OpenAI
  - Design interface so others can be added later
- Support **multimodal** calls (image + text)
- Hard requirement: return **structured JSON** for actions

### Output format constraint

The model must output:

- `high_level`: list of abstract steps (strings)
- `actions`: list of low-level actions (JSON objects; schema below)
- `notes`: optional user-facing text to show in chat

If parsing fails, retry with a stricter system prompt.

---

## Protocol (`protocol.py`)

### Low-level action schema (the executor consumes this)

Each action is a JSON object with an `"op"` field.

> Note: When "Step Mode" is enabled, the controller/server may require an explicit
> control-plane approval message before executing the next action batch:
>
> ```json
> {"op":"approve"}
> ```

#### Mouse

```json
{"op":"move","x":400,"y":300}
{"op":"click","button":"left"}
{"op":"mouse_down","button":"left"}
{"op":"mouse_up","button":"left"}
{"op":"scroll","dx":0,"dy":-3}
```

#### Keyboard

```json
{"op":"key_down","key":"ctrl"}
{"op":"key_up","key":"ctrl"}
{"op":"key_combo","keys":["ctrl","c"]}
{"op":"type","text":"easyeda.com","delay":0}
```

#### Safety / control

```json
{"op":"release_all"}
{"op":"pause"}   
{"op":"resume"}  
{"op":"stop"}    
```

Notes:

- `pause`/`resume` are executor-level: the executor may implement them by not sending actions while paused.
- `controller.py` currently supports many of these already (and ESC stop).

---

## Executor (`executor.py`)

### Responsibilities

- Validate actions against schema
- Execute via `controller.py`
- Provide safety:
  - Max action rate
  - Optional bounds checking for coordinates
  - Timeouts per action batch
  - `release_all` on exceptions
- Emit progress callbacks for UI:
  - “Executing: click(left)”
  - “Executing: type('...')”

---

## Planner (`planner.py`)

### Responsibilities

- Maintain an internal state machine:
  - `IDLE` → `PLANNING` → `EXECUTING` → `WAITING_FOR_USER` → `DONE`/`ERROR`

For each iteration:

- Capture screenshot
- Build prompt with:
  - User goal
  - Current status
  - Last *N* actions and outcomes
- Call LLM
- Parse structured output
- Send high-level updates to UI
- Submit actions to executor

### Critical planner constraint

The planner must always produce:

- Safe, incremental actions
- Avoid long uncontrolled sequences
- Prefer “observe → act → observe” loops

---

## UI (`ui.py`)

### Requirements

- Small window, docked right by default
- Draggable (user can move anywhere)
- Always-on-top option

Components:

- Chat history
- Input box + send button
- Status indicator (**RUNNING**/**PAUSED**)
- “High-level plan” feed (live)
- Action feed (optional, collapsible)

Implementation suggestions:

- Python: **PySide6** (recommended) or **tkinter** (simpler but uglier)

Docking:

- Start window at right edge
- On drag end: optionally snap to nearest edge if within threshold

---

## Safety (non-negotiable)

- ESC emergency stop must always work.
- Any exception in planner/executor must call:
  - `controller.release_all()`
  - transition to **STOPPED**
- Step Mode must exist:
  - Action-level approval **or** plan-step approval

Default behavior should be conservative:

- Short action batches
- Frequent re-observation of screen

---

## Controller JSON examples (for reference / testing)

These are the low-level actions the planner/executor will generate.

### Open a browser URL

```json
{"op":"key_combo","keys":["ctrl","l"]}
{"op":"type","text":"https://easyeda.com","delay":0}
{"op":"key_down","key":"enter"}
{"op":"key_up","key":"enter"}
```

### Drag-select area

```json
{"op":"move","x":400,"y":400}
{"op":"mouse_down","button":"left"}
{"op":"move","x":800,"y":700}
{"op":"mouse_up","button":"left"}
```

### Scroll down

```json
{"op":"scroll","dx":0,"dy":-5}
```

---

## Development order (agent instructions)

Implement in this sequence:

1. `protocol.py`
   - Define action dataclasses / `TypedDict`s
   - Validation helpers
2. `vision.py`
   - Screenshot capture
   - Coordinate conversion utilities
   - Image encoding utilities (JPEG/PNG)
3. `ui.py`
   - Docked-right small window
   - Movable
   - Chat + status + plan feed
   - Thread-safe “append message” API
4. `llm.py`
   - OpenAI multimodal call
   - Strict JSON output parsing / retry
5. `executor.py`
   - Runs actions safely using controller
   - Pause/resume/stop/step-mode integration
6. `planner.py`
   - Main control loop
   - Uses vision + llm + executor
   - Emits high-level updates to UI continuously
7. `main.py`
   - Wiring + lifecycle
   - Start UI + agent thread
   - Clean shutdown

---

## Success criteria (MVP acceptance test)

1. Run `python main.py`
2. Chat window appears docked right (movable)
3. User types: “Open EasyEDA and search for CM4 composite out pin”
4. Agent:
   - Posts a high-level plan (in chat)
   - Executes visible actions (mouse/keyboard)
   - Keeps updating high-level intent as it works
5. **ESC** stops instantly and safely at any moment

---

## Notes to the coding agent

- Prefer clarity over cleverness.
- Keep components loosely coupled.
- Avoid any long chain of actions without re-checking the screen.
- Make everything testable (unit tests where possible for protocol/validation).
- Do not block the UI thread. Use worker threads and signal/queue to UI.

If you want this to be even more actionable, say whether you want **PySide6** or **tkinter** for the UI (I’d default to PySide6 for a clean dockable widget).