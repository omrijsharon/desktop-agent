# desktop-agent — Projects & Components

> **Repo:** `desktop-agent`  
> **Package:** `desktop_agent` (under `src/`)  
> **Python:** ≥ 3.10 (dev uses 3.12.6)  
> **UI framework:** PySide6 (Qt)  
> **LLM backend:** OpenAI API (Responses API + Agents SDK)  
> **Last updated:** 2026-02-26

---

## Table of Contents

1. [Terminal Agent UI (flagship)](#1-terminal-agent-ui-flagship)
2. [Browser Helper UI (Playwright automation)](#2-browser-helper-ui-playwright-automation)
3. [Desktop Coworking Agent (MVP v0)](#3-desktop-coworking-agent-mvp-v0)
4. [Evolving Agent Community Simulator](#4-evolving-agent-community-simulator)
5. [Actor–Critic Training Harness (Adversarial Riddles)](#5-actorcritic-training-harness-adversarial-riddles)
6. [Telegram Relay](#6-telegram-relay)
7. [Self-Extending Tool System](#7-self-extending-tool-system)
8. [Automated Calibration UI](#8-automated-calibration-ui)
9. [Shared Infrastructure](#9-shared-infrastructure)
10. [Scripts & Demos](#10-scripts--demos)
11. [Tests](#11-tests)
12. [Docs](#12-docs)

---

## 1. Terminal Agent UI (flagship)

**What it does:** Split-pane PySide6 GUI with a **chat pane** (left) and an **embedded ConPTY terminal** (right). The user types natural-language requests; the model generates and autonomously executes PowerShell commands via `<Terminal>…</Terminal>` blocks. Supports SSH detection, multi-tab terminals, persistent scratchpad, conversation compaction, streaming responses, peer agents, and 15+ function tools.

**Key files:**

| File | Description |
|------|-------------|
| `src/desktop_agent/terminal_agent_ui.py` | Main application (~2,870 lines) — `TerminalAgentWindow`, `_Worker`, `_TabState`, `_ConptyTerminal` |
| `scripts/terminal_agent_ui.py` | Entry-point script (imports and calls `main()`) |

**Run:**

```powershell
.\.venv\Scripts\python.exe scripts\terminal_agent_ui.py
# or
.\.venv\Scripts\python.exe -m desktop_agent.terminal_agent_ui
```

**Architecture:**

```
TerminalAgentWindow (Qt)
├── Chat Pane (left)         — chat bubbles, input, Send/Pause/Stop
├── Terminal Pane (right)    — ConPTY interactive PTY, tabbed (Main, T2, …)
└── _Worker (QThread)        — agent loop
    ├── send_stream(pending) → ChatSession → OpenAI Responses API
    ├── extract <Terminal> blocks
    ├── execute via _ConptyTerminal.send_and_collect()
    ├── format <TerminalResponse>
    └── loop (up to max_rounds, default 12)
```

**Key features:**
- SSH state detection & context injection (local vs. remote)
- Head+tail stdout truncation (first 40 + last 80 lines)
- Terminal noise stripping (ANSI, progress bars, echoed commands)
- Conversation compaction (drops old turns when tokens exceed 60% of context window)
- Persistent scratchpad (key-value store surviving compaction)
- Skill-based planning (`plan_update` tool — create/track structured task plans with scratchpad integration)
- Error-retry nudge & self-verification prompts
- Peer agent creation & inter-agent messaging
- Configurable max_rounds via UI spinbox
- Token usage color-coded indicator
- "Continue" button for one-click recovery from premature stops
- SSH status indicator (`🟢 SSH: user@host` / `⚪ Local`)

**Registered tools:** `read_file`, `write_file`, `append_file`, `python_sandbox`, `render_plot`, `web_search`, `web_fetch`, `playwright_browser`, `wait`, `ssh_read_file`, `ssh_write_file`, `ssh_replace_line`, `ssh_run_command`, `ssh_patch_file`, `peer_agent_ask`, `peer_terminal_run`, `create_terminal_agent`, `set_system_prompt`, `scratchpad_set`, `scratchpad_clear`, `plan_update`, plus `<Terminal>` text blocks.

---

## 2. Browser Helper UI (Playwright automation)

**What it does:** PySide6 chat UI that drives a Playwright browser instance. The model can navigate URLs, click elements, fill forms, take screenshots, and extract page content through tool calls. Includes an `ask_user` tool for credentials/confirmation.

**Key files:**

| File | Description |
|------|-------------|
| `src/desktop_agent/browser_helper_ui.py` | Main application (749 lines) |
| `scripts/browser_helper_ui.py` | Entry-point script |

**Run:**

```powershell
.\.venv\Scripts\python.exe -m desktop_agent.browser_helper_ui
# or
.\.venv\Scripts\python.exe scripts\browser_helper_ui.py
```

**Key deps:** `ChatSession`, `tools.make_playwright_browser_handler`

---

## 3. Desktop Coworking Agent (MVP v0)

**What it does:** The original "screen observe → plan → execute" loop. Captures full-screen screenshots via `mss`, sends them to GPT-4o for a strict-JSON action plan (mouse moves, clicks, key presses, scrolls, drags, typing), and executes the plan via Windows `SendInput` (ctypes). PySide6 overlay UI with step-mode approval and ESC emergency stop.

**Key files:**

| File | Description |
|------|-------------|
| `src/desktop_agent/main.py` | Entry point — wires vision, LLM, executor, planner, UI |
| `src/desktop_agent/controller.py` | OS-level mouse/keyboard control via `SendInput` (553 lines) |
| `src/desktop_agent/executor.py` | Safe execution layer — translates actions to controller calls (259 lines) |
| `src/desktop_agent/planner.py` | Observe → plan → act state machine (238 lines) |
| `src/desktop_agent/vision.py` | Screenshot capture + coordinate mapping (206 lines) |
| `src/desktop_agent/protocol.py` | Action schema definitions & validation (209 lines) |
| `src/desktop_agent/prompts.py` | LLM prompt templates for action planning (143 lines) |
| `src/desktop_agent/llm.py` | LLM adapter — screenshot + goal → JSON action plan (345 lines) |
| `src/desktop_agent/ui.py` | PySide6 overlay UI for MVP (500 lines) |

**Run:**

```powershell
.\.venv\Scripts\python.exe -m desktop_agent.main
```

**Environment variables:**
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `DESKTOP_AGENT_FAKE_LLM` (optional, `1` = fake mode)
- `STEP_MODE` (optional, `1` = step mode by default)
- `ALWAYS_ON_TOP` (optional, `1` = keep window above others)

**JSON protocol (stdin/stdout):**

```powershell
# Interactive server
python -m desktop_agent.controller
# Single command
'{"op":"move","x":300,"y":300}' | python -m desktop_agent.controller
```

---

## 4. Evolving Agent Community Simulator

**What it does:** A narrator-driven abstract world simulation. Multiple autonomous agents have immutable core drives (survive, reproduce, ensure offspring reach reproduction age) and mutable add-on instructions + personal memory files. The narrator resolves physical actions, social interactions, and outcomes. Agents can evolve their own instructions based on experience across day/night cycles. Supports TensorBoard logging and multiprocessing.

**Key files:**

| File | Description |
|------|-------------|
| `src/desktop_agent/evolving/__init__.py` | Package docstring |
| `src/desktop_agent/evolving/engine.py` | Simulation engine (645 lines) |
| `src/desktop_agent/evolving/engine_mp.py` | Multiprocessing variant |
| `src/desktop_agent/evolving/prompts.py` | Agent & narrator system prompts |
| `src/desktop_agent/evolving/schemas.py` | AgentAction, Gender, validation |
| `src/desktop_agent/evolving/state.py` | WorldState, population, lineage |
| `src/desktop_agent/evolving/config.py` | EvolvingConfig |
| `src/desktop_agent/evolving/tensorboard_logger.py` | TensorBoard telemetry |
| `scripts/evolving/run_sim.py` | Single-process runner |
| `scripts/evolving/run_sim_mp.py` | Multi-process runner |
| `scripts/evolving/config.json` | Simulation configuration |
| `scripts/evolving/README.md` | Concept documentation |

**Run:**

```powershell
python scripts\evolving\run_sim.py
# or (multiprocessing)
python scripts\evolving\run_sim_mp.py
```

**Core concepts:**
- **Narrator (×1):** World/physics engine/judge. Owns world state, outcomes, relationships, lineage.
- **Agents (×N):** Autonomous individuals with immutable core drives and mutable instructions + memory.
- Each agent has `agent_id_xx_memory.md` (experience log) and `agent_id_xx_instructions.md` (mutable add-on instructions).
- Actions: physical world actions, social actions, conversations, inner monologue, self-edit memory/instructions.

---

## 5. Actor–Critic Training Harness (Adversarial Riddles)

**What it does:** Curriculum-based training loop for adversarial riddle solving. A **Riddler** generates riddles targeting a configurable solve-rate sweet spot. A **Critic** validates riddles and grades solver answers. An **Actor** orchestrates N solvers with varied instructions and writes memory for future improvement. All agents return strict JSON for reliable parsing and grading.

**Key files:**

| File | Description |
|------|-------------|
| `src/desktop_agent/training/__init__.py` | Package docstring |
| `src/desktop_agent/training/agents.py` | RiddlerAgent, CriticAgent, ActorAgent, SolverAgent (390 lines) |
| `src/desktop_agent/training/config.py` | TrainingConfig |
| `src/desktop_agent/training/episode.py` | Episode runner |
| `src/desktop_agent/training/llm_json.py` | Strict JSON LLM calling utilities |
| `src/desktop_agent/training/memory.py` | Persistent memory management |
| `src/desktop_agent/training/reward.py` | Reward computation |
| `src/desktop_agent/training/schemas.py` | ActorOutput, CriticReport, RiddlerOutput, SolverOutput |
| `scripts/training/run_curriculum.py` | Full curriculum runner |
| `scripts/training/run_episode.py` | Single episode runner |
| `scripts/training/config.json` | Training configuration |
| `scripts/training/plan.md` | Design document |

**Run:**

```powershell
python scripts\training\run_curriculum.py
# or single episode
python scripts\training\run_episode.py
```

**Outputs:**
- `scripts/training/riddler_memory.md` — riddle history, observed solve-rates, reward signal
- `scripts/training/actor_memory.md` — solver instruction patterns that worked/failed, distilled heuristics
- Per-episode log artifacts (JSONL or Markdown)

---

## 6. Telegram Relay

**What it does:** Headless Telegram Bot API bridge that connects Telegram messages to the Chat UI using file-based IPC. The relay writes inbound messages to `chat_history/telegram/inbox/` and reads outbound responses from `chat_history/telegram/outbox/`. The UI process and relay process never share API keys.

**Key files:**

| File | Description |
|------|-------------|
| `src/desktop_agent/telegram_relay.py` | Bot API polling + IPC (271 lines) |
| `src/desktop_agent/telegram_ipc.py` | File-based IPC protocol (90 lines) |
| `src/desktop_agent/process_manager.py` | Launches relay + Chat UI together (141 lines) |

**Run:**

```powershell
# Relay only
.\.venv\Scripts\python.exe -m desktop_agent.telegram_relay
# Both relay + UI
.\.venv\Scripts\python.exe -m desktop_agent.process_manager
```

**Configuration:**
- `TELEGRAM_BOT_TOKEN` env var (from `.env`)
- Send `/allow_here` in a Telegram group to allowlist it
- See `docs/telegram.md` for setup guide

---

## 7. Self-Extending Tool System

**What it does:** Experimental mechanism for a model to create and register new function tools at runtime. The model calls `create_and_register_python_tool` with a tool schema, Python code, and self-tests. A verifier runs in a subprocess; only passing tools are registered. Subsequent model calls in the same loop include the new tool. Generated tool code is restricted (allow-listed imports, blocked dangerous builtins) and runs in a subprocess per-call.

**Key files:**

| File | Description |
|------|-------------|
| `src/desktop_agent/tools.py` | Tool registry, function-tool runner, tool creation/verification (2,892 lines) |
| `scripts/self_tooling_demo.py` | Minimal demo loop |
| `generated_tools/` | On-disk storage for verified tool code |

**Run:**

```powershell
python scripts\self_tooling_demo.py "Create a tool that converts Celsius to Fahrenheit and then convert 20C"
```

**Safety:**
- Imports are allow-listed
- Some dangerous builtins are blocked
- Generated tools run in a subprocess per-call (no auto-import into main process)

---

## 8. Automated Calibration UI

**What it does:** PySide6 UI for automated hardware calibration workflows via SSH to a Raspberry Pi. Manages WiFi connection, SCP file transfers, remote command execution, and AI-powered analysis of calibration logs. Includes a dedicated analysis sub-window with its own chat session.

**Key files:**

| File | Description |
|------|-------------|
| `src/desktop_agent/automated_calibration_ui.py` | Main calibration UI (589 lines) |
| `src/desktop_agent/automated_calibration_analysis_ui.py` | Analysis chat window (576 lines) |
| `src/desktop_agent/automated_calibration_config.py` | Config dataclasses + system prompt builder (206 lines) |
| `src/desktop_agent/automated_calibration_ops.py` | SSH/SCP/WiFi operations (245 lines) |
| `scripts/automated_calibration_ui.py` | Entry-point script |
| `ui/automated_calibration/style.qss` | Qt stylesheet |

**Run:**

```powershell
.\.venv\Scripts\python.exe scripts\automated_calibration_ui.py
```

**Capabilities:**
- Connect to WiFi networks (Windows `netsh` integration)
- SSH to Raspberry Pi (with fallback host)
- Remote webapp launch & log monitoring
- SCP pull of calibration data directories
- Archiving of calibration runs
- AI-powered analysis of calibration logs via ChatSession

---

## 9. Shared Infrastructure

These modules are used across multiple projects:

| Module | Lines | Description |
|--------|-------|-------------|
| `chat_session.py` | 1,403 | Stateful OpenAI Responses API session with tool calling, submodels, retries, streaming |
| `agent_sdk/` | ~660 | Experimental OpenAI Agents SDK integration (alternative engine to ChatSession) |
| `agent_sdk/session.py` | 539 | `AgentsSdkSession` — Agents SDK-backed chat session matching `ChatSession.send_stream()` shape |
| `agent_sdk/runner.py` | 106 | `ChatEngine` protocol + factory to swap between legacy ChatSession and AgentsSdkSession |
| `agent_hub.py` | 488 | Multi-agent peer registry — create, message, and manage named peer agents |
| `submodels.py` | 150 | Manage auxiliary "sub-model" chat sessions (separate context windows, serializable) |
| `submodel_worker.py` | 256 | Subprocess worker for running model sessions with tools enabled (IPC via `multiprocessing.connection`) |
| `chat_ui.py` | — | Reusable `Bubble`-based chat widget (used by Terminal Agent + Calibration Analysis) |
| `chat_store.py` | 153 | JSON-based per-chat persistence under `chat_history/` |
| `dev_workspace.py` | 278 | Sandboxed per-chat project directory with venv, pip, file I/O (strong path safety) |
| `config.py` | 91 | Env-var config loader with `.env` support |
| `model_caps.py` | 93 | Per-model parameter compatibility cache (avoid 400s from unsupported params) |
| `ui_prefs.py` | 68 | Persist Chat UI defaults (model, system prompt, etc.) across app restarts |
| `ui_tk.py` | 13 | Legacy Tkinter UI stub (deprecated, kept for reference) |

---

## 10. Scripts & Demos

Utility scripts and standalone demos under `scripts/`:

| Script | Description |
|--------|-------------|
| `self_tooling_demo.py` | Self-extending tool creation demo loop |
| `llm_screenshot_demo.py` | Screenshot → GPT description demo |
| `llm_image_file_grid_demo.py` | Send image file grid to GPT |
| `llm_multi_image_grid_demo.py` | Multi-image grid to GPT |
| `openai_web_search_demo.py` | OpenAI web search tool demo (Python) |
| `openai_web_search_demo.mjs` | OpenAI web search tool demo (Node.js) |
| `capture_resolution.py` | Display resolution diagnostics |
| `check_models.py` | List available OpenAI models |
| `gguf_inspect.py` | Inspect GGUF model files (metadata, layers) |
| `gguf_weight_hist.py` | GGUF weight histogram visualization |
| `convert_html_to_md.py` | HTML → Markdown converter |
| `controller_smoke_test.py` | Controller JSON protocol smoke test |
| `playwright_mcp_smoke_test.py` | Playwright MCP server smoke test |
| `cursor_overlay_preview.py` | Cursor overlay rendering preview |
| `mouse_circle_test.py` | Mouse circle drawing test |
| `pytest.ps1` | PowerShell test runner helper |

---

## 11. Tests

**63+ test files** under `tests/`, covering all major components. Run with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Key test groups:

| Test file(s) | Component |
|--------------|-----------|
| `test_terminal_agent_*.py` (10+ files) | Terminal Agent UI — SSH detection, parser, prompt heuristics, improvements |
| `test_agent_hub*.py` | Agent Hub — peer agents, state roundtrip |
| `test_agents_sdk_session_*.py` | Agents SDK session — smoke, streaming |
| `test_automated_calibration_*.py` | Automated Calibration — config, system prompt, UI smoke |
| `test_chat_*.py` | Chat UI, chat store |
| `test_evolving_*.py` | Evolving simulator — state, schemas, resume, instruction history, TensorBoard |
| `test_training_*.py` | Training harness — schemas, reward, LLM JSON |
| `test_tools_*.py` | Tool system — write/append, file search, sandbox |
| `test_controller_server.py` | Controller JSON protocol |
| `test_dev_workspace_paths.py` | Dev workspace path safety |
| `test_planner.py` | Planner state machine |
| `test_protocol.py` | Action protocol validation |
| `test_vision_mapping.py` | Vision coordinate mapping |
| `test_llm.py` | LLM adapter |
| `test_submodel_*.py` | Submodel IPC, auth, records |

---

## 12. Docs

Documentation under `docs/` and root-level markdown files:

| File | Description |
|------|-------------|
| `README.md` | Quickstart, running, testing, architecture overview |
| `project.md` | Product requirements and architecture notes |
| `MVP_v0_plan.md` | MVP v0 plan and milestones |
| `short_term_plan.md` | Immediate tasks / cleanup checklist |
| `t_agent_improvement_plan.md` | 20-item Terminal Agent improvement plan (complete) |
| `openclaw_inspired_upgrade.md` | OpenClaw-inspired upgrade notes |
| `TOOLS.md` | Tool system documentation |
| `docs/gpt-5-2_prompting_guide.md` | Local copy of OpenAI GPT-5.2 prompting guide |
| `docs/dev_workspace.md` | Dev workspace feature documentation |
| `docs/telegram.md` | Telegram relay setup guide |
| `docs/go_online.md` | Going online / deployment notes |
| `docs/plan_skill.md` | Planning skill — format spec & usage guidelines for `plan_update` tool |

---

## Directory Layout (summary)

```
desktop-agent/
├── src/desktop_agent/            # Main Python package
│   ├── terminal_agent_ui.py      # [1] Terminal Agent UI
│   ├── browser_helper_ui.py      # [2] Browser Helper UI
│   ├── main.py                   # [3] Desktop Coworking Agent entry point
│   ├── controller.py             #     └── OS-level input control
│   ├── executor.py               #     └── Safe action execution
│   ├── planner.py                #     └── Observe → plan → act loop
│   ├── vision.py                 #     └── Screenshot capture
│   ├── protocol.py               #     └── Action schema
│   ├── prompts.py                #     └── LLM prompt templates
│   ├── llm.py                    #     └── LLM adapter
│   ├── ui.py                     #     └── PySide6 overlay UI
│   ├── evolving/                 # [4] Evolving Agent Simulator
│   ├── training/                 # [5] Training Harness
│   ├── telegram_relay.py         # [6] Telegram Relay
│   ├── telegram_ipc.py           #     └── File-based IPC
│   ├── process_manager.py        #     └── Multi-process launcher
│   ├── tools.py                  # [7] Self-Extending Tool System
│   ├── automated_calibration_*.py# [8] Automated Calibration
│   ├── agent_sdk/                # [9] Shared: Agents SDK integration
│   ├── chat_session.py           #     Shared: chat engine
│   ├── chat_ui.py                #     Shared: bubble widget
│   ├── chat_store.py             #     Shared: persistence
│   ├── agent_hub.py              #     Shared: multi-agent hub
│   ├── submodels.py              #     Shared: sub-model sessions
│   ├── submodel_worker.py        #     Shared: subprocess worker
│   ├── dev_workspace.py          #     Shared: sandboxed workspace
│   ├── config.py                 #     Shared: configuration
│   ├── model_caps.py             #     Shared: model compatibility
│   ├── ui_prefs.py               #     Shared: UI defaults
│   └── ui_tk.py                  #     Legacy: Tkinter stub
├── scripts/                      # [10] Entry points & demos
│   ├── evolving/                 #      Evolving sim runners
│   └── training/                 #      Training harness runners
├── tests/                        # [11] 63+ test files
├── docs/                         # [12] Documentation
├── chat_history/                 #      Persistent chat data
├── generated_tools/              #      Runtime-created tools
├── python_sandbox_runs/          #      Sandbox execution artifacts
├── ui/                           #      Qt stylesheets
├── pyproject.toml                #      Package metadata
├── requirements.txt              #      Production dependencies
└── requirements-dev.txt          #      Dev/test dependencies
```
