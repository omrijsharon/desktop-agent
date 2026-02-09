# Guide: Using the LLM in this repo

This project uses the OpenAI **Responses API** via a small wrapper in `src/desktop_agent/llm.py`. The core pattern is:

1) Build an `input` payload (system prompt + user prompt; optionally an image).
2) Call the model (`responses.create`).
3) Parse/validate the output (JSON-only for the planner).

---

## Where the LLM call lives (the “official” path)

For the desktop agent loop, start with these modules:

- `src/desktop_agent/llm.py`
  - `PlannerLLM`: the high-level façade used by the planner loop.
  - `LLMConfig`: config (model id, retries, fake-mode).
  - `OpenAIResponsesClient`: a tiny adapter over `openai.OpenAI().responses.create(...)`.
  - `FakeLLMClient`: deterministic offline plans (safe demo mode).
  - Internals worth knowing:
    - `_build_openai_input(goal, screenshot_png)`: builds the Responses-API message list.
    - `_parse_plan_json(text)`: extracts JSON and validates it into an `LLMPlan`.
- `src/desktop_agent/prompts.py`
  - `system_prompt()` (currently an alias for `compiler_prompt()`): your system instructions for JSON plans + action safety.
  - `allowed_actions_text()`: the action schema as text.
  - `narrator_prompt()` / `compiler_prompt()`: prompt templates (not all are wired into the main loop yet).
- `src/desktop_agent/protocol.py`
  - `validate_actions(...)`: validates the LLM’s `actions` list.
  - `Action`: the canonical action schema the executor accepts.

The top-level wiring is:

- `src/desktop_agent/main.py`: constructs `PlannerLLM(...)` and `Planner(...)`
- `src/desktop_agent/planner.py`: observe → plan → act loop
- `src/desktop_agent/vision.py`: screenshot capture (PNG bytes)
- `src/desktop_agent/executor.py` + `src/desktop_agent/controller.py`: executes validated actions on Windows

---

## Environment variables (how the agent decides “real” vs “fake”)

Loaded by `src/desktop_agent/config.py` (it best-effort loads a local `.env` from repo root):

- `OPENAI_API_KEY`: if missing, the app defaults to fake mode (unless you inject your own client).
- `OPENAI_MODEL`: default model id for the UI (must be in `SUPPORTED_MODELS` to show up in the picker).
- `DESKTOP_AGENT_FAKE_LLM=1`: forces fake mode even if a key exists.

---

## The simplest “call the model” (planner-style, with screenshot)

Use `PlannerLLM.plan_next(...)` if you want a **validated JSON plan** back:

```py
from desktop_agent.llm import PlannerLLM, LLMConfig
from desktop_agent.vision import ScreenCapture

llm = PlannerLLM(config=LLMConfig(model="gpt-5.2-2025-12-11"))
shot = ScreenCapture().capture_fullscreen(preview_max_size=None, include_cursor=True)

plan = llm.plan_next(goal="minimize the vscode window", screenshot_png=shot.png_bytes)
print(plan.high_level)
print(plan.actions)  # already validated into Action typed dicts
```

What gets “injected” into the model:

- System prompt: `desktop_agent.prompts.system_prompt()`
- User content:
  - text part: `Goal: ...`
  - image part (optional): base64 PNG as a `data:image/png;base64,...` URL

That assembly happens inside `desktop_agent.llm._build_openai_input(...)`.

---

## Calling the model directly (Responses API wrapper)

If you just want raw text output (not the strict planner JSON), use `OpenAIResponsesClient`:

```py
import os
from desktop_agent.llm import OpenAIResponsesClient

client = OpenAIResponsesClient(api_key=os.environ["OPENAI_API_KEY"])

inp = [
  {"role": "system", "content": [{"type": "input_text", "text": "You are a helpful assistant."}]},
  {"role": "user", "content": [{"type": "input_text", "text": "Write 3 ideas for a desktop agent."}]},
]

txt = client.responses_create(model="gpt-5.2-2025-12-11", input=inp)
print(txt)
```

If you want to include an image, add an `input_image` part:

```py
import base64

b64 = base64.b64encode(png_bytes).decode("ascii")
inp[1]["content"].append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})
```

Examples in this repo:

- `scripts/llm_screenshot_demo.py`: calls `PlannerLLM.plan_next(...)` with a real screenshot
- `scripts/llm_image_file_grid_demo.py`: calls `OpenAIResponsesClient.responses_create(...)` repeatedly for image-tiling

---

## How “prompt injection” works here (and where to edit prompts)

There are two places you can inject instructions:

1) **System prompt** (global rules / formatting / safety)
   - Edit `src/desktop_agent/prompts.py`:
     - `compiler_prompt()` and/or `system_prompt()`
2) **User prompt** (per-run goal / task)
   - In the agent loop: `goal` passed into `PlannerLLM.plan_next(goal=...)`
   - In scripts: whatever you put into the user message parts (`{"type":"input_text","text":"..."}`)

Important: for the planner path, the code enforces JSON output via:

- `text={"format": {"type": "json_object"}}` in `PlannerLLM.plan_next(...)`
- `_parse_plan_json(...)` + `validate_actions(...)` as a second guard

So even if the model tries to output prose, the call is configured to prefer JSON, and the parser/validator will reject unsafe/invalid actions.

---

## “Initiate a new model” (switching models in code/UI)

There are three common ways to choose a model:

1) Set `OPENAI_MODEL` in `.env` and start the UI (`python -m desktop_agent.main`)
2) Pick from the model dropdown in the UI (it uses `SUPPORTED_MODELS` in `src/desktop_agent/config.py`)
3) In code/scripts, pass `LLMConfig(model="...")` when constructing `PlannerLLM`

If you want the UI to show a new model id, add it to:

- `src/desktop_agent/config.py` → `SUPPORTED_MODELS`

---

## Running the full agent loop (observe → plan → act)

The end-to-end desktop agent is:

- `python -m desktop_agent.main`

Internally it does:

- capture screenshot (`ScreenCapture`)
- call LLM (`PlannerLLM.plan_next`)
- validate (`validate_actions`)
- execute (`Executor.execute`)

If you want a minimal non-UI loop for experimenting, `scripts/llm_screenshot_demo.py` is the closest reference.

---

## Notes for the “learning from experience” direction (future)

Keep these seams in mind for later “agents that improve their instructions”:

- Prompts are centralized in `src/desktop_agent/prompts.py` (easy to version, diff, and evolve).
- `Planner.history` in `src/desktop_agent/planner.py` already stores `(high_level, actions)` per iteration; you can extend this to store:
  - screenshot hashes/paths, outcomes, user feedback, and errors
  - reward signals (“worked / didn’t work”)
- A future “trainer” can generate updated prompt text (or prompt deltas) and write new prompt variants to disk, then select among them at runtime.

