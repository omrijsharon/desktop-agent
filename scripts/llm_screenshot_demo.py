r"""scripts/llm_screenshot_demo.py

One-off script to sanity-check the LLM call with a screenshot.

Sends:
1) System instructions (desktop agent safety + allowed actions)
2) A PNG screenshot of the full virtual screen
3) A user goal prompt: "open chrome browser"

Prints the raw model output and (optionally) executes the returned actions.

Usage (PowerShell):
  .\\.venv\\Scripts\\python .\\scripts\\llm_screenshot_demo.py

Notes:
- Requires OPENAI_API_KEY in environment (recommended via .env).
- WARNING: This can control your mouse/keyboard. Keep a hand on ESC.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any

from desktop_agent.config import load_config
from desktop_agent.controller import WindowsControls
from desktop_agent.executor import Executor, ExecutorConfig
from desktop_agent.llm import OpenAIResponsesClient
from desktop_agent.protocol import ProtocolError, validate_actions
from desktop_agent.prompts import system_prompt
from desktop_agent.vision import ScreenCapture
from desktop_agent.prompts import compiler_prompt, narrator_prompt


def _build_narrator_input(*, goal: str, screenshot_png: bytes) -> list[dict[str, Any]]:
    sys_prompt = narrator_prompt()

    b64 = base64.b64encode(screenshot_png).decode("ascii")
    return [
        {"role": "system", "content": [{"type": "input_text", "text": sys_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": f"Goal: {goal}"},
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
            ],
        },
    ]


def _build_translator_input(*, intent: str) -> list[dict[str, Any]]:
    sys_prompt = compiler_prompt()

    return [
        {"role": "system", "content": [{"type": "input_text", "text": sys_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": f"Intent: {intent}"},
            ],
        },
    ]


def _clean_intent_text(txt: str) -> str:
    s = (txt or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return s


# NOTE: legacy single-stage builder kept for reference, but unused by this script.
# def _build_input(...):
# ...existing code...


def main() -> int:
    cfg = load_config()

    api_key = os.environ.get("OPENAI_API_KEY")
    force_fake = os.environ.get("DESKTOP_AGENT_FAKE_LLM", "0") in {"1", "true", "True"}

    if force_fake or not api_key:
        print(
            "ERROR: This script is for real model calls. "
            "Set OPENAI_API_KEY (via .env) and ensure DESKTOP_AGENT_FAKE_LLM is not set.",
            file=sys.stderr,
        )
        return 2

    goal = "open chrome browser only using the mouse"

    controls = WindowsControls()
    ex = Executor(
        controls,
        ExecutorConfig(max_actions_per_second=10.0, batch_timeout_s=10.0),
    )

    max_iters = int(os.environ.get("LLM_DEMO_MAX_ITERS", "12"))

    for it in range(1, max_iters + 1):
        print(f"\n=== ITERATION {it}/{max_iters} ===\n")

        print("Capturing screenshot...")
        cap = ScreenCapture()
        shot = cap.capture_fullscreen(preview_max_size=None, include_cursor=True)
        screenshot_png = shot.png_bytes
        print(f"Captured {len(screenshot_png)} bytes")

        # Save the exact image we send to the LLM (with cursor marker)
        out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "llm_input_with_cursor.png"))
        try:
            with open(out_path, "wb") as f:
                f.write(screenshot_png)
            print(f"Saved LLM input screenshot to: {out_path}")
        except Exception as e:
            print(f"WARNING: Failed to save screenshot: {e}", file=sys.stderr)

        # --- Narrator stage (plain text intent) ---
        narrator_inp = _build_narrator_input(goal=goal, screenshot_png=screenshot_png)

        print(f"Calling narrator (model): {cfg.openai_model}")
        client = OpenAIResponsesClient(api_key=api_key)

        narrator_txt = client.responses_create(
            model=cfg.openai_model,
            input=narrator_inp,
        )
        intent = _clean_intent_text(narrator_txt)
        print(f"\nNARRATOR INTENT: {intent}\n")

        # --- Translator stage (strict JSON actions) ---
        inp = _build_translator_input(intent=intent)

        print(f"Calling translator (model): {cfg.openai_model}")
        txt = client.responses_create(
            model=cfg.openai_model,
            input=inp,
            # Ask for strict JSON.
            text={"format": {"type": "json_object"}},
        )

        print("\n--- RAW MODEL OUTPUT ---\n")
        print(txt)
        print("\n--- END ---\n")

        # Parse + validate
        try:
            payload = json.loads(txt)
        except json.JSONDecodeError as e:
            print(f"ERROR: Model output was not valid JSON: {e}", file=sys.stderr)
            return 3

        self_eval = payload.get("self_eval")
        if isinstance(self_eval, dict):
            print(f"self_eval: {self_eval}")
            status = self_eval.get("status")
            if status == "success":
                print("Model reports success. Exiting.")
                break
            if status == "give_up":
                print("Model reports give_up. Exiting.")
                break

        verification_prompt = payload.get("verification_prompt", "")
        if isinstance(verification_prompt, str) and verification_prompt.strip():
            print(f"verification_prompt: {verification_prompt.strip()}")

        try:
            actions = validate_actions(payload.get("actions"))
        except ProtocolError as e:
            print(f"ERROR: Invalid actions: {e}", file=sys.stderr)
            return 4

        if not actions:
            print("No actions returned; continuing to next iteration.")
            continue

        print("Actions to execute:")
        for i, a in enumerate(actions, start=1):
            print(f"  {i}. {a}")

        resp = input("\nExecute these actions now? Type 'yes' to proceed: ").strip().lower()
        if resp != "yes":
            print("Cancelled by user.")
            return 0

        # Conservative policy: if a batch contains a click, ensure we re-observe after any move.
        # Simplest approach: if there is any move before a click, execute only up to the first move,
        # then loop again (new screenshot) before allowing clicks.
        click_idx = next((i for i, a in enumerate(actions) if a["op"] in {"click", "mouse_down", "mouse_up"}), None)
        if click_idx is not None:
            move_before_click = any(a["op"] in {"move", "move_delta"} for a in actions[:click_idx])
            if move_before_click:
                # execute moves only, then re-observe
                exec_actions = [a for a in actions[:click_idx] if a["op"] in {"move", "move_delta"}]
                if exec_actions:
                    print("\nExecuting MOVE-only sub-batch, then re-observing before click...\n")
                    try:
                        ex.execute(exec_actions)
                    finally:
                        try:
                            controls.release_all()
                        except Exception:
                            pass
                    continue

        print("\nExecuting... (press ESC for emergency stop)\n")
        try:
            ex.execute(actions)
        finally:
            try:
                controls.release_all()
            except Exception:
                pass

        # Note: we intentionally do NOT auto-open the screenshot file.

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
