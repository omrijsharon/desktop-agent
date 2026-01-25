"""Prompt templates for the LLM.

Why this exists:
- Keep prompts version-controlled and easy to iterate on.
- Make the agent's allowed action space explicit and centralized.

This module intentionally mirrors the validated action protocol in
`desktop_agent.protocol`.

Note: the LLM is *not* trusted. The executor validates + enforces safety.
Prompts are guidance; validation is the gate.
"""

from __future__ import annotations

from .config import SUPPORTED_MODELS


def allowed_actions_text() -> str:
    """Human-readable list of allowed ops.

    Keep this in-sync with `desktop_agent.protocol.Action` and
    `desktop_agent.protocol.validate_action`.
    """

    return (
        "Allowed actions (return ONLY these ops):\n"
        "- move: {op:'move', x:int, y:int}\n"
        "- move_delta: {op:'move_delta', dx:int, dy:int}\n"
        "- click: {op:'click', button:'left'|'right'|'middle'}\n"
        "- mouse_down: {op:'mouse_down', button:'left'|'right'|'middle'}\n"
        "- mouse_up: {op:'mouse_up', button:'left'|'right'|'middle'}\n"
        "- scroll: {op:'scroll', dx:int, dy:int}\n"
        "- key_down: {op:'key_down', key:str}\n"
        "- key_up: {op:'key_up', key:str}\n"
        "- key_combo: {op:'key_combo', keys:[str, ...]} (must be non-empty)\n"
        "- type: {op:'type', text:str, delay:float?}\n"
        "- release_all: {op:'release_all'}\n"
        "- pause: {op:'pause'}\n"
        "- resume: {op:'resume'}\n"
        "- stop: {op:'stop'}\n"
    )


def narrator_prompt() -> str:
    """Prompt for the high-level "intent" stage.

    This stage may use natural language, but must remain conservative and
    screenshot-grounded. It does NOT output executable actions.
    """

    return (
        "You are a Windows desktop automation NARRATOR.\n"
        "Given the user's goal and the latest screenshot, describe the NEXT single intent step in plain language.\n\n"
        "Rules:\n"
        "- Output plain text only (no JSON).\n"
        "- Produce exactly ONE intent sentence, starting with an imperative verb.\n"
        "  Examples: 'Move the cursor slightly left toward the Save button.' / 'Scroll down a little to reveal more results.'\n"
        "- The intent must be grounded in what is visible in the screenshot. If unsure, ask for clarification as a single question instead of guessing.\n"
        "- Prefer conservative, reversible steps.\n"
        "- Mouse policy: if a click is likely needed, state a move intent first and explicitly say to re-observe before clicking.\n"
        "\nCoordinate reminder:\n"
        "- x axis is horizontal (left/right).\n"
        "- y axis is vertical (up/down).\n"
    )


def compiler_prompt() -> str:
    """Prompt for compiling a single intent into safe low-level actions."""

    # We reuse the strict JSON contract and allowed action list.
    return (
        "You are a Windows desktop automation COMPILER.\n"
        "You will be given (1) a plain-language intent step, and (2) the latest screenshot.\n"
        "Your job is to compile that intent into the NEXT small batch of safe UI actions.\n\n"
        "<output_format>\n"
        "- Return ONLY valid JSON. No markdown. No code fences. No commentary.\n"
        "- Return exactly ONE JSON object (not an array).\n"
        "- The JSON object MUST have keys: high_level (string), actions (list), notes (string), self_eval (object), verification_prompt (string).\n"
        "</output_format>\n\n"
        + allowed_actions_text()
        + "\n<strategy_and_safety>\n"
        "- Prefer short action batches (1-6 actions).\n"
        "- IMPORTANT: After your actions are executed, a NEW screenshot will be captured and provided to you.\n"
        "- The screenshot includes a cursor overlay: a realistic white mouse arrow with a black outline.\n"
        "  - The arrow's TOP-LEFT corner is the cursor position (mouse hotspot).\n"
        "- Coordinate system and movement:\n"
        "  - x axis: horizontal (left/right).\n"
        "  - y axis: vertical (up/down).\n"
        "  - move_delta: negative values move in the opposite direction (dx < 0 left; dy < 0 up).\n"
        "- Internal pre-action self-check (DO NOT include this narration in your JSON output):\n"
        "  1) State your intended movement/action in plain language.\n"
        "  2) Map it to a concrete action object.\n"
        "  3) Interpret what the numeric action will do.\n"
        "  4) Ask: 'Is this really what we need based on what is visible?' If not, revise and repeat.\n"
        "- Mouse policy: move first, then on the next screenshot confirm the cursor overlay is over the intended target, and only then click.\n"
        "- If the target is ambiguous or cannot be confirmed, output actions: [] and explain what you need in notes.\n"
        "</strategy_and_safety>\n"
    )


# Back-compat: the "system" prompt used by the current planner loop is the compiler prompt.
# (We keep the name `system_prompt()` to avoid changing many call sites.)

def system_prompt() -> str:
    # ...existing docstring...
    return compiler_prompt()


def model_hint_text() -> str:
    """Optional debug text listing supported model IDs."""

    return "Supported model IDs: " + ", ".join(SUPPORTED_MODELS)
