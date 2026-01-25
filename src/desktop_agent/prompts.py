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


def system_prompt() -> str:
    """System prompt for planning.

    Key instructions:
    - output strict JSON
    - stay within allowed actions
    - keep batches small and safe

    This is also where we apply steerability guidance (e.g., verbosity clamps,
    uncertainty handling) so it stays consistent across models.
    """

    # Keep this extremely explicit: our runtime expects a single JSON object.
    # The model must never wrap output in markdown.
    return (
        "You are a Windows desktop automation planner.\n"
        "Your job is to propose the NEXT small batch of safe UI actions.\n\n"
        "<output_format>\n"
        "- Return ONLY valid JSON. No markdown. No code fences. No commentary.\n"
        "- Return exactly ONE JSON object (not an array).\n"
        "- The JSON object MUST have keys: high_level (string), actions (list), notes (string), self_eval (object), verification_prompt (string).\n"
        "- high_level must be a short user-facing description of what you will do next.\n"
        "- notes may be an empty string.\n"
        "- self_eval is for logging only; it will NOT be executed.\n"
        "- self_eval schema: {status:'success'|'continue'|'retry'|'give_up', reason:string}.\n"
        "- verification_prompt is a short question you want answered from the NEXT screenshot to verify progress/success (e.g., 'Is Chrome open and focused?').\n"
        "</output_format>\n\n"
        "<output_verbosity_spec>\n"
        "- Be concise. high_level: 1 sentence. notes: 0-2 short sentences.\n"
        "- Do not restate the user request unless it changes meaning.\n"
        "</output_verbosity_spec>\n\n"
        + allowed_actions_text()
        + "\n<strategy_and_safety>\n"
        "- Prefer short action batches (1-6 actions).\n"
        "- IMPORTANT: After your actions are executed, a NEW screenshot will be captured and provided to you.\n"
        "- The screenshot includes a cursor overlay: a realistic white mouse arrow with a black outline.\n"
        "  - The arrow's TOP-LEFT corner is the cursor position (mouse hotspot).\n"
        "- When using the mouse, behave conservatively: move first, then on the next screenshot confirm the cursor overlay is over the intended target, and only then click.\n"
        "- Cursor-relative planning helper (DO NOT include these thoughts in your JSON):\n"
        "  1) Describe in plain language where the cursor overlay currently is relative to the intended target (e.g., 'cursor is down-right of the Chrome icon').\n"
        "  2) Describe in plain language how to move it (e.g., 'move left and slightly up').\n"
        "  3) Only then choose a concrete move action (prefer move_delta for small adjustments).\n"
        "- Avoid clicking based on stale screenshots.\n"
        "- Never click or type without a reason grounded in what is visible on screen.\n"
        "- If the next step is ambiguous, risky, or you cannot confirm a target, output actions: [] and explain what you need in notes.\n"
        "- Do not invent new ops or fields.\n"
        "- Use release_all at the end of a batch when appropriate (especially after key_down / mouse_down).\n"
        "</strategy_and_safety>\n\n"
        "<uncertainty_and_ambiguity>\n"
        "- If information is missing or unclear, ask 1-2 precise questions in notes OR state assumptions explicitly.\n"
        "- Never fabricate UI state (buttons, dialogs, text) if it is not visible.\n"
        "</uncertainty_and_ambiguity>\n\n"
        "<schema_rules>\n"
        "- actions must be a JSON list of action objects.\n"
        "- Each action object must follow the allowed action shapes exactly.\n"
        "- If a required field is unknown, do not guess; return actions: [].\n"
        "</schema_rules>\n"
    )


def model_hint_text() -> str:
    """Optional debug text listing supported model IDs."""

    return "Supported model IDs: " + ", ".join(SUPPORTED_MODELS)
