"""desktop_agent.llm

LLM adapter for MVP v0.

Goal:
- Given a user goal and (optionally) a screenshot (PNG bytes), return a strict
  JSON plan:

  {
    "high_level": "...",
    "actions": [ {"op": ...}, ... ],
    "notes": "..."  # optional
  }

Key properties:
- JSON-only parsing with retry.
- Validate actions using `desktop_agent.protocol.validate_actions`.
- Keep this module testable by injecting a small client interface.

This module does *not* implement the planner loop; it is a pure “plan next
actions” call.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

from .protocol import Action, ProtocolError, validate_actions
from .prompts import system_prompt


# ---- Public types ----


@dataclass(frozen=True)
class LLMPlan:
    high_level: str
    actions: list[Action]
    notes: str = ""


class LLMError(RuntimeError):
    pass


class LLMParseError(LLMError):
    pass


class LLMClient(Protocol):
    """Minimal client surface for dependency injection / mocking."""

    def responses_create(self, *, model: str, input: Any, **kwargs: Any) -> str:
        """Return the model's text output (JSON string)."""


# ---- OpenAI implementation ----


class OpenAIResponsesClient:
    """Adapter over the `openai` SDK Responses API.

    This wrapper exists so tests can inject a fake client without importing
    `openai`.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        from openai import OpenAI  # imported lazily

        self._client = OpenAI(api_key=api_key)

    def responses_create(self, *, model: str, input: Any, **kwargs: Any) -> str:
        resp = self._client.responses.create(model=model, input=input, **kwargs)
        # The SDK provides a convenience accessor with the concatenated text.
        txt = getattr(resp, "output_text", None)
        if not isinstance(txt, str):
            raise LLMError("OpenAI response missing output_text")
        return txt


class FakeLLMClient:
    """Offline fake LLM client.

    Returns deterministic, valid JSON plans so the UI/loop can be demoed without
    an API key.
    """

    def responses_create(self, *, model: str, input: Any, **kwargs: Any) -> str:  # noqa: ARG002
        # Extract the goal text from the OpenAI-style input payload.
        goal = ""
        try:
            for msg in input:
                if msg.get("role") != "user":
                    continue
                for part in msg.get("content", []):
                    if part.get("type") == "input_text":
                        txt = str(part.get("text", ""))
                        if txt.startswith("Goal:"):
                            goal = txt[len("Goal:") :].strip()
        except Exception:
            goal = ""

        # Provide a very small, safe action batch. The point is UI visibility.
        # Keep it mostly no-op: clicks/typing are real inputs and could be risky.
        # We only move the mouse slightly to indicate activity.
        plan = {
            "high_level": f"(FAKE MODE) Would work on: {goal or 'your request'}",
            "actions": [
                {"op": "move", "x": 10, "y": 10},
                {"op": "move", "x": 20, "y": 20},
                {"op": "release_all"},
            ],
            "notes": "FAKE MODE: No API key detected. This is a canned response.",
        }
        return json.dumps(plan)


# ---- Prompting / parsing ----


# Centralized in `desktop_agent.prompts`.
_SYSTEM_PROMPT = system_prompt()


def _build_openai_input(*, goal: str, screenshot_png: Optional[bytes]) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [{"type": "input_text", "text": f"Goal: {goal}"}]

    if screenshot_png:
        b64 = base64.b64encode(screenshot_png).decode("ascii")
        user_content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
            }
        )

    return [
        {"role": "system", "content": [{"type": "input_text", "text": _SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def _extract_json(text: str) -> str:
    """Extract JSON object from a model response.

    We first try strict parse. If that fails, try to locate the first JSON
    object by scanning braces.
    """

    s = text.strip()
    # Fast path: already JSON
    if s.startswith("{") and s.endswith("}"):
        return s

    # Try to find a JSON object in the text.
    start = s.find("{")
    if start < 0:
        raise LLMParseError("No JSON object found")

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    raise LLMParseError("Unterminated JSON object")


def _parse_plan_json(text: str) -> LLMPlan:
    try:
        payload = json.loads(_extract_json(text))
    except (json.JSONDecodeError, LLMParseError) as e:
        raise LLMParseError(str(e)) from e

    if not isinstance(payload, dict):
        raise LLMParseError("JSON must be an object")

    high_level = payload.get("high_level", "")
    if not isinstance(high_level, str) or not high_level.strip():
        raise LLMParseError("'high_level' must be a non-empty string")

    try:
        actions = validate_actions(payload.get("actions"))
    except ProtocolError as e:
        raise LLMParseError(f"Invalid actions: {e}") from e

    notes = payload.get("notes", "")
    if notes is None:
        notes = ""
    if not isinstance(notes, str):
        raise LLMParseError("'notes' must be a string")

    return LLMPlan(high_level=high_level.strip(), actions=actions, notes=notes)


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    max_retries: int = 2
    # If true, force fake mode regardless of API key.
    fake_mode: bool = False


class PlannerLLM:
    """High-level façade used by the planner."""

    def __init__(
        self,
        *,
        client: Optional[LLMClient] = None,
        config: Optional[LLMConfig] = None,
    ) -> None:
        self._cfg = config or LLMConfig()

        if client is not None:
            self._client = client
        else:
            api_key = os.environ.get(self._cfg.api_key_env)

            # Python equivalent of a compile-time flag:
            # - env var `DESKTOP_AGENT_FAKE_LLM=1` forces fake
            # - missing API key defaults to fake so the UI is still demoable
            force_fake = os.environ.get("DESKTOP_AGENT_FAKE_LLM", "0") in {"1", "true", "True"}

            if self._cfg.fake_mode or force_fake or not api_key:
                self._client = FakeLLMClient()
            else:
                self._client = OpenAIResponsesClient(api_key=api_key)

    def plan_next(self, *, goal: str, screenshot_png: Optional[bytes] = None) -> LLMPlan:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")

        last_err: Optional[Exception] = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                inp = _build_openai_input(goal=goal, screenshot_png=screenshot_png)

                # Ask for strict JSON. The Responses API supports `text.format`.
                txt = self._client.responses_create(
                    model=self._cfg.model,
                    input=inp,
                    text={"format": {"type": "json_object"}},
                )
                return _parse_plan_json(txt)
            except LLMParseError as e:
                # If we got JSON but the *schema/actions* are invalid, retrying
                # won't help; fail fast.
                msg = str(e).lower()
                if msg.startswith("invalid actions") or "high_level" in msg or "actions" in msg or "notes" in msg:
                    raise
                last_err = e
                if attempt < self._cfg.max_retries:
                    continue
                raise
            except LLMError as e:
                last_err = e
                if attempt < self._cfg.max_retries:
                    continue
                raise
            except Exception as e:
                # Unexpected client errors (network, auth, etc.)
                last_err = e
                if attempt < self._cfg.max_retries:
                    continue
                raise LLMError(str(e)) from e

        # Unreachable
        raise LLMError(f"LLM planning failed: {last_err}")
