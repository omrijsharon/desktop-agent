"""Action protocol definitions and validation.

This module defines the low-level action schema that the planner produces and
executor consumes.

Design goals:
- Keep schema stable and explicit
- Validate early with clear errors
- Keep it testable (pure functions where possible)
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, TypedDict, cast


# ---- Action types (TypedDicts) ----


class MoveAction(TypedDict):
    op: Literal["move"]
    x: int
    y: int


class MoveDeltaAction(TypedDict):
    op: Literal["move_delta"]
    dx: int
    dy: int


class ClickAction(TypedDict, total=False):
    op: Literal["click"]
    button: Literal["left", "right", "middle"]


class MouseDownAction(TypedDict, total=False):
    op: Literal["mouse_down"]
    button: Literal["left", "right", "middle"]


class MouseUpAction(TypedDict, total=False):
    op: Literal["mouse_up"]
    button: Literal["left", "right", "middle"]


class ScrollAction(TypedDict):
    op: Literal["scroll"]
    dx: int
    dy: int


class KeyDownAction(TypedDict):
    op: Literal["key_down"]
    key: str


class KeyUpAction(TypedDict):
    op: Literal["key_up"]
    key: str


class KeyComboAction(TypedDict):
    op: Literal["key_combo"]
    keys: list[str]


class TypeAction(TypedDict, total=False):
    op: Literal["type"]
    text: str
    delay: float


class ReleaseAllAction(TypedDict):
    op: Literal["release_all"]


class PauseAction(TypedDict):
    op: Literal["pause"]


class ResumeAction(TypedDict):
    op: Literal["resume"]


class StopAction(TypedDict):
    op: Literal["stop"]


Action = (
    MoveAction
    | MoveDeltaAction
    | ClickAction
    | MouseDownAction
    | MouseUpAction
    | ScrollAction
    | KeyDownAction
    | KeyUpAction
    | KeyComboAction
    | TypeAction
    | ReleaseAllAction
    | PauseAction
    | ResumeAction
    | StopAction
)


# ---- Validation ----


class ProtocolError(ValueError):
    """Raised when an action fails validation."""


_MOUSE_BUTTONS = {"left", "right", "middle"}


def _require_int(obj: dict[str, Any], key: str) -> int:
    if key not in obj:
        raise ProtocolError(f"Missing '{key}'")
    v = obj[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise ProtocolError(f"'{key}' must be an int")
    return v


def _require_str(obj: dict[str, Any], key: str) -> str:
    if key not in obj:
        raise ProtocolError(f"Missing '{key}'")
    v = obj[key]
    if not isinstance(v, str):
        raise ProtocolError(f"'{key}' must be a string")
    return v


def validate_action(action: Any) -> Action:
    """Validate a single action and return it typed.

    Raises:
        ProtocolError: if the action is invalid.
    """

    if not isinstance(action, dict):
        raise ProtocolError("Action must be an object")

    op = action.get("op")
    if not isinstance(op, str) or not op:
        raise ProtocolError("Missing 'op'")

    if op == "move":
        _require_int(action, "x")
        _require_int(action, "y")
        return cast(MoveAction, action)

    if op == "move_delta":
        _require_int(action, "dx")
        _require_int(action, "dy")
        return cast(MoveDeltaAction, action)

    if op in {"click", "mouse_down", "mouse_up"}:
        btn = action.get("button", "left")
        if not isinstance(btn, str) or btn not in _MOUSE_BUTTONS:
            raise ProtocolError("'button' must be one of: left/right/middle")
        return cast(ClickAction | MouseDownAction | MouseUpAction, action)

    if op == "scroll":
        # allow defaults, but validate type if present
        dx = action.get("dx", 0)
        dy = action.get("dy", 0)
        if isinstance(dx, bool) or not isinstance(dx, int):
            raise ProtocolError("'dx' must be an int")
        if isinstance(dy, bool) or not isinstance(dy, int):
            raise ProtocolError("'dy' must be an int")
        return cast(ScrollAction, action)

    if op in {"key_down", "key_up"}:
        _require_str(action, "key")
        return cast(KeyDownAction | KeyUpAction, action)

    if op == "key_combo":
        keys = action.get("keys")
        if not isinstance(keys, list) or not keys:
            raise ProtocolError("'keys' must be a non-empty list")
        if not all(isinstance(k, str) and k for k in keys):
            raise ProtocolError("'keys' must be a list of non-empty strings")
        return cast(KeyComboAction, action)

    if op == "type":
        text = action.get("text", "")
        if not isinstance(text, str):
            raise ProtocolError("'text' must be a string")
        delay = action.get("delay", 0.0)
        if not isinstance(delay, (int, float)) or isinstance(delay, bool):
            raise ProtocolError("'delay' must be a number")
        return cast(TypeAction, action)

    if op in {"release_all", "pause", "resume", "stop"}:
        return cast(Action, action)

    raise ProtocolError(f"Unknown op '{op}'")


def validate_actions(actions: Any) -> list[Action]:
    """Validate a list of actions."""

    if not isinstance(actions, list):
        raise ProtocolError("'actions' must be a list")
    return [validate_action(a) for a in actions]
