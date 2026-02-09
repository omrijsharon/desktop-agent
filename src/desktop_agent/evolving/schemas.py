from __future__ import annotations

from typing import Any, Literal, TypedDict


Gender = Literal["male", "female"]


class AgentAction(TypedDict, total=False):
    actor_id: str
    op: str
    target_id: str
    text: str
    args: dict[str, Any]
    private: bool


class NarratorOutcome(TypedDict, total=False):
    ok: bool
    reason: str
    public_message: str
    private_message: str
    actor_energy_delta: float
    target_energy_delta: float
    actor_health_delta: float
    target_health_delta: float
    interaction: dict[str, Any]
    created_child: dict[str, Any]


def _req_str(obj: dict[str, Any], key: str) -> str:
    v = obj.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return v.strip()


def validate_action(payload: Any) -> AgentAction:
    if not isinstance(payload, dict):
        raise ValueError("action must be an object")
    op = _req_str(payload, "op")
    actor_id = payload.get("actor_id")
    if actor_id is not None and (not isinstance(actor_id, str) or not actor_id.strip()):
        raise ValueError("'actor_id' must be a string if provided")
    target_id = payload.get("target_id")
    if target_id is not None and (not isinstance(target_id, str) or not target_id.strip()):
        raise ValueError("'target_id' must be a string if provided")
    if "args" in payload and not isinstance(payload.get("args"), dict):
        raise ValueError("'args' must be an object if provided")
    if "private" in payload and not isinstance(payload.get("private"), bool):
        raise ValueError("'private' must be a bool if provided")
    return AgentAction(payload | {"op": op})


def validate_outcome(payload: Any) -> NarratorOutcome:
    if not isinstance(payload, dict):
        raise ValueError("outcome must be an object")
    if "ok" in payload and not isinstance(payload.get("ok"), bool):
        raise ValueError("'ok' must be a bool if provided")
    for k in ("actor_energy_delta", "target_energy_delta", "actor_health_delta", "target_health_delta"):
        if k in payload:
            v = payload.get(k)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"'{k}' must be a number if provided")
    # Models may emit JSON null. Treat null as "not provided".
    if "interaction" in payload:
        inter = payload.get("interaction")
        if inter is not None and not isinstance(inter, dict):
            raise ValueError("'interaction' must be an object or null if provided")
    if "created_child" in payload:
        cc = payload.get("created_child")
        if cc is not None and not isinstance(cc, dict):
            raise ValueError("'created_child' must be an object or null if provided")
    return NarratorOutcome(payload)
