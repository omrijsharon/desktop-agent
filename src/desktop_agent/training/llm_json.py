from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Protocol


class TrainingLLMError(RuntimeError):
    pass


class TrainingLLMParseError(TrainingLLMError):
    pass


class ResponsesClient(Protocol):
    def responses_create(self, *, model: str, input: Any, **kwargs: Any) -> str: ...


def _extract_json(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("{") and s.endswith("}"):
        return s

    start = s.find("{")
    if start < 0:
        raise TrainingLLMParseError("No JSON object found")

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

    raise TrainingLLMParseError("Unterminated JSON object")


def loads_json_object(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(_extract_json(text))
    except (json.JSONDecodeError, TrainingLLMParseError) as e:
        raise TrainingLLMParseError(str(e)) from e
    if not isinstance(obj, dict):
        raise TrainingLLMParseError("JSON must be an object")
    return obj


@dataclass(frozen=True)
class JSONCallConfig:
    model: str
    max_retries: int = 2
    temperature: Optional[float] = None


def build_messages(*, system: str, user: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "input_text", "text": system}]},
        {"role": "user", "content": [{"type": "input_text", "text": user}]},
    ]


def call_json_object(
    *,
    client: ResponsesClient,
    cfg: JSONCallConfig,
    system: str,
    user: str,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if not system.strip():
        raise ValueError("system must be non-empty")
    if not user.strip():
        raise ValueError("user must be non-empty")

    extra_kwargs = dict(extra_kwargs or {})
    last_err: Exception | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": cfg.model,
                "input": build_messages(system=system, user=user),
                "truncation": "auto",
                "text": {"format": {"type": "json_object"}},
            }
            if cfg.temperature is not None:
                kwargs["temperature"] = float(cfg.temperature)
            kwargs.update(extra_kwargs)

            txt = client.responses_create(**kwargs)
            return loads_json_object(txt)
        except TrainingLLMParseError as e:
            last_err = e
            if attempt < cfg.max_retries:
                continue
            raise
        except Exception as e:
            last_err = e
            if attempt < cfg.max_retries:
                continue
            raise TrainingLLMError(f"LLM call failed: {e}") from e

    raise TrainingLLMError(f"LLM call failed: {last_err}")

