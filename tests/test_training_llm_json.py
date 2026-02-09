from __future__ import annotations

import pytest

from desktop_agent.training.llm_json import TrainingLLMParseError, loads_json_object


def test_loads_json_object_parses_plain_json() -> None:
    obj = loads_json_object('{"a": 1}')
    assert obj["a"] == 1


def test_loads_json_object_extracts_wrapped_json() -> None:
    txt = "Here:\n```json\n{\"x\": \"y\"}\n```"
    obj = loads_json_object(txt)
    assert obj["x"] == "y"


def test_loads_json_object_raises_when_missing() -> None:
    with pytest.raises(TrainingLLMParseError):
        loads_json_object("no json here")

