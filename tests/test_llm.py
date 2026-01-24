from __future__ import annotations

import pytest

from desktop_agent.llm import LLMParseError, PlannerLLM


class FakeClient:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def responses_create(self, *, model: str, input, **kwargs):
        self.calls += 1
        if not self._outputs:
            raise RuntimeError("no more outputs")
        out = self._outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_llm_parses_valid_json_and_validates_actions() -> None:
    client = FakeClient(
        [
            '{"high_level":"Test","actions":[{"op":"move","x":1,"y":2}],"notes":"ok"}'
        ]
    )
    llm = PlannerLLM(client=client)
    plan = llm.plan_next(goal="do thing", screenshot_png=None)
    assert plan.high_level == "Test"
    assert plan.notes == "ok"
    assert plan.actions[0]["op"] == "move"


def test_llm_extracts_json_from_wrapped_text() -> None:
    client = FakeClient(
        [
            'Here you go:\n```json\n{"high_level":"X","actions":[{"op":"click"}],"notes":""}\n```'
        ]
    )
    llm = PlannerLLM(client=client)
    plan = llm.plan_next(goal="x")
    assert plan.high_level == "X"
    assert plan.actions[0]["op"] == "click"


def test_llm_retries_on_parse_error() -> None:
    client = FakeClient(
        [
            "not json",
            '{"high_level":"Ok","actions":[{"op":"type","text":"hi"}],"notes":""}',
        ]
    )
    llm = PlannerLLM(client=client)
    plan = llm.plan_next(goal="x")
    assert plan.high_level == "Ok"
    assert client.calls == 2


def test_llm_raises_on_invalid_actions() -> None:
    client = FakeClient(
        [
            '{"high_level":"Bad","actions":[{"op":"move","x":"no","y":2}],"notes":""}'
        ]
    )
    llm = PlannerLLM(client=client)
    with pytest.raises(LLMParseError):
        llm.plan_next(goal="x")
