from __future__ import annotations

from typing import Any, Iterator


class _FakeUnsupported(Exception):
    def __init__(self, param: str) -> None:
        self.body = {
            "error": {
                "message": f"Unsupported parameter: '{param}' is not supported with this model.",
                "type": "invalid_request_error",
                "param": param,
                "code": None,
            }
        }


class _FakeResp:
    def __init__(self) -> None:
        self.output = []
        self.output_text = "ok"


class _FakeStream:
    def __init__(self, *, fail_first: bool, param: str) -> None:
        self._fail_first = bool(fail_first)
        self._param = str(param)
        self._iterated = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self._iterated and self._fail_first:
            self._iterated = True
            raise _FakeUnsupported(self._param)
        yield {"type": "response.completed", "response": {"output": []}}


class _FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_param = "top_p"
        self.fail_params_seq = ["temperature", "top_p"]
        self.fail_once = True

    def create(self, **kwargs: Any) -> Any:  # noqa: ANN401
        self.calls.append(dict(kwargs))
        if kwargs.get("stream"):
            if self.fail_params_seq:
                p = self.fail_params_seq[0]
                if p in kwargs:
                    self.fail_params_seq.pop(0)
                    raise _FakeUnsupported(p)
            # Fail on first stream iteration only (common SDK behavior).
            s = _FakeStream(fail_first=self.fail_once, param=self.fail_param)
            self.fail_once = False
            return s
        if self.fail_params_seq:
            p = self.fail_params_seq[0]
            if p in kwargs:
                self.fail_params_seq.pop(0)
                raise _FakeUnsupported(p)
        return _FakeResp()


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


def test_run_responses_drops_unsupported_param_and_calls_callback() -> None:
    from desktop_agent.tools import ToolRegistry, run_responses_with_function_tools

    client = _FakeClient()
    client.responses.fail_params_seq = ["top_p"]
    dropped: list[str] = []
    reg = ToolRegistry(tools={}, handlers={})

    _ = run_responses_with_function_tools(
        client=client,
        model="x",
        input_items=[],
        registry=reg,
        on_unsupported_param=lambda p: dropped.append(p),
        top_p=0.9,
    )
    assert dropped == ["top_p"]


def test_run_responses_stream_drops_unsupported_param_when_iterator_raises() -> None:
    from desktop_agent.tools import ToolRegistry, run_responses_with_function_tools_stream

    client = _FakeClient()
    client.responses.fail_params_seq = []
    dropped: list[str] = []
    reg = ToolRegistry(tools={}, handlers={})

    # The fake stream raises on first iteration; runner should retry after dropping.
    events = list(
        run_responses_with_function_tools_stream(
            client=client,
            model="x",
            input_items=[],
            registry=reg,
            on_unsupported_param=lambda p: dropped.append(p),
            top_p=0.9,
        )
    )
    assert dropped == ["top_p"]
    assert any(e.get("type") == "done" for e in events)


def test_run_responses_stream_can_drop_multiple_unsupported_params() -> None:
    from desktop_agent.tools import ToolRegistry, run_responses_with_function_tools_stream

    client = _FakeClient()
    # Override stream failure behavior; we want create() failures for multiple params here.
    client.responses.fail_once = False
    client.responses.fail_param = "top_p"
    client.responses.fail_params_seq = ["temperature", "top_p"]

    dropped: list[str] = []
    reg = ToolRegistry(tools={}, handlers={})

    events = list(
        run_responses_with_function_tools_stream(
            client=client,
            model="x",
            input_items=[],
            registry=reg,
            on_unsupported_param=lambda p: dropped.append(p),
            temperature=0.7,
            top_p=0.9,
        )
    )
    assert dropped == ["temperature", "top_p"]
    assert any(e.get("type") == "done" for e in events)
