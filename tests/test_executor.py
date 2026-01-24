from __future__ import annotations

import time
import pytest

from desktop_agent.executor import Executor, ExecutorConfig, ExecutorStopped


class StubControls:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def move(self, x: int, y: int) -> None:
        self.calls.append(("move", (x, y)))

    def click(self, button: str = "left") -> None:
        self.calls.append(("click", button))

    def mouse_down(self, button: str = "left") -> None:
        self.calls.append(("mouse_down", button))

    def mouse_up(self, button: str = "left") -> None:
        self.calls.append(("mouse_up", button))

    def scroll(self, dx: int = 0, dy: int = 0) -> None:
        self.calls.append(("scroll", (dx, dy)))

    def key_down(self, key: str) -> None:
        self.calls.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", key))

    def key_combo(self, keys):
        self.calls.append(("key_combo", tuple(keys)))

    def type(self, text: str) -> None:
        self.calls.append(("type", text))

    def release_all(self) -> None:
        self.calls.append(("release_all", None))


def test_executor_executes_actions_in_order() -> None:
    c = StubControls()
    ex = Executor(c, ExecutorConfig(max_actions_per_second=0))
    ex.execute(
        [
            {"op": "move", "x": 10, "y": 20},
            {"op": "click", "button": "left"},
            {"op": "type", "text": "hello"},
        ]
    )
    assert c.calls == [
        ("move", (10, 20)),
        ("click", "left"),
        ("type", "hello"),
    ]


def test_executor_bounds_check() -> None:
    c = StubControls()
    ex = Executor(c, ExecutorConfig(max_actions_per_second=0, bounds=(0, 0, 100, 100)))
    with pytest.raises(ValueError):
        ex.execute([{"op": "move", "x": 200, "y": 20}])
    # release_all should have been called due to exception
    assert ("release_all", None) in c.calls


def test_executor_stop_action_releases_and_raises() -> None:
    c = StubControls()
    ex = Executor(c, ExecutorConfig(max_actions_per_second=0))
    with pytest.raises(ExecutorStopped):
        ex.execute([
            {"op": "move", "x": 1, "y": 2},
            {"op": "stop"},
            {"op": "move", "x": 3, "y": 4},
        ])
    assert c.calls[0] == ("move", (1, 2))
    assert ("release_all", None) in c.calls


def test_executor_pause_resume() -> None:
    c = StubControls()
    ex = Executor(c, ExecutorConfig(max_actions_per_second=0))

    def run():
        ex.execute([
            {"op": "move", "x": 1, "y": 1},
            {"op": "pause"},
            {"op": "move", "x": 2, "y": 2},
        ])

    import threading

    t = threading.Thread(target=run, daemon=True)
    t.start()

    # Wait until pause is hit
    deadline = time.time() + 2
    while time.time() < deadline:
        if ("move", (1, 1)) in c.calls:
            break
        time.sleep(0.01)

    assert ("move", (1, 1)) in c.calls

    # Should still be alive due to pause
    time.sleep(0.05)
    assert t.is_alive()

    ex.resume()
    t.join(timeout=2)
    assert not t.is_alive()
    assert ("move", (2, 2)) in c.calls
