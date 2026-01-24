from __future__ import annotations

import threading
import time

import pytest

from desktop_agent.executor import Executor, ExecutorConfig
from desktop_agent.llm import LLMPlan
from desktop_agent.planner import (
    EventApprovalGate,
    Planner,
    PlannerConfig,
    PlannerState,
)
from desktop_agent.vision import ScreenRegion, Screenshot


class FakeVision:
    def __init__(self) -> None:
        self.calls = 0

    def capture_fullscreen(self, preview_max_size=None):
        self.calls += 1
        return Screenshot(region=ScreenRegion(0, 0, 100, 100), png_bytes=b"png")


class FakeLLM:
    def __init__(self, plans):
        self._plans = list(plans)
        self.calls = 0

    def plan_next(self, *, goal: str, screenshot_png=None):
        self.calls += 1
        if not self._plans:
            return LLMPlan(high_level="done", actions=[], notes="")
        return self._plans.pop(0)


class StubControls:
    def __init__(self):
        self.calls = []

    def move(self, x: int, y: int) -> None:
        self.calls.append(("move", x, y))

    def click(self, button: str = "left") -> None:
        self.calls.append(("click", button))

    def mouse_down(self, button: str = "left") -> None:
        self.calls.append(("mouse_down", button))

    def mouse_up(self, button: str = "left") -> None:
        self.calls.append(("mouse_up", button))

    def scroll(self, dx: int = 0, dy: int = 0) -> None:
        self.calls.append(("scroll", dx, dy))

    def key_down(self, key: str) -> None:
        self.calls.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", key))

    def key_combo(self, keys):
        self.calls.append(("key_combo", tuple(keys)))

    def type(self, text: str) -> None:
        self.calls.append(("type", text))

    def release_all(self) -> None:
        self.calls.append(("release_all",))


def test_planner_runs_one_iteration_and_executes_actions() -> None:
    vision = FakeVision()
    llm = FakeLLM(
        [
            LLMPlan(
                high_level="move",
                actions=[{"op": "move", "x": 1, "y": 2}],
                notes="",
            )
        ]
    )
    controls = StubControls()
    executor = Executor(controls, ExecutorConfig(max_actions_per_second=0))

    cfg = PlannerConfig(max_iters=1)
    planner = Planner(vision=vision, llm=llm, executor=executor, config=cfg)

    planner.run(goal="x")

    assert vision.calls == 1
    assert llm.calls == 1
    assert ("move", 1, 2) in controls.calls
    assert planner.status.state in {PlannerState.DONE, PlannerState.PLANNING}


def test_planner_step_mode_waits_for_approval() -> None:
    vision = FakeVision()
    llm = FakeLLM(
        [
            LLMPlan(
                high_level="click",
                actions=[{"op": "click", "button": "left"}],
                notes="",
            )
        ]
    )
    controls = StubControls()
    executor = Executor(controls, ExecutorConfig(max_actions_per_second=0))

    gate = EventApprovalGate()
    cfg = PlannerConfig(max_iters=1)
    planner = Planner(
        vision=vision,
        llm=llm,
        executor=executor,
        config=cfg,
        approval_gate=gate,
        step_mode=True,
    )

    t = threading.Thread(target=lambda: planner.run(goal="x"), daemon=True)
    t.start()

    # Wait until it enters WAITING_FOR_USER
    deadline = time.time() + 2
    while time.time() < deadline:
        if planner.status.state == PlannerState.WAITING_FOR_USER:
            break
        time.sleep(0.01)

    assert planner.status.state == PlannerState.WAITING_FOR_USER

    # Still should not have executed.
    assert not any(c[0] == "click" for c in controls.calls)

    gate.approve()
    t.join(timeout=2)
    assert not t.is_alive()
    assert any(c[0] == "click" for c in controls.calls)


def test_planner_rejects_invalid_actions() -> None:
    vision = FakeVision()
    llm = FakeLLM(
        [
            LLMPlan(
                high_level="bad",
                actions=[{"op": "move", "x": "no", "y": 2}],
                notes="",
            )
        ]
    )
    controls = StubControls()
    executor = Executor(controls, ExecutorConfig(max_actions_per_second=0))

    cfg = PlannerConfig(max_iters=1)
    planner = Planner(vision=vision, llm=llm, executor=executor, config=cfg)

    with pytest.raises(Exception):
        planner.run(goal="x")

    assert planner.status.state == PlannerState.ERROR
