"""desktop_agent.planner

Planner loop (observe → plan → act) for MVP v0.

Design:
- Pure, testable state machine + loop runner.
- No UI code here. UI can subscribe via callbacks.
- Uses:
  - `ScreenCapture` from `desktop_agent.vision`
  - `PlannerLLM` from `desktop_agent.llm`
  - `Executor` from `desktop_agent.executor`

Safety:
- Stop should be cooperative and immediate.
- On stop/error: ensure `executor.stop()` is called (which releases inputs).

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Callable, Optional, Protocol

from .executor import Executor, ExecutorStopped
from .llm import LLMPlan, PlannerLLM
from .protocol import Action, ProtocolError, validate_actions
from .vision import ScreenCapture, Screenshot


class PlannerState(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    DONE = "DONE"
    ERROR = "ERROR"


@dataclass
class PlannerStatus:
    state: PlannerState = PlannerState.IDLE
    high_level: str = ""
    last_error: str = ""
    step_mode: bool = False


@dataclass
class PlannerHistoryItem:
    ts: float
    high_level: str
    actions: list[Action]


@dataclass
class PlannerConfig:
    max_iters: int = 50
    # Small batches keep it safe; LLM can return multiple actions.
    max_actions_per_batch: int = 10
    # Time between iterations when no actions are returned.
    idle_sleep_s: float = 0.2


StatusCallback = Callable[[PlannerStatus], None]
MessageCallback = Callable[[str], None]
ScreenshotCallback = Callable[[Screenshot], None]


class ApprovalGate(Protocol):
    def requires_approval(self, actions: list[Action]) -> bool: ...

    def wait_for_approval(self, timeout_s: float) -> bool: ...


class EventApprovalGate:
    """A simple approval gate using a threading.Event.

    UI can call `approve()` to release a batch.
    """

    def __init__(self) -> None:
        self._evt = threading.Event()

    def requires_approval(self, actions: list[Action]) -> bool:
        return True

    def approve(self) -> None:
        self._evt.set()

    def wait_for_approval(self, timeout_s: float) -> bool:
        return self._evt.wait(timeout=timeout_s)

    def reset(self) -> None:
        self._evt.clear()


class Planner:
    def __init__(
        self,
        *,
        vision: ScreenCapture,
        llm: PlannerLLM,
        executor: Executor,
        config: PlannerConfig | None = None,
        approval_gate: ApprovalGate | None = None,
        step_mode: bool = False,
        on_status: StatusCallback | None = None,
        on_message: MessageCallback | None = None,
        on_screenshot: ScreenshotCallback | None = None,
    ) -> None:
        self._vision = vision
        self._llm = llm
        self._executor = executor
        self._cfg = config or PlannerConfig()
        self._approval_gate = approval_gate

        self._status = PlannerStatus(step_mode=step_mode)
        self._on_status = on_status
        self._on_message = on_message
        self._on_screenshot = on_screenshot

        self._stop_evt = threading.Event()
        self._history: list[PlannerHistoryItem] = []

    @property
    def status(self) -> PlannerStatus:
        return self._status

    @property
    def history(self) -> list[PlannerHistoryItem]:
        return list(self._history)

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            self._executor.stop()
        except Exception:
            pass

    def is_stopped(self) -> bool:
        return self._stop_evt.is_set()

    def run(self, *, goal: str) -> None:
        """Run the planner loop synchronously (call from a worker thread)."""

        self._set_state(PlannerState.PLANNING, high_level="Starting")

        try:
            for _ in range(self._cfg.max_iters):
                self._check_stop()

                # --- Observe ---
                shot = self._vision.capture_fullscreen(preview_max_size=(512, 512))
                if self._on_screenshot:
                    self._on_screenshot(shot)

                # --- Plan ---
                self._set_state(PlannerState.PLANNING, high_level="Planning next steps")
                plan = self._llm.plan_next(goal=goal, screenshot_png=shot.png_bytes)

                # Validate (defense in depth)
                actions = validate_actions(plan.actions)

                self._status.high_level = plan.high_level
                self._emit_message(plan.high_level)

                if not actions:
                    time.sleep(self._cfg.idle_sleep_s)
                    continue

                # Clamp batch size.
                actions = actions[: self._cfg.max_actions_per_batch]

                self._history.append(
                    PlannerHistoryItem(ts=time.time(), high_level=plan.high_level, actions=list(actions))
                )

                # --- Optional step approval ---
                if self._status.step_mode and self._approval_gate is not None:
                    if self._approval_gate.requires_approval(list(actions)):
                        self._set_state(PlannerState.WAITING_FOR_USER, high_level=plan.high_level)
                        self._emit_message("Waiting for approval")
                        if isinstance(self._approval_gate, EventApprovalGate):
                            self._approval_gate.reset()

                        while True:
                            self._check_stop()
                            if self._approval_gate.wait_for_approval(timeout_s=0.1):
                                break

                # --- Act ---
                self._set_state(PlannerState.EXECUTING, high_level=plan.high_level)
                self._executor.execute(actions)

            self._set_state(PlannerState.DONE)

        except ExecutorStopped:
            # Cooperative stop
            self._set_state(PlannerState.DONE)
            return
        except (ProtocolError, ValueError) as e:
            self._set_error(str(e))
            raise
        except Exception as e:
            self._set_error(str(e))
            raise
        finally:
            try:
                self._executor.stop()
            except Exception:
                pass

    # ---- internals ----

    def _emit_message(self, msg: str) -> None:
        if self._on_message:
            self._on_message(msg)

    def _set_state(self, state: PlannerState, *, high_level: str | None = None) -> None:
        self._status.state = state
        if high_level is not None:
            self._status.high_level = high_level
        self._status.last_error = ""
        if self._on_status:
            self._on_status(self._status)

    def _set_error(self, msg: str) -> None:
        self._status.state = PlannerState.ERROR
        self._status.last_error = msg
        if self._on_status:
            self._on_status(self._status)

    def _check_stop(self) -> None:
        if self._stop_evt.is_set():
            raise ExecutorStopped("planner stopped")
