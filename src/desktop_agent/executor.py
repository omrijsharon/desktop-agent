"""desktop_agent.executor

Safe execution layer that translates validated protocol actions into calls against
`desktop_agent.controller.WindowsControls`.

MVP goals:
- Never block the UI thread: expose an interruptible `Executor` API that can be
  driven from a worker thread.
- Strong safety: on any error or stop request, ensure `release_all()` is called.
- Cooperative pause/resume/stop suitable for step-mode approvals.

This module is intentionally controller-agnostic: it accepts any object that
provides the small `Controls` surface area used here.

"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Iterable, Optional, Protocol, Sequence

from .protocol import Action, validate_action


class Controls(Protocol):
    """Subset of controller surface used by the executor."""

    def move(self, x: int, y: int) -> None: ...

    def click(self, button: str = "left") -> None: ...

    def mouse_down(self, button: str = "left") -> None: ...

    def mouse_up(self, button: str = "left") -> None: ...

    def scroll(self, dx: int = 0, dy: int = 0) -> None: ...

    def key_down(self, key: str) -> None: ...

    def key_up(self, key: str) -> None: ...

    def key_combo(self, keys: Sequence[str]) -> None: ...

    def type(self, text: str) -> None: ...

    def release_all(self) -> None: ...


class ExecutorStopped(RuntimeError):
    pass


class ExecutorTimeout(RuntimeError):
    pass


ProgressCallback = Callable[[str, Action], None]
ApprovalCallback = Callable[[Sequence[Action]], bool]


@dataclass(frozen=True)
class ExecutorConfig:
    max_actions_per_second: float = 10.0
    batch_timeout_s: Optional[float] = 10.0
    # If set, block execution to coordinates within these bounds.
    # (left, top, right, bottom) – right/bottom are inclusive.
    bounds: Optional[tuple[int, int, int, int]] = None


class Executor:
    """Executes protocol actions with safety controls.

    Designed to be called from a worker thread.
    """

    def __init__(
        self,
        controls: Controls,
        config: ExecutorConfig | None = None,
        *,
        requires_approval: ApprovalCallback | None = None,
        on_progress: ProgressCallback | None = None,
        check_stop: Callable[[], bool] | None = None,
    ) -> None:
        self._controls = controls
        self._cfg = config or ExecutorConfig()
        self._requires_approval = requires_approval
        self._on_progress = on_progress
        self._check_stop = check_stop

        self._pause_evt = threading.Event()
        self._pause_evt.set()  # unpaused
        self._stop_evt = threading.Event()

    # --- lifecycle controls ---

    def pause(self) -> None:
        self._pause_evt.clear()

    def resume(self) -> None:
        self._pause_evt.set()

    def stop(self) -> None:
        self._stop_evt.set()
        self._pause_evt.set()  # unblock any waits
        try:
            self._controls.release_all()
        except Exception:
            # best-effort safety
            pass

    def is_stopped(self) -> bool:
        return self._stop_evt.is_set()

    # --- execution ---

    def execute(self, actions: Iterable[Action]) -> None:
        """Execute a batch of validated protocol actions.

        Raises:
            ExecutorStopped: if stop requested.
            ExecutorTimeout: if `batch_timeout_s` exceeded.
            ValueError/ProtocolError: if an invalid action is passed.
        """

        actions_list = list(actions)
        for a in actions_list:
            validate_action(a)

        if self._requires_approval and self._requires_approval(actions_list):
            # Approval is expected to be handled externally (UI), so we just
            # pause here until resumed/stopped.
            self.pause()

        start = time.monotonic()
        delay = 0.0
        if self._cfg.max_actions_per_second > 0:
            delay = 1.0 / self._cfg.max_actions_per_second

        try:
            for action in actions_list:
                self._wait_if_paused_or_stopped()
                self._check_timeout(start)

                if self._on_progress:
                    self._on_progress("executing", action)

                self._execute_one(action)

                if delay:
                    self._sleep_interruptible(delay)

        except BaseException:
            # Always release on unexpected errors.
            try:
                self._controls.release_all()
            finally:
                raise

    # --- internals ---

    def _wait_if_paused_or_stopped(self) -> None:
        while True:
            if self._stop_evt.is_set() or (self._check_stop and self._check_stop()):
                raise ExecutorStopped("stop requested")
            if self._pause_evt.wait(timeout=0.05):
                return

    def _check_timeout(self, start: float) -> None:
        if self._cfg.batch_timeout_s is None:
            return
        if (time.monotonic() - start) > self._cfg.batch_timeout_s:
            raise ExecutorTimeout("batch timeout")

    def _sleep_interruptible(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._wait_if_paused_or_stopped()
            time.sleep(0.01)

    def _bounds_check(self, x: int, y: int) -> None:
        if self._cfg.bounds is None:
            return
        l, t, r, b = self._cfg.bounds
        if not (l <= x <= r and t <= y <= b):
            raise ValueError(f"move out of bounds: ({x},{y}) not in {self._cfg.bounds}")

    def _execute_one(self, action: Action) -> None:
        op = action["op"]

        if op == "move":
            x = int(action["x"])
            y = int(action["y"])
            self._bounds_check(x, y)
            self._controls.move(x, y)
            return

        if op == "click":
            self._controls.click(action.get("button", "left"))
            return

        if op == "mouse_down":
            self._controls.mouse_down(action.get("button", "left"))
            return

        if op == "mouse_up":
            self._controls.mouse_up(action.get("button", "left"))
            return

        if op == "scroll":
            self._controls.scroll(int(action.get("dx", 0)), int(action.get("dy", 0)))
            return

        if op == "key_down":
            self._controls.key_down(action["key"])
            return

        if op == "key_up":
            self._controls.key_up(action["key"])
            return

        if op == "key_combo":
            self._controls.key_combo(list(action["keys"]))
            return

        if op == "type":
            self._controls.type(action["text"])
            return

        if op == "release_all":
            self._controls.release_all()
            return

        if op == "pause":
            self.pause()
            return

        if op == "resume":
            self.resume()
            return

        if op == "stop":
            self.stop()
            raise ExecutorStopped("stop action")

        raise ValueError(f"unsupported op: {op!r}")
