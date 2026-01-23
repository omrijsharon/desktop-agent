import pytest

import controller


class FakeControls:
    def __init__(self, step_mode: bool = False, max_actions_without_approval: int = 1):
        self.config = controller.ControlsConfig(
            default_delay_s=0.0,
            clamp_mouse_xy=False,
            step_mode=step_mode,
            max_actions_without_approval=max_actions_without_approval,
        )
        self.calls: list[tuple[str, object]] = []

    def release_all(self) -> None:
        self.calls.append(("release_all", None))

    def move_mouse(self, x: int, y: int) -> None:
        self.calls.append(("move_mouse", (x, y)))

    def click(self, button: controller.MouseButton) -> None:
        self.calls.append(("click", button))

    def mouse_down(self, button: controller.MouseButton) -> None:
        self.calls.append(("mouse_down", button))

    def mouse_up(self, button: controller.MouseButton) -> None:
        self.calls.append(("mouse_up", button))

    def scroll(self, dx: int = 0, dy: int = 0) -> None:
        self.calls.append(("scroll", (dx, dy)))

    def key_down(self, key: str) -> None:
        self.calls.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", key))

    def key_combo(self, keys):
        self.calls.append(("key_combo", list(keys)))

    def type_text(self, text: str, inter_key_delay_s: float = 0.0) -> None:
        self.calls.append(("type_text", (text, inter_key_delay_s)))


def test_ping_ok():
    c = FakeControls(step_mode=False)
    s = controller.ControlsServer(c)  # type: ignore[arg-type]
    assert s.handle({"op": "ping"}) == {"ok": True, "op": "ping"}


def test_approve_ok():
    c = FakeControls(step_mode=True)
    s = controller.ControlsServer(c)  # type: ignore[arg-type]
    assert s.handle({"op": "approve"}) == {"ok": True, "op": "approve"}


def test_release_all_ok():
    c = FakeControls(step_mode=False)
    s = controller.ControlsServer(c)  # type: ignore[arg-type]
    assert s.handle({"op": "release_all"}) == {"ok": True, "op": "release_all"}
    assert ("release_all", None) in c.calls


def test_key_combo_validation():
    c = FakeControls(step_mode=False)
    s = controller.ControlsServer(c)  # type: ignore[arg-type]
    with pytest.raises(controller.CommandError):
        s.handle({"op": "key_combo", "keys": []})


def test_step_mode_blocks_without_approval(monkeypatch):
    c = FakeControls(step_mode=True)
    s = controller.ControlsServer(c)  # type: ignore[arg-type]

    # If approval is required, a call that would wait should be interruptible
    # by stop. We simulate it by calling stop before issuing an action.
    s.stop()
    with pytest.raises(controller.CommandError):
        s.handle({"op": "move", "x": 1, "y": 2})


def test_step_mode_allows_after_approval():
    c = FakeControls(step_mode=True, max_actions_without_approval=1)
    s = controller.ControlsServer(c)  # type: ignore[arg-type]

    assert s.handle({"op": "approve"})["ok"] is True
    assert s.handle({"op": "move", "x": 10, "y": 20}) == {"ok": True, "op": "move"}
    assert ("move_mouse", (10, 20)) in c.calls
