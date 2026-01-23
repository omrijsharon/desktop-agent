from desktop_agent import controller


class FakeWindowsControls(controller.WindowsControls):
    def __init__(self):
        # Avoid calling real WinDLL APIs in tests.
        self.config = controller.ControlsConfig()
        self._held_keys: set[str] = set()
        self._held_buttons: set[controller.MouseButton] = set()

    def key_up(self, key: str) -> None:
        self._held_keys.discard(key.strip().lower())

    def mouse_up(self, button: controller.MouseButton = controller.MouseButton.LEFT) -> None:
        self._held_buttons.discard(button)


def test_release_all_clears_bookkeeping():
    c = FakeWindowsControls()
    c._held_keys.update({"ctrl", "alt"})
    c._held_buttons.update({controller.MouseButton.LEFT, controller.MouseButton.RIGHT})

    c.release_all()

    assert c._held_keys == set()
    assert c._held_buttons == set()
