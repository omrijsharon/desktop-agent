#!/usr/bin/env python3
r"""controller.py

Windows "visible cowork" controller core:

- OS-level mouse/keyboard control via SendInput (ctypes)
- JSON command protocol over stdin/stdout (one JSON object per line)
- Emergency stop: press ESC anytime (global, no window focus required)
  - Releases held inputs
  - Terminates the process *immediately* (hard exit) to avoid hangs
- Optional "step mode": require an explicit {"op":"approve"} before allowing
  the next action(s)

Usage (manual testing, PowerShell):

  # Run interactive server
  python .\controller.py

  # Send a single command
  '{"op":"move","x":300,"y":300}' | python .\controller.py

Protocol:
- One JSON object per line in stdin.
- One JSON response per line in stdout.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
import threading
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

# ---------------------------
# Input backend (SendInput)
# ---------------------------

class MouseButton(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class ControlsConfig:
    default_delay_s: float = 0.0
    clamp_mouse_xy: bool = False  # allow negative coords (left/top monitors)
    # Safety / UX
    step_mode: bool = False  # if True, each action requires approval
    max_actions_without_approval: int = 1  # in step_mode, typically 1


class WindowsControls:
    def __init__(self, config: Optional[ControlsConfig] = None):
        self.config = config or ControlsConfig()
        self._held_keys: set[str] = set()
        self._held_buttons: set[MouseButton] = set()

        self.user32 = ctypes.WinDLL("user32", use_last_error=True)

        # constants
        self.INPUT_MOUSE = 0
        self.INPUT_KEYBOARD = 1

        self.MOUSEEVENTF_MOVE = 0x0001
        self.MOUSEEVENTF_ABSOLUTE = 0x8000
        self.MOUSEEVENTF_LEFTDOWN = 0x0002
        self.MOUSEEVENTF_LEFTUP = 0x0004
        self.MOUSEEVENTF_RIGHTDOWN = 0x0008
        self.MOUSEEVENTF_RIGHTUP = 0x0010
        self.MOUSEEVENTF_MIDDLEDOWN = 0x0020
        self.MOUSEEVENTF_MIDDLEUP = 0x0040
        self.MOUSEEVENTF_WHEEL = 0x0800
        self.MOUSEEVENTF_HWHEEL = 0x01000

        self.KEYEVENTF_KEYUP = 0x0002
        self.KEYEVENTF_UNICODE = 0x0004

        self.SM_XVIRTUALSCREEN = 76
        self.SM_YVIRTUALSCREEN = 77
        self.SM_CXVIRTUALSCREEN = 78
        self.SM_CYVIRTUALSCREEN = 79

        ULONG_PTR = wintypes.WPARAM

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

        class INPUT_UNION(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("union", INPUT_UNION)]

        self.MOUSEINPUT = MOUSEINPUT
        self.KEYBDINPUT = KEYBDINPUT
        self.INPUT = INPUT

        self.SendInput = self.user32.SendInput
        self.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        self.SendInput.restype = wintypes.UINT

        self.GetSystemMetrics = self.user32.GetSystemMetrics
        self.GetSystemMetrics.argtypes = (ctypes.c_int,)
        self.GetSystemMetrics.restype = ctypes.c_int

        self.VK = self._vk_map()

    def _vk_map(self) -> dict[str, int]:
        VK = {
            "backspace": 0x08,
            "tab": 0x09,
            "enter": 0x0D,
            "shift": 0x10,
            "ctrl": 0x11,
            "alt": 0x12,
            "pause": 0x13,
            "capslock": 0x14,
            "esc": 0x1B,
            "space": 0x20,
            "pageup": 0x21,
            "pagedown": 0x22,
            "end": 0x23,
            "home": 0x24,
            "left": 0x25,
            "up": 0x26,
            "right": 0x27,
            "down": 0x28,
            "insert": 0x2D,
            "delete": 0x2E,
            "win": 0x5B,
            "cmd": 0x5B,
        }
        for i in range(1, 25):
            VK[f"f{i}"] = 0x6F + i  # F1=0x70
        for i in range(10):
            VK[str(i)] = 0x30 + i
        for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
            VK[ch] = 0x41 + i
        return VK

    def _norm_key(self, key: str) -> str:
        return key.strip().lower()

    def _delay(self) -> None:
        if self.config.default_delay_s > 0:
            time.sleep(self.config.default_delay_s)

    def _send(self, inp: "WindowsControls.INPUT") -> None:
        sent = self.SendInput(1, ctypes.byref(inp), ctypes.sizeof(self.INPUT))
        if sent != 1:
            raise OSError(f"SendInput failed: {ctypes.get_last_error()}")

    def _send_mouse(self, dx: int, dy: int, flags: int, mouseData: int = 0) -> None:
        inp = self.INPUT()
        inp.type = self.INPUT_MOUSE
        inp.union.mi = self.MOUSEINPUT(dx=dx, dy=dy, mouseData=mouseData, dwFlags=flags, time=0, dwExtraInfo=0)
        self._send(inp)

    def _send_key_vk(self, vk: int, is_up: bool) -> None:
        inp = self.INPUT()
        inp.type = self.INPUT_KEYBOARD
        inp.union.ki = self.KEYBDINPUT(
            wVk=vk, wScan=0, dwFlags=(self.KEYEVENTF_KEYUP if is_up else 0), time=0, dwExtraInfo=0
        )
        self._send(inp)

    def _send_key_unicode(self, code_unit: int, is_up: bool) -> None:
        inp = self.INPUT()
        inp.type = self.INPUT_KEYBOARD
        inp.union.ki = self.KEYBDINPUT(
            wVk=0,
            wScan=code_unit,
            dwFlags=self.KEYEVENTF_UNICODE | (self.KEYEVENTF_KEYUP if is_up else 0),
            time=0,
            dwExtraInfo=0,
        )
        self._send(inp)

    def _virtual_screen(self) -> tuple[int, int, int, int]:
        vx = self.GetSystemMetrics(self.SM_XVIRTUALSCREEN)
        vy = self.GetSystemMetrics(self.SM_YVIRTUALSCREEN)
        vw = self.GetSystemMetrics(self.SM_CXVIRTUALSCREEN)
        vh = self.GetSystemMetrics(self.SM_CYVIRTUALSCREEN)
        return vx, vy, vw, vh

    # ---- Mouse ----

    def move_mouse(self, x: int, y: int) -> None:
        if self.config.clamp_mouse_xy:
            x = max(0, x)
            y = max(0, y)
        vx, vy, vw, vh = self._virtual_screen()
        ax = int((x - vx) * 65535 / max(1, vw - 1))
        ay = int((y - vy) * 65535 / max(1, vh - 1))
        self._send_mouse(ax, ay, self.MOUSEEVENTF_MOVE | self.MOUSEEVENTF_ABSOLUTE)

    def mouse_down(self, button: MouseButton = MouseButton.LEFT) -> None:
        if button == MouseButton.LEFT:
            self._send_mouse(0, 0, self.MOUSEEVENTF_LEFTDOWN)
        elif button == MouseButton.RIGHT:
            self._send_mouse(0, 0, self.MOUSEEVENTF_RIGHTDOWN)
        else:
            self._send_mouse(0, 0, self.MOUSEEVENTF_MIDDLEDOWN)
        self._held_buttons.add(button)

    def mouse_up(self, button: MouseButton = MouseButton.LEFT) -> None:
        if button == MouseButton.LEFT:
            self._send_mouse(0, 0, self.MOUSEEVENTF_LEFTUP)
        elif button == MouseButton.RIGHT:
            self._send_mouse(0, 0, self.MOUSEEVENTF_RIGHTUP)
        else:
            self._send_mouse(0, 0, self.MOUSEEVENTF_MIDDLEUP)
        self._held_buttons.discard(button)

    def click(self, button: MouseButton = MouseButton.LEFT) -> None:
        self.mouse_down(button)
        self._delay()
        self.mouse_up(button)

    def scroll(self, dx: int = 0, dy: int = 0) -> None:
        if dy:
            self._send_mouse(0, 0, self.MOUSEEVENTF_WHEEL, mouseData=int(dy) * 120)
        if dx:
            self._send_mouse(0, 0, self.MOUSEEVENTF_HWHEEL, mouseData=int(dx) * 120)

    # ---- Keyboard ----

    def key_down(self, key: str) -> None:
        k = self._norm_key(key)
        vk = self.VK.get(k)
        if vk is None and len(k) == 1:
            vk = self.VK.get(k.lower())
        if vk is None:
            raise ValueError(f"Unknown key '{key}'. Extend VK map.")
        self._send_key_vk(vk, is_up=False)
        self._held_keys.add(k)

    def key_up(self, key: str) -> None:
        k = self._norm_key(key)
        vk = self.VK.get(k)
        if vk is None and len(k) == 1:
            vk = self.VK.get(k.lower())
        if vk is None:
            raise ValueError(f"Unknown key '{key}'. Extend VK map.")
        self._send_key_vk(vk, is_up=True)
        self._held_keys.discard(k)

    def key_combo(self, keys: Sequence[str]) -> None:
        for k in keys:
            self.key_down(k)
            self._delay()
        for k in reversed(keys):
            self.key_up(k)
            self._delay()

    def type_text(self, text: str, inter_key_delay_s: float = 0.0) -> None:
        # Send UTF-16LE code units (handles surrogate pairs safely)
        units = text.encode("utf-16-le")
        for i in range(0, len(units), 2):
            cu = units[i] | (units[i + 1] << 8)
            self._send_key_unicode(cu, is_up=False)
            self._send_key_unicode(cu, is_up=True)
            if inter_key_delay_s > 0:
                time.sleep(inter_key_delay_s)

    # ---- Safety ----

    def release_all(self) -> None:
        for b in list(self._held_buttons):
            try:
                self.mouse_up(b)
            except Exception:
                pass
        for k in list(self._held_keys):
            try:
                self.key_up(k)
            except Exception:
                pass
        self._held_buttons.clear()
        self._held_keys.clear()


# ---------------------------
# Global hotkey: ESC to stop
# ---------------------------

class EmergencyStop:
    """
    Poll-based ESC detection using GetAsyncKeyState.
    Global, does not require window focus.

    Tradeoff: polling loop, but simple and dependency-free.
    """
    VK_ESCAPE = 0x1B

    def __init__(self, user32: Any, on_stop):
        self.user32 = user32
        self.on_stop = on_stop
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self.GetAsyncKeyState = self.user32.GetAsyncKeyState
        self.GetAsyncKeyState.argtypes = (wintypes.INT,)
        self.GetAsyncKeyState.restype = wintypes.SHORT

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()

    def _run(self) -> None:
        # Detect a key-down state (high bit set)
        while not self._stop_evt.is_set():
            state = self.GetAsyncKeyState(self.VK_ESCAPE)
            if state & 0x8000:
                try:
                    self.on_stop()
                finally:
                    # Ensure we terminate the process even if other threads are blocked.
                    hard_exit(2)
            time.sleep(0.01)


def hard_exit(code: int) -> None:
    """Terminate the process immediately.

    Used for emergency stop to avoid deadlocks caused by blocked threads.
    """
    os._exit(int(code))


# ---------------------------
# JSON command server
# ---------------------------

class CommandError(Exception):
    pass


class ControlsServer:
    def __init__(self, controls: WindowsControls):
        self.c = controls
        self._approved = threading.Event()
        self._approved.set()  # default: not in step mode
        self._pending_actions = 0
        self._stop_evt = threading.Event()

        if self.c.config.step_mode:
            self._approved.clear()

    def stop(self) -> None:
        """Request the server to stop waiting for approvals."""
        self._stop_evt.set()
        self._approved.set()

    def _require_approval_if_needed(self) -> None:
        if not self.c.config.step_mode:
            return

        # Allow N actions after an approval.
        if self._pending_actions <= 0:
            # Don't block forever: loop with timeout so stop/shutdown can interrupt.
            while not self._stop_evt.is_set():
                if self._approved.wait(timeout=0.1):
                    break
            if self._stop_evt.is_set():
                raise CommandError("Server stopping while waiting for approval")

            self._approved.clear()
            self._pending_actions = max(1, self.c.config.max_actions_without_approval)

        self._pending_actions -= 1

    def handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        op = msg.get("op")
        if not op:
            raise CommandError("Missing 'op'")

        # approval is a control-plane command (doesn't need approval)
        if op == "approve":
            self._approved.set()
            return {"ok": True, "op": op}

        if op == "ping":
            return {"ok": True, "op": op}

        if op == "stop":
            self.c.release_all()
            self.stop()
            hard_exit(0)

        # data-plane operations require approval (if step_mode)
        self._require_approval_if_needed()

        # ---- dispatch ----
        if op == "move":
            x = int(msg["x"]); y = int(msg["y"])
            self.c.move_mouse(x, y)
            return {"ok": True, "op": op}

        if op == "click":
            button = MouseButton(msg.get("button", "left"))
            self.c.click(button)
            return {"ok": True, "op": op}

        if op == "mouse_down":
            button = MouseButton(msg.get("button", "left"))
            self.c.mouse_down(button)
            return {"ok": True, "op": op}

        if op == "mouse_up":
            button = MouseButton(msg.get("button", "left"))
            self.c.mouse_up(button)
            return {"ok": True, "op": op}

        if op == "scroll":
            dx = int(msg.get("dx", 0)); dy = int(msg.get("dy", 0))
            self.c.scroll(dx=dx, dy=dy)
            return {"ok": True, "op": op}

        if op == "key_down":
            key = str(msg["key"])
            self.c.key_down(key)
            return {"ok": True, "op": op}

        if op == "key_up":
            key = str(msg["key"])
            self.c.key_up(key)
            return {"ok": True, "op": op}

        if op == "key_combo":
            keys = msg["keys"]
            if not isinstance(keys, list) or not keys:
                raise CommandError("'keys' must be a non-empty list")
            self.c.key_combo([str(k) for k in keys])
            return {"ok": True, "op": op}

        if op == "type":
            text = str(msg.get("text", ""))
            delay = float(msg.get("delay", 0.0))
            self.c.type_text(text, inter_key_delay_s=delay)
            return {"ok": True, "op": op}

        if op == "release_all":
            self.c.release_all()
            return {"ok": True, "op": op}

        raise CommandError(f"Unknown op '{op}'")


def main() -> None:
    # You can flip step_mode=True for "one action per approve"
    cfg = ControlsConfig(
        default_delay_s=0.02,
        clamp_mouse_xy=False,
        step_mode=False,             # set True if you want explicit approve gating
        max_actions_without_approval=1,
    )
    controls = WindowsControls(cfg)
    server = ControlsServer(controls)

    # Emergency stop: ESC
    estop = EmergencyStop(controls.user32, on_stop=lambda: controls.release_all())
    estop.start()

    def write(obj: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    write({"ok": True, "op": "ready", "step_mode": cfg.step_mode})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            if not isinstance(msg, dict):
                raise CommandError("Message must be a JSON object")
            resp = server.handle(msg)
            write(resp)
        except SystemExit:
            raise
        except Exception as e:
            # keep running; return structured error
            write(
                {
                    "ok": False,
                    "op": str(msg.get("op")) if isinstance(locals().get("msg"), dict) else None,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

    # stdin closed -> cleanup
    server.stop()
    controls.release_all()
    estop.stop()


if __name__ == "__main__":
    main()
