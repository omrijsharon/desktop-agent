"""Mouse circle test.

Moves the mouse cursor in a circle around the center of the primary screen for:
- duration: 2 seconds
- frequency: 1 Hz (1 full circle per second)
- radius: 100 pixels

This script will generate real mouse movement.
"""

from __future__ import annotations

import math
import time

import ctypes

from desktop_agent.controller import ControlsConfig, WindowsControls


def _screen_center() -> tuple[int, int]:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
    user32.GetSystemMetrics.restype = ctypes.c_int

    SM_CXSCREEN = 0
    SM_CYSCREEN = 1

    w = int(user32.GetSystemMetrics(SM_CXSCREEN))
    h = int(user32.GetSystemMetrics(SM_CYSCREEN))
    return w // 2, h // 2


def main() -> int:
    duration_s = 2.0
    hz = 1.0
    radius_px = 100

    cx, cy = _screen_center()

    controls = WindowsControls(ControlsConfig(default_delay_s=0.0, clamp_mouse_xy=False))

    start = time.perf_counter()
    end = start + duration_s

    # 120 Hz update gives smooth motion without too much overhead.
    dt = 1.0 / 120.0

    try:
        while True:
            now = time.perf_counter()
            if now >= end:
                break

            t = now - start
            angle = 2.0 * math.pi * hz * t

            x = int(round(cx + radius_px * math.cos(angle)))
            y = int(round(cy + radius_px * math.sin(angle)))
            controls.move_mouse(x, y)

            # Sleep until next tick (best-effort)
            time.sleep(dt)
    finally:
        controls.release_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
