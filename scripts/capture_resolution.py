"""Print the current virtual-screen (all monitors) resolution.

This uses the same `ScreenCapture` service as the agent, which captures the
entire virtual desktop (all monitors combined).

Run (PowerShell):
  .\\.venv\\Scripts\\python scripts\\capture_resolution.py
"""

from __future__ import annotations

from desktop_agent.vision import ScreenCapture


def main() -> int:
    cap = ScreenCapture()

    shot = cap.capture_fullscreen(preview_max_size=None)
    w, h = shot.region.width, shot.region.height

    print(f"Virtual screen resolution: {w}x{h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
