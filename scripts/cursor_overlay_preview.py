r"""scripts/cursor_overlay_preview.py

Capture a screenshot of the full virtual desktop and overlay a *realistic* cursor
marker (white arrow with black outline) at the current mouse position.

Then open the resulting image so you can visually verify the cursor overlay.

Usage (PowerShell):
  .\\.venv\\Scripts\\python .\\scripts\\cursor_overlay_preview.py

Output:
  scripts/cursor_overlay_preview.png

Notes:
- This uses `desktop_agent.vision.ScreenCapture(..., include_cursor=True)`.
- The cursor marker is drawn in-process; it is not the OS cursor bitmap.
"""

from __future__ import annotations

import os

from desktop_agent.vision import ScreenCapture


def main() -> int:
    cap = ScreenCapture()
    shot = cap.capture_fullscreen(preview_max_size=None, include_cursor=True)

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "cursor_overlay_preview.png"))
    with open(out_path, "wb") as f:
        f.write(shot.png_bytes)

    print(f"Wrote: {out_path}")

    # Open the image in the default image viewer (Windows)
    os.startfile(out_path)  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
