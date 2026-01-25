"""Vision subsystem (screenshot capture + image utilities).

MVP v0 requirements (see project.md / MVP_v0_plan.md):
- capture full-screen screenshots (~1–5 FPS)
- provide raw bytes (PNG/JPEG) and optional scaled preview
- provide coordinate conversion helpers (scaled preview <-> real screen)

Implementation uses `mss` for capture and Pillow for image representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ScreenRegion:
    """A rectangular region in real screen coordinates."""

    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class Screenshot:
    """A captured screenshot plus metadata."""

    region: ScreenRegion
    png_bytes: bytes
    preview_png_bytes: Optional[bytes] = None
    preview_size: Optional[Tuple[int, int]] = None


def map_point_real_to_preview(
    x: int,
    y: int,
    real_region: ScreenRegion,
    preview_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Map a point from real screen coords into preview pixel coords."""

    pw, ph = preview_size
    rx = (x - real_region.left) / max(1, real_region.width)
    ry = (y - real_region.top) / max(1, real_region.height)
    return int(round(rx * pw)), int(round(ry * ph))


def map_point_preview_to_real(
    px: int,
    py: int,
    real_region: ScreenRegion,
    preview_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Map a point from preview pixel coords back into real screen coords."""

    pw, ph = preview_size
    rx = px / max(1, pw)
    ry = py / max(1, ph)
    x = int(round(real_region.left + rx * real_region.width))
    y = int(round(real_region.top + ry * real_region.height))
    return x, y


def _get_cursor_pos_win32() -> Optional[Tuple[int, int]]:
    """Return the current cursor position in *virtual screen* coordinates.

    Uses Win32 GetCursorPos, which returns screen coordinates in the unified
    virtual desktop space.

    Returns None if not on Windows or if the call fails.
    """

    try:
        import ctypes

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)) == 0:
            return None
        return int(pt.x), int(pt.y)
    except Exception:
        return None


def _draw_cursor_marker(img: "Image.Image", *, x: int, y: int) -> None:
    """Draw a mouse cursor marker onto a Pillow image (in-place).

    The marker is a simple arrow-like white triangle with a black outline.

    Anchor convention:
    - (x, y) is the *hotspot*.
    - The drawn cursor's bounding box top-left coincides with (x, y).

    This intentionally does not attempt to match the exact OS cursor bitmap;
    it provides a consistent, recognizable cue for the model.
    """

    from PIL import ImageDraw  # type: ignore

    draw = ImageDraw.Draw(img)

    # Cursor geometry (approximate). (x, y) is top-left of bbox.
    w, h = 22, 32

    # A simple arrow/triangle shape with a small tail notch.
    # Points are relative to (x, y).
    pts = [
        (x + 0, y + 0),
        (x + 0, y + h),
        (x + 8, y + 24),
        (x + 13, y + 34),
        (x + 17, y + 32),
        (x + 12, y + 22),
        (x + w, y + 18),
    ]

    # Draw outline and fill. Use black outline for readability.
    outline = (0, 0, 0)
    fill = (255, 255, 255)

    # Pillow versions vary in polygon outline support; draw outline by
    # stroking the edges as lines, then fill.
    draw.polygon(pts, fill=fill)
    # Outline via connected lines
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        draw.line([(x1, y1), (x2, y2)], fill=outline, width=2)


class ScreenCapture:
    """Capture screenshots using mss.

    Note: Implemented in a way that can be mocked in tests.
    """

    def __init__(self):
        # Imported lazily so unit tests can run without mss installed.
        import mss  # type: ignore

        self._mss = mss.mss()

    def capture_fullscreen(
        self,
        preview_max_size: Optional[Tuple[int, int]] = (512, 512),
        *,
        include_cursor: bool = False,
    ) -> Screenshot:
        """Capture the full virtual screen.

        Args:
            preview_max_size: if provided, a scaled down PNG preview is returned.
            include_cursor: if true, draws a simple cursor marker at the current
                mouse position onto the screenshot.

        Returns:
            Screenshot: PNG bytes + metadata.
        """

        # mss.monitors[0] is the full virtual screen.
        mon = self._mss.monitors[0]
        region = ScreenRegion(left=int(mon["left"]), top=int(mon["top"]), width=int(mon["width"]), height=int(mon["height"]))

        raw = self._mss.grab(mon)

        # Convert raw BGRA to PNG via Pillow.
        from PIL import Image  # type: ignore

        img = Image.frombytes("RGB", raw.size, raw.rgb)

        if include_cursor:
            pos = _get_cursor_pos_win32()
            if pos is not None:
                cx, cy = pos
                # Convert from virtual desktop coords to image-local coords.
                lx = int(cx - region.left)
                ly = int(cy - region.top)
                if 0 <= lx < region.width and 0 <= ly < region.height:
                    _draw_cursor_marker(img, x=lx, y=ly)

        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        preview_png_bytes: Optional[bytes] = None
        preview_size: Optional[Tuple[int, int]] = None

        if preview_max_size is not None:
            pmw, pmh = preview_max_size
            preview = img.copy()
            preview.thumbnail((pmw, pmh))
            preview_size = (int(preview.size[0]), int(preview.size[1]))

            pbuf = io.BytesIO()
            preview.save(pbuf, format="PNG")
            preview_png_bytes = pbuf.getvalue()

        return Screenshot(region=region, png_bytes=png_bytes, preview_png_bytes=preview_png_bytes, preview_size=preview_size)
