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
    ) -> Screenshot:
        """Capture the full virtual screen.

        Args:
            preview_max_size: if provided, a scaled down PNG preview is returned.

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
