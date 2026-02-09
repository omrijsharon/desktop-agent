r"""scripts/llm_image_file_grid_demo.py

Interactive multi-image tiling probe using a user-selected image file.

What it does:
- Lets you pick an image from disk (file picker).
- Lets you type the question/prompt text.
- Runs the SAME "tiling + rounds + branching" approach as llm_multi_image_grid_demo.py,
  but using the chosen image as the starting point instead of a screenshot.

Branches (defaults):
- Branch A: 8x8 grid for 2 rounds
- Branch B: 3x3 grid for 4 rounds

Saved PNG naming:
- selected_tile_branch<BRANCH>_round<R>_tile<T>.png

Usage (PowerShell):
  .\.venv\Scripts\python .\scripts\llm_image_file_grid_demo.py

Env:
- OPENAI_API_KEY required (recommended via .env)
- OPENAI_MODEL optional (defaults to config DEFAULT_MODEL)

Notes:
- This script does NOT execute any actions; it only tests vision input handling.
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
import time

from desktop_agent.config import load_config
from desktop_agent.llm import OpenAIResponsesClient


def _pick_image_path() -> str | None:
    """Open a native file picker and return the selected path (or None)."""

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path or None
    except Exception:
        return None


def _load_image_as_png_bytes(path: str) -> bytes:
    from PIL import Image  # type: ignore

    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _question_for_grid(user_prompt: str, n: int) -> str:
    max_tiles = n * n
    # Keep user text verbatim, then add the numbering scheme.
    return (
        f"{user_prompt.strip()}\n\n"
        f"Answer with a number 1-{max_tiles} (row-major order): "
        f"1=top-left, {n}=top-right, {max_tiles - n + 1}=bottom-left, {max_tiles}=bottom-right."
    )


def _split_png_grid(png_bytes: bytes, *, n: int) -> list[bytes]:
    from PIL import Image  # type: ignore

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = img.size

    tw = max(1, w // n)
    th = max(1, h // n)

    out: list[bytes] = []
    for row in range(n):
        for col in range(n):
            left = col * tw
            top = row * th
            right = (col + 1) * tw if col < n - 1 else w
            bottom = (row + 1) * th if row < n - 1 else h
            tile = img.crop((left, top, right, bottom))
            buf = io.BytesIO()
            tile.save(buf, format="PNG")
            out.append(buf.getvalue())
    return out


def _build_prompt_parts(*, tiles: list[bytes], question: str) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = [{"type": "input_text", "text": question}]
    for i, tile in enumerate(tiles, start=1):
        b64 = base64.b64encode(tile).decode("ascii")
        parts.append({"type": "input_text", "text": f"Picture {i}:"})
        parts.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})
    return parts


def _build_single_image_question(*, png_bytes: bytes, question: str) -> list[dict[str, str]]:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return [
        {"type": "input_text", "text": question},
        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
    ]


def _critic_question(user_prompt: str) -> str:
    return (
        "CRITIC CHECK: Given ONLY this single image tile, answer yes/no: "
        "Is the thing the user looked for really present in this tile?\n\n"
        f"User prompt: {user_prompt}"
    )


def _parse_yes_no(text: str) -> bool | None:
    s = (text or "").strip().lower()
    if not s:
        return None
    if s.startswith("yes"):
        return True
    if s.startswith("no"):
        return False
    if s == "y":
        return True
    if s == "n":
        return False
    return None


def _extract_choice_1_to_n(text: str, *, n: int) -> int | None:
    s = (text or "").strip().lower()
    if not s:
        return None

    m = re.search(r"\b(\d{1,4})\b", s)
    if m:
        try:
            v = int(m.group(1))
        except Exception:
            v = -1
        if 1 <= v <= n:
            return v

    return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


_BASE_TEMPERATURE = float(os.environ.get("GRID_DEMO_TEMPERATURE", "0.2"))
_TEMP_STEP = float(os.environ.get("GRID_DEMO_TEMPERATURE_STEP", "0.6"))
_MAX_TEMPERATURE = float(os.environ.get("GRID_DEMO_TEMPERATURE_MAX", "1.2"))


def _tile_bounds(*, parent_w: int, parent_h: int, n: int, index_1_based: int) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) bounds of tile index in an n×n grid."""

    idx = index_1_based - 1
    row = idx // n
    col = idx % n

    tw = max(1, parent_w // n)
    th = max(1, parent_h // n)

    left = col * tw
    top = row * th
    right = (col + 1) * tw if col < n - 1 else parent_w
    bottom = (row + 1) * th if row < n - 1 else parent_h
    return left, top, right, bottom


def _save_full_image_with_circle(
    *,
    full_png: bytes,
    out_path: str,
    center_x: int,
    center_y: int,
    prompt_text: str = "",
    radius: int = 25,
    stroke: int = 5,
    banner_pad: int = 10,
) -> None:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    img = Image.open(io.BytesIO(full_png)).convert("RGB")
    draw = ImageDraw.Draw(img)

    banner = (prompt_text or "").strip()
    if banner:
        font = None
        try:
            font = ImageFont.truetype("arial.ttf", size=70)
        except Exception:
            try:
                font = ImageFont.truetype("SegoeUI.ttf", size=70)
            except Exception:
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

        if font is not None and hasattr(draw, "textbbox"):
            bx0, by0, bx1, by1 = draw.textbbox((0, 0), banner, font=font)
            tw = bx1 - bx0
            th = by1 - by0
        else:
            tw, th = draw.textsize(banner, font=font)  # type: ignore[attr-defined]

        rect_w = tw + (banner_pad * 2)
        rect_h = th + (banner_pad * 2)

        cx = img.size[0] // 2
        x0 = max(0, cx - (rect_w // 2))
        y0 = 0
        x1 = min(img.size[0], x0 + rect_w)
        y1 = min(img.size[1], y0 + rect_h)

        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
        tx = x0 + banner_pad
        ty = y0 + banner_pad

        if font is not None:
            draw.text((tx, ty), banner, fill=(255, 255, 255), font=font)
        else:
            draw.text((tx, ty), banner, fill=(255, 255, 255))

    r = max(2, int(radius))
    s = max(1, int(stroke))

    left = center_x - r
    top = center_y - r
    right = center_x + r
    bottom = center_y + r

    draw.ellipse([left, top, right, bottom], outline=(255, 0, 0), width=s)
    draw.ellipse([center_x - 2, center_y - 2, center_x + 2, center_y + 2], fill=(255, 0, 0))

    img.save(out_path, format="PNG")


def main() -> int:
    cfg = load_config()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set (use .env).", file=sys.stderr)
        return 2

    path = _pick_image_path()
    if not path:
        print("No image selected.")
        return 0

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 2

    print(f"Image: {path}")
    user_prompt = input("Enter your prompt/question: ").strip()
    if not user_prompt:
        print("ERROR: Prompt cannot be empty.", file=sys.stderr)
        return 2

    # Load once; each branch starts from these bytes.
    start_png = _load_image_as_png_bytes(path)

    # Work on a smaller image for model calls to reduce latency/cost.
    # NOTE: downscaling hurts small-text targets; default back to full resolution.
    DOWNSCALE = float(os.environ.get("GRID_DEMO_IMAGE_DOWNSCALE", "1.0"))
    DOWNSCALE = _clamp(DOWNSCALE, 0.1, 1.0)
    INV_DOWNSCALE = int(round(1.0 / DOWNSCALE))

    from PIL import Image  # type: ignore

    full_png = start_png
    full_img = Image.open(io.BytesIO(full_png)).convert("RGB")
    if DOWNSCALE >= 0.999:
        small_png = full_png
    else:
        small_w = max(1, int(full_img.size[0] * DOWNSCALE))
        small_h = max(1, int(full_img.size[1] * DOWNSCALE))
        small_img = full_img.resize((small_w, small_h), resample=Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        small_img.save(buf, format="PNG")
        small_png = buf.getvalue()

    client = OpenAIResponsesClient(api_key=api_key)
    out_dir = os.path.dirname(__file__)

    def ask_once(*, tiles: list[bytes], question: str, label: str, temperature: float) -> str:
        inp = [{"role": "user", "content": _build_prompt_parts(tiles=tiles, question=question)}]
        print(f"\n=== {label} ===")
        print(f"Model: {cfg.openai_model}")
        print(f"Temperature: {temperature}")
        return client.responses_create(
            model=cfg.openai_model,
            input=inp,
            truncation="auto",
            temperature=temperature,
        )

    def run_branch(*, branch: str, grid_n: int | list[int], rounds: int) -> bool:
        t0 = time.perf_counter()
        ok = False
        try:
            # Allow per-round grid sizes (e.g., [16, 2]).
            grid_schedule: list[int]
            if isinstance(grid_n, int):
                grid_schedule = [grid_n] * rounds
            else:
                grid_schedule = list(grid_n)
                if len(grid_schedule) != rounds:
                    raise ValueError(f"grid_n schedule length {len(grid_schedule)} != rounds {rounds}")

            # Use downscaled image for the model/tiling loop.
            current_png = small_png

            offset_x = 0
            offset_y = 0

            # Fallback: if later rounds fail, use the best known center from earlier rounds.
            fallback_center_x_small: int | None = None
            fallback_center_y_small: int | None = None

            print(f"\n##### BRANCH {branch}: {grid_schedule} grids for {rounds} rounds #####\n")

            max_retries_per_round = 3

            last_tile_w = 0
            last_tile_h = 0

            rounds_completed = 0

            for r in range(1, rounds + 1):
                this_grid_n = grid_schedule[r - 1]
                question = _question_for_grid(user_prompt, this_grid_n)

                attempt = 0
                temperature = _clamp(_BASE_TEMPERATURE, 0.0, _MAX_TEMPERATURE)
                while True:
                    attempt += 1

                    from PIL import Image  # type: ignore

                    parent_img = Image.open(io.BytesIO(current_png)).convert("RGB")
                    parent_w, parent_h = parent_img.size

                    tiles = _split_png_grid(current_png, n=this_grid_n)
                    try:
                        txt = ask_once(
                            tiles=tiles,
                            question=question,
                            label=f"BRANCH {branch} / ROUND {r} ({this_grid_n}x{this_grid_n}) [attempt {attempt}]",
                            temperature=temperature,
                        )
                    except Exception as e:
                        print(f"\nBRANCH {branch} failed calling model on round {r}: {e}", file=sys.stderr)
                        if fallback_center_x_small is not None and fallback_center_y_small is not None:
                            print(
                                f"\nBRANCH {branch}: falling back to earlier-round center due to failure on round {r}.",
                                file=sys.stderr,
                            )
                            break
                        return False

                    print(f"\n--- RAW MODEL OUTPUT (BRANCH {branch} ROUND {r}) ---\n")
                    print(txt)

                    choice = _extract_choice_1_to_n(txt, n=len(tiles))
                    if choice is None:
                        if attempt <= max_retries_per_round:
                            temperature = _clamp(temperature + _TEMP_STEP, 0.0, _MAX_TEMPERATURE)
                            print(
                                f"\nBRANCH {branch}: could not parse a 1-{len(tiles)} selection; retrying with higher temperature {temperature}...",
                                file=sys.stderr,
                            )
                            continue
                        print(
                            f"\nBRANCH {branch}: could not parse a 1-{len(tiles)} selection from round {r}; stopping.",
                            file=sys.stderr,
                        )
                        if fallback_center_x_small is not None and fallback_center_y_small is not None:
                            break
                        return False

                    selected_png = tiles[choice - 1]

                    left, top, right, bottom = _tile_bounds(
                        parent_w=parent_w,
                        parent_h=parent_h,
                        n=this_grid_n,
                        index_1_based=choice,
                    )
                    offset_x += left
                    offset_y += top
                    last_tile_w = right - left
                    last_tile_h = bottom - top

                    # Critic step: verify the target is actually present in the chosen tile.
                    try:
                        critic_inp = [
                            {
                                "role": "user",
                                "content": _build_single_image_question(
                                    png_bytes=selected_png,
                                    question=_critic_question(user_prompt),
                                ),
                            }
                        ]
                        critic_txt = client.responses_create(
                            model=cfg.openai_model,
                            input=critic_inp,
                            truncation="auto",
                            temperature=0.0,
                        )
                        critic_ok = _parse_yes_no(critic_txt)
                    except Exception as e:
                        print(f"\nBRANCH {branch}: critic call failed on round {r}: {e}", file=sys.stderr)
                        critic_ok = None

                    print(f"\n--- CRITIC OUTPUT (BRANCH {branch} ROUND {r}) ---\n")
                    print(critic_txt if 'critic_txt' in locals() else "")

                    if critic_ok is False:
                        print(f"\nBRANCH {branch}: critic says NO for choice {choice} on round {r}. Retrying selection...")
                        if attempt <= max_retries_per_round:
                            temperature = _clamp(temperature + _TEMP_STEP, 0.0, _MAX_TEMPERATURE)
                            continue
                        print(f"\nBRANCH {branch}: exceeded retry budget on round {r}. Stopping.", file=sys.stderr)
                        if fallback_center_x_small is not None and fallback_center_y_small is not None:
                            break
                        return False

                    # If critic can't be parsed, proceed (best-effort) but still save.
                    current_png = selected_png

                    selected_path = os.path.abspath(
                        os.path.join(out_dir, f"selected_tile_branch{branch}_round{r}_tile{choice}.png")
                    )
                    try:
                        with open(selected_path, "wb") as f:
                            f.write(current_png)
                        print(f"\nSaved selected tile: {selected_path}")
                    except Exception as e:
                        print(f"\nWARNING: Failed to save selected tile: {e}", file=sys.stderr)

                    rounds_completed += 1

                    # Record fallback center from round 1 (or earliest successful round).
                    if fallback_center_x_small is None and fallback_center_y_small is None:
                        fallback_center_x_small = int(offset_x + (last_tile_w / 2.0))
                        fallback_center_y_small = int(offset_y + (last_tile_h / 2.0))

                    break

                if rounds_completed < r:
                    break

            if rounds_completed <= 0:
                return False

            center_x_small = int(offset_x + (last_tile_w / 2.0))
            center_y_small = int(offset_y + (last_tile_h / 2.0))
            if rounds_completed < rounds and fallback_center_x_small is not None and fallback_center_y_small is not None:
                center_x_small = fallback_center_x_small
                center_y_small = fallback_center_y_small

            center_x = int(center_x_small * INV_DOWNSCALE)
            center_y = int(center_y_small * INV_DOWNSCALE)

            full_out_path = os.path.abspath(os.path.join(out_dir, f"full_image_branch{branch}.png"))
            print(
                f"\nBRANCH {branch}: attempting to save full image overlay to: {full_out_path} "
                f"(center_small=({center_x_small},{center_y_small}) -> center_full=({center_x},{center_y}))"
            )
            try:
                # Best-effort clamp of coords to image bounds.
                from PIL import Image  # type: ignore

                img = Image.open(io.BytesIO(full_png))
                w, h = img.size
                cx = max(0, min(w - 1, center_x))
                cy = max(0, min(h - 1, center_y))

                _save_full_image_with_circle(
                    full_png=full_png,
                    out_path=full_out_path,
                    center_x=cx,
                    center_y=cy,
                    prompt_text=user_prompt,
                )
                print(f"\nSaved full image w/ center marker: {full_out_path} (center=({cx},{cy}))")
            except Exception as e:
                print(
                    f"\nWARNING: Failed to save full image overlay for branch {branch}: {e!r}",
                    file=sys.stderr,
                )

            ok = True
            return rounds_completed == rounds
        finally:
            dt = time.perf_counter() - t0
            status = "OK" if ok else "FAILED"
            print(f"\n##### BRANCH {branch} DONE: {status} in {dt:.2f}s #####\n")

    # Run only Branch A (8x8).
    ok_a = run_branch(branch="A", grid_n=8, rounds=2)

    print("\n--- END ---\n")
    if not ok_a:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
