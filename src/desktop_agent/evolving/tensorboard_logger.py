from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


try:
    from tensorboard.compat.proto.event_pb2 import Event  # type: ignore
    from tensorboard.compat.proto.histogram_pb2 import HistogramProto  # type: ignore
    from tensorboard.compat.proto.summary_pb2 import Summary  # type: ignore
    from tensorboard.summary.writer.event_file_writer import EventFileWriter  # type: ignore

    _TB_OK = True
except Exception:  # pragma: no cover
    Event = None  # type: ignore[assignment]
    HistogramProto = None  # type: ignore[assignment]
    Summary = None  # type: ignore[assignment]
    EventFileWriter = None  # type: ignore[assignment]
    _TB_OK = False

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    _PIL_OK = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]
    _PIL_OK = False

from .config import EvolvingConfig
from .state import WorldState


def tensorboard_available() -> bool:
    return _TB_OK


def _wall_time() -> float:
    return float(time.time())


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _make_histogram(values: list[float], *, bins: int = 30) -> "HistogramProto":
    if HistogramProto is None:
        raise RuntimeError("tensorboard is not available")
    if not values:
        return HistogramProto(min=0.0, max=0.0, num=0, sum=0.0, sum_squares=0.0)

    vs = [float(v) for v in values]
    vmin = min(vs)
    vmax = max(vs)
    num = len(vs)
    s = sum(vs)
    ss = sum(v * v for v in vs)

    if vmin == vmax:
        # Single bucket.
        return HistogramProto(
            min=vmin,
            max=vmax,
            num=num,
            sum=s,
            sum_squares=ss,
            bucket_limit=[vmax],
            bucket=[float(num)],
        )

    bins = max(1, int(bins))
    width = (vmax - vmin) / bins
    counts = [0] * bins
    for v in vs:
        idx = int((v - vmin) / width)
        if idx == bins:
            idx = bins - 1
        counts[idx] += 1
    bucket_limit = [vmin + width * (i + 1) for i in range(bins)]
    bucket = [float(c) for c in counts]
    return HistogramProto(
        min=vmin,
        max=vmax,
        num=num,
        sum=s,
        sum_squares=ss,
        bucket_limit=bucket_limit,
        bucket=bucket,
    )


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _relationship_heatmap_png(
    *,
    st: WorldState,
    cell: int = 18,
    pad: int = 70,
    font: Optional[ImageFont.ImageFont] = None,
) -> tuple[bytes, int, int]:
    if not _PIL_OK:  # pragma: no cover
        raise RuntimeError("Pillow not installed")
    ids = sorted(st.agents.keys())
    n = len(ids)
    w = pad + n * cell + 10
    h = pad + n * cell + 10

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = font or ImageFont.load_default()

    # Collect min/max for scaling.
    vals: list[float] = []
    for i in ids:
        row = st.relationship.get(i, {})
        for j in ids:
            if i == j:
                continue
            vals.append(_safe_float(row.get(j, 0.0)))
    vmax = max([abs(v) for v in vals], default=1.0) or 1.0

    def color(v: float) -> tuple[int, int, int]:
        # Negative -> red, Positive -> green, 0 -> gray/white.
        x = max(-vmax, min(vmax, v)) / vmax
        if x >= 0:
            g = int(255 * x)
            return (255 - g, 255, 255 - g)
        r = int(255 * (-x))
        return (255, 255 - r, 255 - r)

    # Labels
    draw.text((5, 5), "Relationship matrix (signed)", fill=(0, 0, 0), font=font)
    for idx, aid in enumerate(ids):
        draw.text((pad + idx * cell + 2, pad - 14), aid, fill=(0, 0, 0), font=font)
        draw.text((5, pad + idx * cell + 2), aid, fill=(0, 0, 0), font=font)

    # Cells
    for r, i in enumerate(ids):
        for c, j in enumerate(ids):
            x0 = pad + c * cell
            y0 = pad + r * cell
            if i == j:
                draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=(220, 220, 220), outline=(200, 200, 200))
                continue
            v = _safe_float(st.relationship.get(i, {}).get(j, 0.0))
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=color(v), outline=(230, 230, 230))

    return _png_bytes(img), h, w


def _family_tree_png(*, st: WorldState, font: Optional[ImageFont.ImageFont] = None) -> tuple[bytes, int, int]:
    if not _PIL_OK:  # pragma: no cover
        raise RuntimeError("Pillow not installed")
    # Simple layered layout by generation.
    ids = sorted(st.agents.keys())

    parents: dict[str, list[str]] = {}
    children: dict[str, list[str]] = {}
    for aid in ids:
        node = st.family_tree.get(aid, {})
        ps = node.get("parents", [])
        if isinstance(ps, list) and all(isinstance(x, str) for x in ps):
            parents[aid] = ps
        else:
            parents[aid] = []
        cs = node.get("children", [])
        if isinstance(cs, list) and all(isinstance(x, str) for x in cs):
            children[aid] = cs
        else:
            children[aid] = []

    gen: dict[str, int] = {aid: 0 for aid in ids}
    # Iterate to convergence.
    for _ in range(50):
        changed = False
        for aid in ids:
            ps = parents.get(aid, [])
            if ps:
                g = max(gen.get(p, 0) for p in ps) + 1
                if g != gen.get(aid, 0):
                    gen[aid] = g
                    changed = True
        if not changed:
            break

    max_gen = max(gen.values(), default=0)
    layers: list[list[str]] = [[] for _ in range(max_gen + 1)]
    for aid in ids:
        layers[gen[aid]].append(aid)
    for layer in layers:
        layer.sort()

    node_w, node_h = 90, 22
    x_pad, y_pad = 30, 40
    layer_gap = 60
    img_w = max(500, x_pad * 2 + max((len(layer) * (node_w + 20)) for layer in layers) - 20)
    img_h = y_pad * 2 + (max_gen + 1) * layer_gap + 60

    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = font or ImageFont.load_default()
    draw.text((5, 5), "Family tree", fill=(0, 0, 0), font=font)

    pos: dict[str, tuple[int, int]] = {}
    for g, layer in enumerate(layers):
        y = y_pad + g * layer_gap + 30
        total_w = len(layer) * (node_w + 20) - 20
        x0 = max(x_pad, (img_w - total_w) // 2)
        for i, aid in enumerate(layer):
            x = x0 + i * (node_w + 20)
            pos[aid] = (x, y)

    # Edges first
    for aid in ids:
        for ch in children.get(aid, []):
            if aid in pos and ch in pos:
                x1, y1 = pos[aid]
                x2, y2 = pos[ch]
                draw.line([(x1 + node_w // 2, y1 + node_h), (x2 + node_w // 2, y2)], fill=(160, 160, 160), width=2)

    # Nodes
    for aid in ids:
        x, y = pos.get(aid, (10, 10))
        rec = st.agents.get(aid)
        g = getattr(rec, "gender", "male")
        fill = (200, 220, 255) if g == "male" else (255, 210, 230)
        if rec is not None and not rec.alive:
            fill = (210, 210, 210)
        draw.rounded_rectangle([x, y, x + node_w, y + node_h], radius=6, fill=fill, outline=(80, 80, 80))
        draw.text((x + 6, y + 4), aid, fill=(0, 0, 0), font=font)

    return _png_bytes(img), img_h, img_w


@dataclass
class TensorboardLogger:
    logdir: Path

    def __post_init__(self) -> None:
        if not tensorboard_available():  # pragma: no cover
            raise RuntimeError("tensorboard is not installed. Install with: pip install tensorboard")
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._writer = EventFileWriter(str(self.logdir))
        self._step = 0

    def close(self) -> None:
        try:
            self._writer.close()
        except Exception:
            pass

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def _add_summary(self, summary: "Summary", step: Optional[int] = None) -> None:
        if step is None:
            step = self._step
        ev = Event(wall_time=_wall_time(), step=int(step), summary=summary)
        self._writer.add_event(ev)
        self._writer.flush()

    def add_scalar(self, tag: str, value: float, *, step: Optional[int] = None) -> None:
        summ = Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
        self._add_summary(summ, step=step)

    def add_histogram(self, tag: str, values: Iterable[float], *, step: Optional[int] = None, bins: int = 30) -> None:
        vs = [_safe_float(v) for v in values]
        hist = _make_histogram(vs, bins=bins)
        summ = Summary(value=[Summary.Value(tag=tag, histo=hist)])
        self._add_summary(summ, step=step)

    def add_image_png(self, tag: str, png: bytes, *, height: int, width: int, step: Optional[int] = None) -> None:
        img = Summary.Image(encoded_image_string=png, height=int(height), width=int(width), colorspace=3)
        summ = Summary(value=[Summary.Value(tag=tag, image=img)])
        self._add_summary(summ, step=step)

    def log_day(
        self,
        *,
        cfg: EvolvingConfig,
        st: WorldState,
        day_stats: dict[str, Any],
        day_index: int,
        world_root: Path,
    ) -> None:
        self.set_step(day_index)

        # Population
        agents = list(st.agents.values())
        alive = [a for a in agents if a.alive]
        self.add_scalar("pop/alive", len(alive))
        self.add_scalar("pop/total", len(agents))
        self.add_histogram("pop/age_days", [a.age_days for a in alive])
        self.add_histogram("pop/energy", [a.energy for a in alive])

        # Basic energy stats
        if alive:
            energies = sorted([float(a.energy) for a in alive])
            self.add_scalar("pop/energy_p10", energies[int(0.1 * (len(energies) - 1))])
            self.add_scalar("pop/energy_p50", energies[int(0.5 * (len(energies) - 1))])
            self.add_scalar("pop/energy_p90", energies[int(0.9 * (len(energies) - 1))])
            self.add_scalar("pop/starvation_risk", sum(1 for e in energies if e < 20.0))

        # Daily events
        self.add_scalar("events/births", _safe_float(day_stats.get("births", 0)))
        self.add_scalar("events/deaths", _safe_float(day_stats.get("deaths", 0)))
        self.add_scalar("events/mate_requests", _safe_float(day_stats.get("mate_requests", 0)))
        self.add_scalar("events/mate_accepts", _safe_float(day_stats.get("mate_accepts", 0)))

        # Interaction type counts
        self.add_scalar("interactions/cooperative", _safe_float(day_stats.get("coop", 0)))
        self.add_scalar("interactions/competitive", _safe_float(day_stats.get("comp", 0)))
        self.add_scalar("interactions/neutral", _safe_float(day_stats.get("neutral", 0)))
        self.add_histogram("interactions/type", [1] * int(_safe_float(day_stats.get("coop", 0))) + [-1] * int(_safe_float(day_stats.get("comp", 0))))

        # Ops histogram (top-level)
        op_counts = day_stats.get("op_counts", {})
        if isinstance(op_counts, dict) and op_counts:
            # encode as histogram of categorical ids by repeating small ints
            ops = sorted(op_counts.keys())
            vals: list[float] = []
            for idx, op in enumerate(ops):
                vals.extend([float(idx)] * int(op_counts.get(op, 0)))
            if vals:
                self.add_histogram("actions/op_id_hist", vals, bins=min(30, max(5, len(ops))))

        # Instruction/memory sizes (from files)
        agents_dir = world_root / "agents"
        ins_lens: list[float] = []
        mem_lens: list[float] = []
        for a in agents:
            ins = agents_dir / f"{a.agent_id}_instructions.md"
            mem = agents_dir / f"{a.agent_id}_memory.md"
            try:
                ins_lens.append(float(len(ins.read_text(encoding="utf-8"))))
            except Exception:
                pass
            try:
                mem_lens.append(float(len(mem.read_text(encoding="utf-8"))))
            except Exception:
                pass
        if ins_lens:
            self.add_histogram("files/instruction_chars", ins_lens)
        if mem_lens:
            self.add_histogram("files/memory_chars", mem_lens)
        self.add_scalar("files/edits_today_total", _safe_float(day_stats.get("edits", 0)))

        # Relationship matrix + family tree as images
        if _PIL_OK:
            try:
                png, h, w = _relationship_heatmap_png(st=st)
                self.add_image_png("viz/relationship_matrix", png, height=h, width=w)
            except Exception:
                pass
            try:
                png2, h2, w2 = _family_tree_png(st=st)
                self.add_image_png("viz/family_tree", png2, height=h2, width=w2)
            except Exception:
                pass
