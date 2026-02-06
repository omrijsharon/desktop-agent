#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

import gguf_inspect


QK_K = 256
Q5K_BLOCK_BYTES = 176
Q6K_BLOCK_BYTES = 210


class WeightHistError(RuntimeError):
    pass


def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise WeightHistError(f"Unexpected EOF (wanted {n} bytes, got {len(b)})")
    return b


def _fp16(buf2: bytes) -> float:
    return float(np.frombuffer(buf2, dtype=np.dtype("<f2"), count=1)[0].astype(np.float32))


def _get_scale_min_k4(j: int, q: np.ndarray) -> tuple[int, int, int, int]:
    # Port of get_scale_min_k4() from ggml-metal (llama.cpp).
    # q is uint8 array of length 12.
    if j < 4:
        return (
            int(q[j + 0] & 63),
            int(q[j + 4] & 63),
            int(q[j + 1] & 63),
            int(q[j + 5] & 63),
        )
    return (
        int((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)),
        int((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4)),
        int((q[j + 5] & 0xF) | ((q[j - 3] >> 6) << 4)),
        int((q[j + 5] >> 4) | ((q[j + 1] >> 6) << 4)),
    )


def dequant_q5_k_block(block: bytes) -> np.ndarray:
    if len(block) != Q5K_BLOCK_BYTES:
        raise WeightHistError(f"Bad Q5_K block size: {len(block)}")

    d = _fp16(block[0:2])
    dmin = _fp16(block[2:4])
    scales = np.frombuffer(block[4:16], dtype=np.uint8, count=12)
    qh = np.frombuffer(block[16:48], dtype=np.uint8, count=32)
    qs = np.frombuffer(block[48:176], dtype=np.uint8, count=128)

    out = np.empty((QK_K,), dtype=np.float32)

    u1 = 1
    u2 = 2
    for chunk in range(4):
        sc0, sc1, sc2, sc3 = _get_scale_min_k4(chunk * 2, scales)
        d1 = d * sc0
        m1 = dmin * sc1
        d2 = d * sc2
        m2 = dmin * sc3

        ql = qs[chunk * 32 : (chunk + 1) * 32]

        lo = (ql & 0xF).astype(np.float32) + ((qh & u1) != 0).astype(np.float32) * 16.0
        hi = (ql >> 4).astype(np.float32) + ((qh & u2) != 0).astype(np.float32) * 16.0

        base = chunk * 64
        out[base : base + 32] = d1 * lo - m1
        out[base + 32 : base + 64] = d2 * hi - m2

        u1 <<= 2
        u2 <<= 2

    return out


def dequant_q6_k_block(block: bytes) -> np.ndarray:
    if len(block) != Q6K_BLOCK_BYTES:
        raise WeightHistError(f"Bad Q6_K block size: {len(block)}")

    ql_all = np.frombuffer(block[0:128], dtype=np.uint8, count=128)
    qh_all = np.frombuffer(block[128:192], dtype=np.uint8, count=64)
    sc_all = np.frombuffer(block[192:208], dtype=np.int8, count=16)
    d = _fp16(block[208:210])

    out = np.empty((QK_K,), dtype=np.float32)
    l = np.arange(32, dtype=np.int32)
    is_idx = (l // 16).astype(np.int32)  # 0 for 0..15, 1 for 16..31

    for half in range(2):
        ql = ql_all[half * 64 : (half + 1) * 64]
        qh = qh_all[half * 32 : (half + 1) * 32]
        sc = sc_all[half * 8 : (half + 1) * 8].astype(np.float32)

        ql0 = ql[0:32]
        ql1 = ql[32:64]

        q1 = ((ql0 & 0xF) | (((qh >> 0) & 3) << 4)).astype(np.int16) - 32
        q2 = ((ql1 & 0xF) | (((qh >> 2) & 3) << 4)).astype(np.int16) - 32
        q3 = ((ql0 >> 4) | (((qh >> 4) & 3) << 4)).astype(np.int16) - 32
        q4 = ((ql1 >> 4) | (((qh >> 6) & 3) << 4)).astype(np.int16) - 32

        s1 = sc[is_idx + 0]
        s2 = sc[is_idx + 2]
        s3 = sc[is_idx + 4]
        s4 = sc[is_idx + 6]

        base = half * 128
        out[base + 0 : base + 32] = (d * s1) * q1.astype(np.float32)
        out[base + 32 : base + 64] = (d * s2) * q2.astype(np.float32)
        out[base + 64 : base + 96] = (d * s3) * q3.astype(np.float32)
        out[base + 96 : base + 128] = (d * s4) * q4.astype(np.float32)

    return out


@dataclass
class LogHist:
    exp_min: float
    exp_max: float
    bins_per_decade: int
    counts: np.ndarray
    zeros: int = 0
    underflow: int = 0
    overflow: int = 0
    nonfinite: int = 0

    @property
    def n_bins(self) -> int:
        return int(self.counts.size)

    def add(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        v = values.astype(np.float64, copy=False)
        finite = np.isfinite(v)
        if not np.all(finite):
            self.nonfinite += int((~finite).sum())
            v = v[finite]
        if v.size == 0:
            return

        z = (v == 0)
        if np.any(z):
            self.zeros += int(z.sum())
            v = v[~z]
        if v.size == 0:
            return

        logv = np.log10(v)
        idx = np.floor((logv - self.exp_min) * self.bins_per_decade).astype(np.int64)
        self.underflow += int((idx < 0).sum())
        self.overflow += int((idx >= self.n_bins).sum())
        idx = idx[(idx >= 0) & (idx < self.n_bins)]
        if idx.size:
            self.counts += np.bincount(idx, minlength=self.n_bins).astype(np.int64)

    def edges(self) -> np.ndarray:
        n = self.n_bins
        return np.logspace(self.exp_min, self.exp_min + (n / self.bins_per_decade), n + 1)


@dataclass
class RunningMoments:
    n: int = 0
    sum: float = 0.0
    sumsq: float = 0.0

    def add(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        v = values.astype(np.float64, copy=False)
        finite = np.isfinite(v)
        if not np.all(finite):
            v = v[finite]
        if v.size == 0:
            return
        self.n += int(v.size)
        self.sum += float(v.sum(dtype=np.float64))
        self.sumsq += float((v * v).sum(dtype=np.float64))

    def mean(self) -> float:
        return self.sum / self.n if self.n else float("nan")

    def std(self) -> float:
        if not self.n:
            return float("nan")
        mean = self.sum / self.n
        var = (self.sumsq / self.n) - (mean * mean)
        return float(math.sqrt(max(0.0, var)))


def _geom_indices(rng: np.random.Generator, p: float, n: int) -> list[int]:
    if p <= 0:
        return []
    if p >= 1:
        return list(range(n))

    out: list[int] = []
    i = int(rng.geometric(p) - 1)
    while i < n:
        out.append(i)
        i += int(rng.geometric(p))
    return out


def _select_tensors(
    gguf: gguf_inspect.GGUFFile,
    *,
    tensor_regex: str,
    types: set[int],
) -> list[gguf_inspect.GGUFTensor]:
    rx = re.compile(tensor_regex) if tensor_regex else None
    selected: list[gguf_inspect.GGUFTensor] = []
    for t in gguf.tensors:
        if t.ggml_type not in types:
            continue
        if rx and not rx.search(t.name):
            continue
        if t.n_elems <= 0:
            continue
        selected.append(t)
    return selected


def _plot_png(
    hist: LogHist,
    *,
    title: str,
    out_png: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise WeightHistError(
            "matplotlib is required for plotting. Install it (python -m pip install matplotlib) "
            "or run with --out-json to inspect histogram counts."
        ) from e

    edges = hist.edges()
    centers = np.sqrt(edges[:-1] * edges[1:])
    counts = hist.counts.astype(np.float64)

    fig = plt.figure(figsize=(10, 6), dpi=160)
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(centers, counts, width=(edges[1:] - edges[:-1]), align="center", linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|weight|")
    ax.set_ylabel("count")
    ax.set_title(title)

    note = f"zeros={hist.zeros} underflow={hist.underflow} overflow={hist.overflow} nonfinite={hist.nonfinite}"
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=9, va="bottom", ha="left")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Sample and plot a log-log histogram of absolute weight magnitudes from a GGUF model (Q5_K/Q6_K)."
    )
    ap.add_argument(
        "gguf_path",
        nargs="?",
        default=os.environ.get("GGUF_PATH", ""),
        help="Path to .gguf (or set GGUF_PATH)",
    )
    ap.add_argument("--tensor-regex", default="", help="Only include tensors whose name matches this regex")
    ap.add_argument("--seed", type=int, default=0, help="Sampling RNG seed")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=20_000_000,
        help="Approx max number of weights to sample across all selected tensors",
    )
    ap.add_argument("--exp-min", type=float, default=-12.0, help="Min log10(|w|) bin edge")
    ap.add_argument("--exp-max", type=float, default=2.0, help="Max log10(|w|) bin edge")
    ap.add_argument("--bins-per-decade", type=int, default=10, help="Histogram resolution")
    ap.add_argument("--out-png", type=str, default="", help="Write PNG plot to this path")
    ap.add_argument("--out-json", type=str, default="", help="Write histogram data to this JSON path")
    ap.add_argument(
        "--types",
        type=str,
        default="Q5_K,Q6_K",
        help="Comma-separated ggml types to include (default: Q5_K,Q6_K)",
    )

    args = ap.parse_args(argv)
    if not args.gguf_path:
        ap.error("gguf_path is required (or set GGUF_PATH)")

    gguf_path = Path(args.gguf_path)
    gguf = gguf_inspect.read_gguf(gguf_path)

    type_names = {k.upper().strip() for k in args.types.split(",") if k.strip()}
    name_to_type = {v: k for k, v in gguf_inspect.GGML_TYPE_NAME.items()}
    ggml_types: set[int] = set()
    for tn in type_names:
        if tn not in name_to_type:
            raise WeightHistError(f"Unknown type name {tn!r}. Known: {sorted(name_to_type)[:15]} ...")
        ggml_types.add(name_to_type[tn])

    selected = _select_tensors(gguf, tensor_regex=args.tensor_regex, types=ggml_types)
    if not selected:
        raise WeightHistError("No tensors matched selection.")

    # Determine sampling probability in block units.
    blocks_by_tensor: list[tuple[gguf_inspect.GGUFTensor, int, int]] = []
    total_blocks = 0
    for t in selected:
        if t.ggml_type == gguf_inspect.GGMLType.Q5_K:
            if t.n_elems % QK_K != 0:
                continue
            b = t.n_elems // QK_K
            blocks_by_tensor.append((t, Q5K_BLOCK_BYTES, b))
            total_blocks += b
        elif t.ggml_type == gguf_inspect.GGMLType.Q6_K:
            if t.n_elems % QK_K != 0:
                continue
            b = t.n_elems // QK_K
            blocks_by_tensor.append((t, Q6K_BLOCK_BYTES, b))
            total_blocks += b
        else:
            raise WeightHistError(f"Type not supported for dequant sampling: {t.type_name}")

    if total_blocks == 0:
        raise WeightHistError("No block-quantized tensors selected (or shapes not divisible by 256).")

    target_blocks = max(1, int(math.ceil(args.max_samples / QK_K)))
    p = min(1.0, target_blocks / total_blocks)

    n_bins = int((args.exp_max - args.exp_min) * args.bins_per_decade)
    if n_bins <= 0:
        raise WeightHistError("Invalid exp range / bins-per-decade.")
    hist = LogHist(
        exp_min=float(args.exp_min),
        exp_max=float(args.exp_max),
        bins_per_decade=int(args.bins_per_decade),
        counts=np.zeros((n_bins,), dtype=np.int64),
    )

    rng = np.random.default_rng(seed=args.seed)

    sampled_blocks = 0
    sampled_weights = 0
    abs_moments = RunningMoments()
    with gguf_path.open("rb") as f:
        for t, block_bytes, n_blocks in blocks_by_tensor:
            base = gguf.data_start + t.offset
            idxs = _geom_indices(rng, p, n_blocks)
            if not idxs:
                continue
            f.seek(base + idxs[0] * block_bytes)
            cur = base + idxs[0] * block_bytes
            for bi in idxs:
                pos = base + bi * block_bytes
                if pos != cur:
                    f.seek(pos)
                block = _read_exact(f, block_bytes)
                cur = pos + block_bytes

                if block_bytes == Q5K_BLOCK_BYTES:
                    w = dequant_q5_k_block(block)
                else:
                    w = dequant_q6_k_block(block)

                absw = np.abs(w).astype(np.float64, copy=False)
                hist.add(absw)
                abs_moments.add(absw)
                sampled_blocks += 1
                sampled_weights += QK_K

    payload: dict[str, Any] = {
        "gguf_path": str(gguf_path),
        "model_name": gguf.kv.get("general.name") or gguf.kv.get("general.basename") or "",
        "tensor_regex": args.tensor_regex,
        "types": sorted(type_names),
        "seed": args.seed,
        "sampling": {
            "max_samples": int(args.max_samples),
            "qk_k": QK_K,
            "target_blocks": int(target_blocks),
            "total_blocks": int(total_blocks),
            "p": float(p),
            "sampled_blocks": int(sampled_blocks),
            "sampled_weights": int(sampled_weights),
        },
        "abs_moments": {
            "n": int(abs_moments.n),
            "mean": float(abs_moments.mean()),
            "std": float(abs_moments.std()),
        },
        "histogram": {
            "exp_min": hist.exp_min,
            "exp_max": hist.exp_max,
            "bins_per_decade": hist.bins_per_decade,
            "counts": hist.counts.tolist(),
            "zeros": hist.zeros,
            "underflow": hist.underflow,
            "overflow": hist.overflow,
            "nonfinite": hist.nonfinite,
        },
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    title = payload["model_name"] or gguf_path.name
    title = f"{title} | abs(weight) histogram (log-log)"
    if args.out_png:
        _plot_png(hist, title=title, out_png=Path(args.out_png))

    if not args.out_png and not args.out_json:
        # Default: just print a short summary and how to plot.
        print(f"Sampled weights: {sampled_weights} (blocks={sampled_blocks}, p={p:.3g})")
        print(f"zeros={hist.zeros} underflow={hist.underflow} overflow={hist.overflow} nonfinite={hist.nonfinite}")
        print("Use --out-png to write a plot, or --out-json to dump counts.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (WeightHistError, gguf_inspect.GGUFError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
