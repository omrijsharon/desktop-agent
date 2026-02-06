#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable


GGUF_MAGIC = b"GGUF"


class GGUFError(RuntimeError):
    pass


def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise GGUFError(f"Unexpected EOF (wanted {n} bytes, got {len(b)})")
    return b


def _u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _i8(f: BinaryIO) -> int:
    return struct.unpack("<b", _read_exact(f, 1))[0]


def _u8(f: BinaryIO) -> int:
    return struct.unpack("<B", _read_exact(f, 1))[0]


def _i16(f: BinaryIO) -> int:
    return struct.unpack("<h", _read_exact(f, 2))[0]


def _u16(f: BinaryIO) -> int:
    return struct.unpack("<H", _read_exact(f, 2))[0]


def _i32(f: BinaryIO) -> int:
    return struct.unpack("<i", _read_exact(f, 4))[0]


def _i64(f: BinaryIO) -> int:
    return struct.unpack("<q", _read_exact(f, 8))[0]


def _f32(f: BinaryIO) -> float:
    return struct.unpack("<f", _read_exact(f, 4))[0]


def _f64(f: BinaryIO) -> float:
    return struct.unpack("<d", _read_exact(f, 8))[0]


def _gguf_string(f: BinaryIO) -> str:
    n = _u64(f)
    if n > 1_000_000_000:
        raise GGUFError(f"Refusing to read absurd string length: {n}")
    return _read_exact(f, n).decode("utf-8", errors="replace")


class GGUFValueType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


def _read_kv_value(f: BinaryIO, vtype: int) -> Any:
    if vtype == GGUFValueType.UINT8:
        return _u8(f)
    if vtype == GGUFValueType.INT8:
        return _i8(f)
    if vtype == GGUFValueType.UINT16:
        return _u16(f)
    if vtype == GGUFValueType.INT16:
        return _i16(f)
    if vtype == GGUFValueType.UINT32:
        return _u32(f)
    if vtype == GGUFValueType.INT32:
        return _i32(f)
    if vtype == GGUFValueType.FLOAT32:
        return _f32(f)
    if vtype == GGUFValueType.BOOL:
        return bool(_u8(f))
    if vtype == GGUFValueType.STRING:
        return _gguf_string(f)
    if vtype == GGUFValueType.UINT64:
        return _u64(f)
    if vtype == GGUFValueType.INT64:
        return _i64(f)
    if vtype == GGUFValueType.FLOAT64:
        return _f64(f)
    if vtype == GGUFValueType.ARRAY:
        elem_type = _u32(f)
        n = _u64(f)
        if n > 1_000_000_000:
            raise GGUFError(f"Refusing to read absurd array length: {n}")
        return [_read_kv_value(f, elem_type) for _ in range(n)]
    raise GGUFError(f"Unknown GGUF kv value type: {vtype}")


class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    I8 = 16
    I16 = 17
    I32 = 18
    I64 = 19
    F64 = 20


GGML_TYPE_NAME: dict[int, str] = {
    GGMLType.F32: "F32",
    GGMLType.F16: "F16",
    GGMLType.Q4_0: "Q4_0",
    GGMLType.Q4_1: "Q4_1",
    GGMLType.Q5_0: "Q5_0",
    GGMLType.Q5_1: "Q5_1",
    GGMLType.Q8_0: "Q8_0",
    GGMLType.Q8_1: "Q8_1",
    GGMLType.Q2_K: "Q2_K",
    GGMLType.Q3_K: "Q3_K",
    GGMLType.Q4_K: "Q4_K",
    GGMLType.Q5_K: "Q5_K",
    GGMLType.Q6_K: "Q6_K",
    GGMLType.Q8_K: "Q8_K",
    GGMLType.I8: "I8",
    GGMLType.I16: "I16",
    GGMLType.I32: "I32",
    GGMLType.I64: "I64",
    GGMLType.F64: "F64",
}


def _align_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) // alignment * alignment


def _prod(xs: Iterable[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def _ggml_nbytes(tensor_type: int, n_elems: int) -> int:
    # GGML quantized formats are block-based. We only implement enough to size tensors
    # correctly for the common types encountered in GGUF files.
    if tensor_type == GGMLType.F32:
        return n_elems * 4
    if tensor_type == GGMLType.F16:
        return n_elems * 2
    if tensor_type == GGMLType.F64:
        return n_elems * 8
    if tensor_type == GGMLType.I8:
        return n_elems * 1
    if tensor_type == GGMLType.I16:
        return n_elems * 2
    if tensor_type == GGMLType.I32:
        return n_elems * 4
    if tensor_type == GGMLType.I64:
        return n_elems * 8

    # The block sizes below match ggml conventions (may not cover every type/version).
    # If sizing fails, we still keep offsets so you can inspect metadata safely.
    if tensor_type == GGMLType.Q4_0:
        # block size 32, 18 bytes per block
        return (n_elems // 32) * 18
    if tensor_type == GGMLType.Q4_1:
        # block size 32, 20 bytes per block
        return (n_elems // 32) * 20
    if tensor_type == GGMLType.Q5_0:
        # block size 32, 22 bytes per block
        return (n_elems // 32) * 22
    if tensor_type == GGMLType.Q5_1:
        # block size 32, 24 bytes per block
        return (n_elems // 32) * 24
    if tensor_type == GGMLType.Q8_0:
        # block size 32, 34 bytes per block
        return (n_elems // 32) * 34

    # K-quants use block size 256; sizes vary by type.
    if tensor_type == GGMLType.Q2_K:
        return (n_elems // 256) * 84
    if tensor_type == GGMLType.Q3_K:
        return (n_elems // 256) * 110
    if tensor_type == GGMLType.Q4_K:
        return (n_elems // 256) * 144
    if tensor_type == GGMLType.Q5_K:
        return (n_elems // 256) * 176
    if tensor_type == GGMLType.Q6_K:
        return (n_elems // 256) * 210
    if tensor_type == GGMLType.Q8_K:
        return (n_elems // 256) * 288

    return -1


@dataclass(frozen=True)
class GGUFTensor:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int  # relative to data section

    @property
    def n_elems(self) -> int:
        return _prod(self.shape)

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAME.get(self.ggml_type, f"TYPE_{self.ggml_type}")


@dataclass(frozen=True)
class GGUFFile:
    path: Path
    version: int
    kv: dict[str, Any]
    tensors: list[GGUFTensor]
    data_start: int


def read_gguf(path: Path) -> GGUFFile:
    with path.open("rb") as f:
        magic = _read_exact(f, 4)
        if magic != GGUF_MAGIC:
            raise GGUFError(f"Not a GGUF file (magic={magic!r})")
        version = _u32(f)
        n_tensors = _u64(f)
        n_kv = _u64(f)

        kv: dict[str, Any] = {}
        for _ in range(n_kv):
            key = _gguf_string(f)
            vtype = _u32(f)
            kv[key] = _read_kv_value(f, vtype)

        tensors: list[GGUFTensor] = []
        for _ in range(n_tensors):
            name = _gguf_string(f)
            n_dims = _u32(f)
            dims = tuple(int(_u64(f)) for _ in range(n_dims))
            ggml_type = _u32(f)
            offset = _u64(f)
            tensors.append(GGUFTensor(name=name, shape=dims, ggml_type=ggml_type, offset=offset))

        data_start = _align_up(f.tell(), 32)
        return GGUFFile(path=path, version=version, kv=kv, tensors=tensors, data_start=data_start)


def _human_bytes(n: int) -> str:
    if n < 0:
        return "?"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(x)} {u}"
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} TiB"


def _infer_layer_id(tensor_name: str) -> int | None:
    # Most transformer GGUF files use blk.<n>.*
    m = re.match(r"^blk\.(\d+)\.", tensor_name)
    if m:
        return int(m.group(1))
    return None


def _summarize_arch(gguf: GGUFFile) -> dict[str, Any]:
    kv = gguf.kv
    out: dict[str, Any] = {}

    def take(*keys: str) -> None:
        for k in keys:
            if k in kv:
                out[k] = kv[k]

    take(
        "general.architecture",
        "general.name",
        "general.description",
        "general.file_type",
        "tokenizer.ggml.model",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
        "llama.context_length",
        "llama.embedding_length",
        "llama.block_count",
        "llama.feed_forward_length",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.rope.freq_base",
        "llama.rope.dimension_count",
        "llama.vocab_size",
    )

    # Some models use non-llama prefixes (qwen2/qwen3). Keep anything that looks like a core hyperparam.
    for k in sorted(kv.keys()):
        if any(
            k.startswith(prefix)
            for prefix in (
                "qwen",
                "transformer.",
                "model.",
                "attention.",
                "rope.",
            )
        ):
            if k not in out:
                out[k] = kv[k]

    # Infer from tensors if missing.
    if "llama.block_count" not in out:
        layer_ids = [lid for t in gguf.tensors if (lid := _infer_layer_id(t.name)) is not None]
        if layer_ids:
            out["inferred.block_count"] = max(layer_ids) + 1

    return out


def _group_tensors_by_layer(tensors: list[GGUFTensor]) -> dict[int, list[GGUFTensor]]:
    layers: dict[int, list[GGUFTensor]] = {}
    for t in tensors:
        lid = _infer_layer_id(t.name)
        if lid is None:
            continue
        layers.setdefault(lid, []).append(t)
    for lid in layers:
        layers[lid].sort(key=lambda x: x.name)
    return dict(sorted(layers.items(), key=lambda kv: kv[0]))


def _tensor_bytes(gguf: GGUFFile, t: GGUFTensor) -> int:
    nbytes = _ggml_nbytes(t.ggml_type, t.n_elems)
    return nbytes


def _tensor_file_span(gguf: GGUFFile, t: GGUFTensor) -> tuple[int, int]:
    start = gguf.data_start + t.offset
    nbytes = _tensor_bytes(gguf, t)
    if nbytes < 0:
        return (start, start)
    return (start, start + nbytes)


def _dtype_size_summary(gguf: GGUFFile) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    total_bytes = 0
    total_elems = 0
    unknown = 0
    for t in gguf.tensors:
        tn = t.type_name
        b = _tensor_bytes(gguf, t)
        if b < 0:
            unknown += 1
            b = 0
        total_bytes += b
        total_elems += t.n_elems
        ent = by_type.setdefault(tn, {"tensors": 0, "elems": 0, "bytes": 0})
        ent["tensors"] += 1
        ent["elems"] += t.n_elems
        ent["bytes"] += b

    return {
        "total_tensors": len(gguf.tensors),
        "total_elems": total_elems,
        "total_bytes_est": total_bytes,
        "unknown_sized_tensors": unknown,
        "by_type": dict(sorted(by_type.items(), key=lambda kv: kv[1]["bytes"], reverse=True)),
    }


def _safe_import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise GGUFError("numpy is required for value histograms; install it or run without --values") from e
    return np


def _float_value_stats(
    gguf: GGUFFile,
    tensor: GGUFTensor,
    *,
    max_values: int,
    seed: int,
) -> dict[str, Any]:
    if tensor.ggml_type not in (GGMLType.F16, GGMLType.F32, GGMLType.F64):
        raise GGUFError(f"Tensor {tensor.name} is not a float tensor (type={tensor.type_name})")

    np = _safe_import_numpy()
    start, end = _tensor_file_span(gguf, tensor)
    if end <= start:
        raise GGUFError(f"Unable to locate tensor bytes for {tensor.name} (unsupported type sizing?)")

    dtype = {GGMLType.F16: np.float16, GGMLType.F32: np.float32, GGMLType.F64: np.float64}[tensor.ggml_type]
    itemsize = int(np.dtype(dtype).itemsize)
    n = tensor.n_elems
    if n <= 0:
        raise GGUFError(f"Tensor {tensor.name} has no elements")

    # Sample roughly uniformly without reading the entire tensor.
    max_values = min(int(max_values), int(n))
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(n, size=max_values, replace=False) if max_values < n else np.arange(n, dtype=np.int64)
    idx.sort()

    values = np.empty((max_values,), dtype=np.float64)
    with gguf.path.open("rb") as f:
        for j, i in enumerate(idx.tolist()):
            f.seek(start + i * itemsize)
            values[j] = np.frombuffer(_read_exact(f, itemsize), dtype=dtype, count=1)[0].item()

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise GGUFError(f"No finite values sampled from {tensor.name}")

    def pct(p: float) -> float:
        return float(np.percentile(finite, p))

    return {
        "tensor": tensor.name,
        "type": tensor.type_name,
        "shape": list(tensor.shape),
        "sampled": int(values.size),
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p01": pct(1),
        "p05": pct(5),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
    }


def _byte_histogram(
    gguf: GGUFFile,
    tensor: GGUFTensor,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    np = _safe_import_numpy()
    start, end = _tensor_file_span(gguf, tensor)
    if end <= start:
        raise GGUFError(f"Unable to locate tensor bytes for {tensor.name} (unsupported type sizing?)")
    size = end - start
    to_read = min(int(max_bytes), int(size))
    with gguf.path.open("rb") as f:
        f.seek(start)
        data = _read_exact(f, to_read)
    arr = np.frombuffer(data, dtype=np.uint8)
    hist = np.bincount(arr, minlength=256)
    top = np.argsort(hist)[::-1][:10]
    return {
        "tensor": tensor.name,
        "type": tensor.type_name,
        "shape": list(tensor.shape),
        "sampled_bytes": int(to_read),
        "top_bytes": [{"byte": int(b), "count": int(hist[b])} for b in top.tolist()],
    }


def _print_architecture(gguf: GGUFFile, *, max_layers: int | None) -> None:
    arch = _summarize_arch(gguf)
    print(f"GGUF: {gguf.path}")
    print(f"Version: {gguf.version}")
    if "general.architecture" in arch:
        print(f"Architecture: {arch.get('general.architecture')}")
    if "general.name" in arch:
        print(f"Name: {arch.get('general.name')}")
    if "general.file_type" in arch:
        print(f"File type: {arch.get('general.file_type')}")

    # Core hyperparams: print a stable subset if present.
    for k in (
        "llama.vocab_size",
        "llama.context_length",
        "llama.embedding_length",
        "llama.feed_forward_length",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.rope.dimension_count",
        "llama.rope.freq_base",
        "llama.block_count",
        "inferred.block_count",
    ):
        if k in arch:
            print(f"{k}: {arch[k]}")

    for k in (
        "qwen3.vocab_size",
        "qwen3.context_length",
        "qwen3.embedding_length",
        "qwen3.feed_forward_length",
        "qwen3.attention.head_count",
        "qwen3.attention.head_count_kv",
        "qwen3.attention.key_length",
        "qwen3.attention.value_length",
        "qwen3.rope.freq_base",
        "qwen3.block_count",
    ):
        if k in arch:
            print(f"{k}: {arch[k]}")

    non_block = [t for t in gguf.tensors if _infer_layer_id(t.name) is None]
    if non_block:
        print(f"\nNon-block tensors: {len(non_block)}")
        for t in sorted(non_block, key=lambda x: x.name):
            nbytes = _tensor_bytes(gguf, t)
            nbytes_s = _human_bytes(nbytes) if nbytes >= 0 else "?"
            print(f"  - {t.name} :: {t.type_name} {list(t.shape)} ({nbytes_s})")

    layers = _group_tensors_by_layer(gguf.tensors)
    if layers:
        lids = list(layers.keys())
        shown = lids if max_layers is None else lids[: max_layers]
        print(f"Layers (from tensors): {len(lids)}")
        for lid in shown:
            print(f"\nblk.{lid}: {len(layers[lid])} tensors")
            for t in layers[lid]:
                nbytes = _tensor_bytes(gguf, t)
                nbytes_s = _human_bytes(nbytes) if nbytes >= 0 else "?"
                print(f"  - {t.name} :: {t.type_name} {list(t.shape)} ({nbytes_s})")
        if max_layers is not None and len(lids) > max_layers:
            print(f"\n... ({len(lids) - max_layers} more layers omitted; use --max-layers to show more)")
    else:
        print("No blk.<n> tensors found; run --list-tensors for full list.")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _kv_for_report(kv: dict[str, Any], *, max_list_items: int = 200, max_string_len: int = 16_384) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in kv.items():
        if isinstance(v, list) and len(v) > max_list_items:
            out[k] = {"__type__": "array", "items": len(v)}
            continue
        if isinstance(v, str) and len(v) > max_string_len:
            out[k] = {"__type__": "string", "chars": len(v)}
            continue
        out[k] = v
    return out


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Inspect a .gguf file: metadata, tensor table, and lightweight stats (no dequant by default)."
    )
    p.add_argument(
        "gguf_path",
        nargs="?",
        help="Path to a .gguf file",
        default=os.environ.get("GGUF_PATH"),
    )
    p.add_argument("--list-kv", action="store_true", help="Print all key/values")
    p.add_argument("--list-tensors", action="store_true", help="Print all tensors (name/type/shape/bytes)")
    p.add_argument("--arch", action="store_true", help="Print an architecture-style summary (by layers)")
    p.add_argument("--max-layers", type=int, default=4, help="Limit layers shown in --arch (default: 4; use 0 for all)")
    p.add_argument("--json-out", type=str, default="", help="Write a JSON report to this path")

    stats = p.add_argument_group("stats")
    stats.add_argument("--tensor-regex", type=str, default="", help="Regex filter for tensors when computing stats")
    stats.add_argument(
        "--values",
        action="store_true",
        help="Compute value stats for float tensors (F16/F32/F64); samples without reading full tensors",
    )
    stats.add_argument("--max-values", type=int, default=200_000, help="Max sampled values per float tensor")
    stats.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    stats.add_argument(
        "--bytes",
        action="store_true",
        help="Compute raw byte histograms for selected tensors (useful for quantized weights; not dequantized)",
    )
    stats.add_argument("--max-bytes", type=int, default=8 * 1024 * 1024, help="Max bytes to sample per tensor")

    args = p.parse_args(argv)
    if not args.gguf_path:
        p.error("gguf_path is required (or set GGUF_PATH)")

    gguf_path = Path(args.gguf_path)
    if not gguf_path.exists():
        raise GGUFError(f"File not found: {gguf_path}")

    gguf = read_gguf(gguf_path)
    report: dict[str, Any] = {
        "path": str(gguf.path),
        "version": gguf.version,
        "data_start": gguf.data_start,
        "kv": _kv_for_report(gguf.kv),
        "arch": _summarize_arch(gguf),
        "dtype_summary": _dtype_size_summary(gguf),
        "tensors": [
            {
                "name": t.name,
                "shape": list(t.shape),
                "type": t.type_name,
                "ggml_type": t.ggml_type,
                "n_elems": t.n_elems,
                "offset": t.offset,
                "bytes_est": _tensor_bytes(gguf, t),
            }
            for t in gguf.tensors
        ],
    }

    if args.list_kv:
        for k in sorted(gguf.kv.keys()):
            v = gguf.kv[k]
            if isinstance(v, list) and len(v) > 20:
                print(f"{k} = [.. {len(v)} items ..]")
            else:
                print(f"{k} = {v!r}")

    if args.list_tensors:
        for t in gguf.tensors:
            nbytes = _tensor_bytes(gguf, t)
            nbytes_s = _human_bytes(nbytes) if nbytes >= 0 else "?"
            print(f"{t.name} :: {t.type_name} {list(t.shape)} ({nbytes_s}) @+{t.offset}")

    if args.arch:
        max_layers = None if args.max_layers == 0 else max(0, int(args.max_layers))
        _print_architecture(gguf, max_layers=max_layers)

    if args.tensor_regex and (args.values or args.bytes):
        rx = re.compile(args.tensor_regex)
        selected = [t for t in gguf.tensors if rx.search(t.name)]
        report["stats"] = {"tensor_regex": args.tensor_regex, "selected": len(selected), "items": []}
        for t in selected:
            item: dict[str, Any] = {"name": t.name, "type": t.type_name, "shape": list(t.shape)}
            if args.values and t.ggml_type in (GGMLType.F16, GGMLType.F32, GGMLType.F64):
                try:
                    item["value_stats"] = _float_value_stats(gguf, t, max_values=args.max_values, seed=args.seed)
                except GGUFError as e:
                    item["value_stats_error"] = str(e)
            if args.bytes:
                try:
                    item["byte_hist"] = _byte_histogram(gguf, t, max_bytes=args.max_bytes)
                except GGUFError as e:
                    item["byte_hist_error"] = str(e)
            report["stats"]["items"].append(item)

    if args.json_out:
        _write_json(Path(args.json_out), report)

    # If user didn't ask for anything, default to a small arch printout.
    if not (args.list_kv or args.list_tensors or args.arch or args.json_out or args.values or args.bytes):
        _print_architecture(gguf, max_layers=4)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except GGUFError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
