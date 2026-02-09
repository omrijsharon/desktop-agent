from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_file(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")
    return p


def read_tail(path: str | Path, *, max_chars: int = 6000) -> str:
    p = ensure_file(path)
    txt = p.read_text(encoding="utf-8")
    if len(txt) <= max_chars:
        return txt
    return txt[-max_chars:]


def append_md_section(path: str | Path, *, title: str, body: str) -> None:
    p = ensure_file(path)
    title = title.strip()
    body = (body or "").rstrip()
    chunk = f"\n\n## {title}\n\n{body}\n"
    p.write_text(p.read_text(encoding="utf-8") + chunk, encoding="utf-8")


@dataclass(frozen=True)
class MemoryPaths:
    riddler: Path
    actor: Path

