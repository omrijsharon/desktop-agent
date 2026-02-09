"""desktop_agent.chat_store

Persistent multi-chat storage for the chat UI.

Each chat is saved as a single JSON file under `<repo>/chat_history/`.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{6,80}$")


def new_chat_id() -> str:
    # e.g. 20260206T012233Z_ab12cd34
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{secrets.token_hex(4)}"


def _validate_chat_id(chat_id: str) -> str:
    if not isinstance(chat_id, str) or not _SAFE_ID_RE.match(chat_id):
        raise ValueError("Invalid chat_id")
    return chat_id


@dataclass(frozen=True)
class ChatMeta:
    chat_id: str
    title: str
    updated_at: str


def ensure_store_dir(root: Path) -> Path:
    d = (root / "chat_history").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def chat_path(root: Path, chat_id: str) -> Path:
    cid = _validate_chat_id(chat_id)
    return ensure_store_dir(root) / f"{cid}.json"


def list_chats(root: Path) -> list[ChatMeta]:
    d = ensure_store_dir(root)
    out: list[ChatMeta] = []
    for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            cid = str(data.get("chat_id") or p.stem)
            title = str(data.get("title") or "Untitled")
            updated_at = str(data.get("updated_at") or "")
            if not updated_at:
                updated_at = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
            out.append(ChatMeta(chat_id=cid, title=title, updated_at=updated_at))
        except Exception:
            # Skip unreadable entries.
            continue
    return out


def load_chat(root: Path, chat_id: str) -> JsonDict:
    p = chat_path(root, chat_id)
    return json.loads(p.read_text(encoding="utf-8"))


def save_chat(root: Path, chat: JsonDict) -> None:
    cid = _validate_chat_id(str(chat.get("chat_id", "")))
    p = chat_path(root, cid)
    chat = dict(chat)
    chat.setdefault("created_at", _utc_now_iso())
    chat["updated_at"] = _utc_now_iso()
    p.write_text(json.dumps(chat, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def delete_chat(root: Path, chat_id: str) -> None:
    p = chat_path(root, chat_id)
    if p.exists():
        p.unlink()

