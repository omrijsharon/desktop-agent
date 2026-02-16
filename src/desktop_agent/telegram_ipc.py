"""desktop_agent.telegram_ipc

File-based IPC between the Chat UI process and a headless Telegram relay.

Why file IPC?
- Works across processes without additional dependencies.
- Easy to debug (JSON files on disk).
- Safe default: the UI never needs the bot token; the relay never touches OpenAI.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def telegram_root(repo_root: Path) -> Path:
    return (repo_root / "chat_history" / "telegram").resolve()


def telegram_inbox_dir(repo_root: Path) -> Path:
    return telegram_root(repo_root) / "inbox"


def telegram_outbox_dir(repo_root: Path) -> Path:
    return telegram_root(repo_root) / "outbox"


def ensure_telegram_dirs(repo_root: Path) -> tuple[Path, Path]:
    inbox = telegram_inbox_dir(repo_root)
    outbox = telegram_outbox_dir(repo_root)
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    return inbox, outbox


@dataclass(frozen=True)
class TelegramInboundMessage:
    chat_id: int
    message_id: int
    date: int
    from_name: str
    from_username: str
    text: str
    update_id: Optional[int] = None


def write_outbox_message(*, repo_root: Path, chat_id: int, text: str) -> Path:
    """Queue a message to be sent to Telegram by the relay."""
    _, outbox = ensure_telegram_dirs(repo_root)
    payload = {
        "v": 1,
        "ts": time.time(),
        "chat_id": int(chat_id),
        "text": str(text or ""),
    }
    name = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}.json"
    p = outbox / name
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def read_inbox_message(path: Path) -> TelegramInboundMessage:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError("inbox message must be an object")
    return TelegramInboundMessage(
        chat_id=int(d.get("chat_id")),
        message_id=int(d.get("message_id")),
        date=int(d.get("date") or 0),
        from_name=str(d.get("from_name") or ""),
        from_username=str(d.get("from_username") or ""),
        text=str(d.get("text") or ""),
        update_id=(int(d["update_id"]) if "update_id" in d and d["update_id"] is not None else None),
    )


def safe_list_json_files(dir_path: Path) -> list[Path]:
    try:
        if not dir_path.exists():
            return []
        return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    except Exception:
        return []

