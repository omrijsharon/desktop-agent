"""desktop_agent.telegram_relay

Headless Telegram Bot API relay.

It bridges messages between Telegram and the Chat UI using file-based IPC:
- Inbound (Telegram -> UI): writes JSON files to `chat_history/telegram/inbox/`
- Outbound (UI -> Telegram): reads JSON files from `chat_history/telegram/outbox/`

Security:
- Bot token is read from env var `TELEGRAM_BOT_TOKEN` (e.g. from `.env`).
- Messages are restricted to a single allowlisted group chat_id.
  To set it, send `/allow_here` in the target Telegram group.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from .telegram_ipc import ensure_telegram_dirs, safe_list_json_files, telegram_root


_LOG = logging.getLogger("desktop_agent.telegram_relay")


def _setup_logging(repo_root: Path) -> None:
    """Best-effort logging to file + stderr (no external deps)."""

    try:
        log_dir = (repo_root / "chat_history" / "telegram").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "telegram_relay.log"

        root = logging.getLogger()
        root.setLevel(logging.INFO)

        fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S")

        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        root.addHandler(sh)

        _LOG.info("telegram_relay_log_path=%s", str(log_path))
    except Exception:
        logging.basicConfig(level=logging.INFO)


@dataclass
class RelayState:
    allowed_chat_id: Optional[int] = None
    last_update_id: int = 0


def _state_path(repo_root: Path) -> Path:
    return telegram_root(repo_root) / "relay_state.json"


def load_state(repo_root: Path) -> RelayState:
    p = _state_path(repo_root)
    try:
        if not p.exists():
            return RelayState()
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            return RelayState()
        return RelayState(
            allowed_chat_id=(int(d["allowed_chat_id"]) if d.get("allowed_chat_id") is not None else None),
            last_update_id=int(d.get("last_update_id") or 0),
        )
    except Exception:
        return RelayState()


def save_state(repo_root: Path, st: RelayState) -> None:
    p = _state_path(repo_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = {"allowed_chat_id": st.allowed_chat_id, "last_update_id": int(st.last_update_id)}
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


class TelegramBotApi:
    def __init__(self, *, token: str, timeout_s: float = 45.0) -> None:
        self._token = token
        self._base = f"https://api.telegram.org/bot{token}"
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def get_updates(self, *, offset: int, timeout_s: int) -> list[dict[str, Any]]:
        r = self._client.get(
            f"{self._base}/getUpdates",
            params={"offset": int(offset), "timeout": int(timeout_s), "allowed_updates": ["message"]},
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict) or not data.get("ok"):
            raise RuntimeError(f"getUpdates failed: {data!r}")
        res = data.get("result")
        return [x for x in res if isinstance(x, dict)] if isinstance(res, list) else []

    def send_message(self, *, chat_id: int, text: str) -> None:
        r = self._client.post(f"{self._base}/sendMessage", data={"chat_id": int(chat_id), "text": str(text or "")})
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict) or not data.get("ok"):
            raise RuntimeError(f"sendMessage failed: {data!r}")


def _extract_text(msg: dict[str, Any]) -> str:
    t = msg.get("text")
    if isinstance(t, str) and t.strip():
        return t
    # Ignore non-text for now.
    return ""


def _msg_from_name(msg: dict[str, Any]) -> tuple[str, str, bool]:
    fr = msg.get("from") if isinstance(msg.get("from"), dict) else {}
    is_bot = bool(fr.get("is_bot"))
    username = str(fr.get("username") or "")
    first = str(fr.get("first_name") or "")
    last = str(fr.get("last_name") or "")
    nm = (first + " " + last).strip() or username or ("bot" if is_bot else "user")
    return nm, username, is_bot


def _write_inbox(repo_root: Path, payload: dict[str, Any]) -> None:
    inbox, _ = ensure_telegram_dirs(repo_root)
    name = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{payload.get('update_id', 0)}.json"
    p = inbox / name
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _drain_outbox(*, api: TelegramBotApi, repo_root: Path, allowed_chat_id: int) -> int:
    _, outbox = ensure_telegram_dirs(repo_root)
    sent = 0
    for p in safe_list_json_files(outbox):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                p.unlink(missing_ok=True)
                continue
            chat_id = int(d.get("chat_id") or 0)
            text = str(d.get("text") or "")
            if not text.strip():
                p.unlink(missing_ok=True)
                continue
            if chat_id != int(allowed_chat_id):
                # Safety: never send to unknown chats.
                _LOG.warning("drop_outbox_wrong_chat file=%s chat_id=%s allowed=%s", p.name, chat_id, allowed_chat_id)
                p.unlink(missing_ok=True)
                continue
            api.send_message(chat_id=int(allowed_chat_id), text=text)
            sent += 1
            p.unlink(missing_ok=True)
        except Exception as e:
            _LOG.error("send_outbox_failed file=%s error=%s", p.name, f"{type(e).__name__}: {e}")
            # Leave file for retry.
            continue
    return sent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="desktop_agent.telegram_relay")
    ap.add_argument("--repo-root", type=str, default=str(Path.cwd()), help="Repo root (default: cwd)")
    ap.add_argument("--poll-timeout-s", type=int, default=25, help="Telegram long-poll timeout")
    ap.add_argument("--loop-sleep-s", type=float, default=0.2, help="Small delay between loops")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    _setup_logging(repo_root)
    ensure_telegram_dirs(repo_root)

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    if not token.strip():
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN (set it in .env or env vars).")

    st = load_state(repo_root)
    api = TelegramBotApi(token=token)
    _LOG.info("telegram_relay_start repo_root=%s state=%s", str(repo_root), st)

    try:
        while True:
            # Outbound: UI -> Telegram
            if st.allowed_chat_id is not None:
                _drain_outbox(api=api, repo_root=repo_root, allowed_chat_id=int(st.allowed_chat_id))

            # Inbound: Telegram -> UI
            offset = int(st.last_update_id) + 1 if int(st.last_update_id) > 0 else 0
            updates = api.get_updates(offset=offset, timeout_s=int(args.poll_timeout_s))
            for up in updates:
                try:
                    uid = int(up.get("update_id") or 0)
                    st.last_update_id = max(int(st.last_update_id), uid)

                    msg = up.get("message")
                    if not isinstance(msg, dict):
                        continue
                    chat = msg.get("chat")
                    if not isinstance(chat, dict):
                        continue
                    chat_id = int(chat.get("id") or 0)
                    text = _extract_text(msg)
                    if not text:
                        continue
                    from_name, from_username, is_bot = _msg_from_name(msg)
                    if is_bot:
                        continue

                    if st.allowed_chat_id is None:
                        if text.strip() == "/allow_here":
                            st.allowed_chat_id = int(chat_id)
                            save_state(repo_root, st)
                            api.send_message(chat_id=int(chat_id), text=f"Allowed chat_id set to {chat_id}.")
                            _LOG.info("telegram_allowed_chat_set chat_id=%s", chat_id)
                        else:
                            # Not allowlisted yet: explain once in the group.
                            api.send_message(
                                chat_id=int(chat_id),
                                text="This relay is not configured yet. Send /allow_here in the target group to bind it.",
                            )
                        continue

                    if int(chat_id) != int(st.allowed_chat_id):
                        continue

                    payload = {
                        "v": 1,
                        "update_id": uid,
                        "chat_id": int(chat_id),
                        "message_id": int(msg.get("message_id") or 0),
                        "date": int(msg.get("date") or 0),
                        "from_name": str(from_name or ""),
                        "from_username": str(from_username or ""),
                        "text": str(text),
                    }
                    _write_inbox(repo_root, payload)
                finally:
                    save_state(repo_root, st)

            time.sleep(float(args.loop_sleep_s))
    except KeyboardInterrupt:
        _LOG.info("telegram_relay_stop keyboard_interrupt")
        return 0
    finally:
        save_state(repo_root, st)
        api.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
