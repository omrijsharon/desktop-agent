from __future__ import annotations

import html
import json
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from .chat_session import ChatConfig, ChatSession
from .tools import make_playwright_browser_handler


JsonDict = dict[str, Any]


def _tool_spec_ask_user() -> JsonDict:
    return {
        "type": "function",
        "name": "ask_user",
        "description": (
            "Ask the human for information (optionally secret) and return it. "
            "Use this when credentials or one-time confirmation is needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "What to ask the user."},
                "secret": {"type": "boolean", "description": "If true, mask input (password).", "default": False},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    }


class _AskUserBridge(QtCore.QObject):
    request = QtCore.Signal(str, bool, str)  # prompt, secret, request_id

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self.request.connect(self._on_request, QtCore.Qt.ConnectionType.QueuedConnection)

    def ask(self, prompt: str, *, secret: bool, timeout_s: float = 600.0) -> str:
        req_id = uuid.uuid4().hex
        ev = threading.Event()
        box: dict[str, Any] = {"ok": False, "value": ""}
        with self._lock:
            self._pending[req_id] = (ev, box)
        self.request.emit(prompt, bool(secret), req_id)
        if not ev.wait(timeout=timeout_s):
            with self._lock:
                self._pending.pop(req_id, None)
            raise RuntimeError("Timed out waiting for user input")
        with self._lock:
            _, box2 = self._pending.pop(req_id, (ev, box))
        if not box2.get("ok", False):
            raise RuntimeError("User cancelled")
        return str(box2.get("value") or "")

    @QtCore.Slot(str, bool, str)
    def _on_request(self, prompt: str, secret: bool, request_id: str) -> None:
        text, ok = self._prompt_dialog(prompt=prompt, secret=secret)
        with self._lock:
            tup = self._pending.get(request_id)
        if tup is None:
            return
        ev, box = tup
        box["ok"] = bool(ok)
        box["value"] = str(text or "")
        ev.set()

    def _prompt_dialog(self, *, prompt: str, secret: bool) -> tuple[str, bool]:
        dlg = QtWidgets.QInputDialog()
        dlg.setWindowTitle("Browser Helper")
        dlg.setLabelText(prompt)
        dlg.setTextEchoMode(
            QtWidgets.QLineEdit.EchoMode.Password if secret else QtWidgets.QLineEdit.EchoMode.Normal
        )
        ok = dlg.exec()
        return dlg.textValue(), bool(ok)


_IMAGE_RE = re.compile(r"\\[\\[image:(.+?)\\]\\]")


def _render_text_with_images(text: str, *, repo_root: Path) -> str:
    # Convert plain text to basic HTML and inline any [[image:...]] markers.
    parts: list[str] = []
    last = 0
    for m in _IMAGE_RE.finditer(text or ""):
        parts.append(html.escape((text or "")[last : m.start()]))
        raw = (m.group(1) or "").strip()
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            if p.exists():
                uri = p.as_uri()
                parts.append(f'<div style="margin:8px 0;"><img src="{uri}" style="max-width:100%; border-radius:12px;"></div>')
            else:
                parts.append(html.escape(f"[missing image: {raw}]"))
        last = m.end()
    parts.append(html.escape((text or "")[last:]))
    body = "".join(parts)
    body = body.replace("\n", "<br>")
    return f"<div style='white-space:pre-wrap'>{body}</div>"


@dataclass
class _UiMessage:
    role: str  # user|assistant|event
    text: str


class _ChatWorker(QtCore.QThread):
    assistant_delta = QtCore.Signal(str)
    assistant_done = QtCore.Signal(str)
    event_line = QtCore.Signal(str)
    error = QtCore.Signal(str)
    busy = QtCore.Signal(bool)

    def __init__(self, *, session: ChatSession, user_text: str, show_tool_events: bool) -> None:
        super().__init__()
        self._session = session
        self._user_text = user_text
        self._show_tool_events = show_tool_events

    def run(self) -> None:  # type: ignore[override]
        self.busy.emit(True)
        try:
            full = ""
            last_emit = 0.0
            new_items: list[JsonDict] = []
            for ev in self._session.send_stream(self._user_text):
                if self.isInterruptionRequested():
                    return
                et = ev.get("type")
                if et == "assistant_delta":
                    d = ev.get("delta")
                    if isinstance(d, str) and d:
                        full += d
                    now = time.monotonic()
                    if now - last_emit >= 0.05:
                        self.assistant_delta.emit(full)
                        last_emit = now
                elif et == "turn_done":
                    ni = ev.get("new_items")
                    if isinstance(ni, list):
                        new_items = [x for x in ni if isinstance(x, dict)]

            if self.isInterruptionRequested():
                return

            # Surface any images produced by tools (e.g. playwright screenshots) by appending markers.
            tool_images: list[str] = []
            for item in new_items:
                if item.get("type") != "function_call_output":
                    continue
                out_full = item.get("output", "")
                if not (isinstance(out_full, str) and out_full.strip().startswith("{")):
                    continue
                try:
                    d = json.loads(out_full)
                except Exception:
                    continue
                if not isinstance(d, dict):
                    continue
                imgs = d.get("image_paths")
                if not isinstance(imgs, list):
                    continue
                for p in imgs:
                    if isinstance(p, str) and p.strip() and p.strip() not in tool_images:
                        tool_images.append(p.strip())

            if self._show_tool_events:
                for item in new_items:
                    t = item.get("type")
                    if t == "function_call":
                        name = item.get("name", "")
                        args = item.get("arguments", "")
                        self.event_line.emit(f"[tool call] {name}({args})")
                    elif t == "function_call_output":
                        out = item.get("output", "")
                        out_s = out if isinstance(out, str) else str(out)
                        if len(out_s) > 2000:
                            out_s = out_s[:2000] + " …"
                        self.event_line.emit(f"[tool output] {out_s}")
                    elif t == "web_search_call":
                        action = item.get("action") or {}
                        q = action.get("query") or ""
                        self.event_line.emit(f"[web_search] query={q!r}")
            final = full.strip()
            if tool_images:
                for p in tool_images[:6]:
                    final += f"\n\n[[image:{p}]]"
            self.assistant_done.emit(final)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.busy.emit(False)


class _BrowserWorker(QtCore.QObject):
    status = QtCore.Signal(str)
    event_line = QtCore.Signal(str)
    screenshot_path = QtCore.Signal(str)  # repo-relative path
    error = QtCore.Signal(str)
    ready = QtCore.Signal(bool)

    def __init__(self, *, pw_handler, home_url: str) -> None:
        super().__init__()
        self._pw_handler = pw_handler
        self._home_url = home_url
        self._q: "queue.Queue[tuple[str, dict[str, Any]]]" = queue.Queue()
        self._stop = threading.Event()
        self._screenshot_pending = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(("__stop__", {}))
        except Exception:
            pass

    def enqueue(self, action: str, params: dict[str, Any] | None = None) -> None:
        try:
            self._q.put_nowait((action, params or {}))
        except Exception:
            pass

    def request_screenshot(self) -> None:
        # Coalesce refreshes to avoid flooding the MCP server.
        if self._screenshot_pending.is_set():
            return
        self._screenshot_pending.set()
        self.enqueue("browser_take_screenshot", {"fullPage": False})

    def _call_pw(self, action: str, params: dict[str, Any]) -> JsonDict:
        out_s = self._pw_handler({"action": action, "params": params})
        d = json.loads(out_s)
        if not isinstance(d, dict) or not d.get("ok", False):
            raise RuntimeError(str(d))
        return d

    @QtCore.Slot()
    def run(self) -> None:
        # Important: this runs inside its own QThread so the GUI never freezes.
        self.status.emit("Browser: initializing…")
        self.event_line.emit("playwright: init")
        try:
            self._call_pw("browser_install", {})
            self._call_pw("browser_tabs", {"action": "new"})
            self._call_pw("browser_navigate", {"url": self._home_url})
            self.status.emit("Browser: ready")
            self.ready.emit(True)
            self.request_screenshot()
        except Exception as e:  # noqa: BLE001
            self.status.emit("Browser: error")
            self.error.emit(f"{type(e).__name__}: {e}")
            self.ready.emit(False)

        while not self._stop.is_set():
            try:
                action, params = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            if action == "__stop__":
                break

            is_shot = action == "browser_take_screenshot"
            if is_shot:
                self._screenshot_pending.clear()

            try:
                t0 = time.monotonic()
                if not is_shot:
                    self.event_line.emit(f"pw: {action}")
                if action == "browser_navigate":
                    self.status.emit("Browser: navigating…")
                res = self._call_pw(action, params)
                if action == "browser_navigate":
                    self.status.emit("Browser: ready")
                if not is_shot:
                    dt_ms = int((time.monotonic() - t0) * 1000.0)
                    self.event_line.emit(f"pw: {action} done ({dt_ms} ms)")
                if is_shot:
                    imgs = res.get("image_paths")
                    if isinstance(imgs, list):
                        p = next((x for x in imgs if isinstance(x, str) and x.strip()), None)
                        if p:
                            self.screenshot_path.emit(p.strip())
            except Exception as e:  # noqa: BLE001
                self.error.emit(f"{type(e).__name__}: {e}")
                if action == "browser_navigate":
                    self.status.emit("Browser: error")


class BrowserHelperWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self._repo_root = Path(__file__).resolve().parents[2]
        self._default_home_url = "https://www.google.com"

        self.setWindowTitle("Browser Helper")
        self.resize(1400, 900)

        self._ask_bridge = _AskUserBridge(self)

        cfg = ChatConfig()
        cfg.enable_playwright_browser = True
        cfg.allow_playwright_browser = True
        cfg.playwright_headless = False
        cfg.playwright_watch_mode = True
        cfg.playwright_screenshot_full_page = True
        cfg.allow_playwright_eval = True
        cfg.enable_web_search = True

        self._session = ChatSession(config=cfg)
        self._session.set_system_prompt(
            "\n".join(
                [
                    "You are Browser Helper, an assistant that can use a real browser to help the user do things online.",
                    "",
                    "You have a tool: playwright_browser(action, params). Use it to open pages, click, type, and take screenshots.",
                    "Prefer using watch mode screenshots so the user can see progress.",
                    "Assume the browser starts on google.com unless told otherwise.",
                    "",
                    "If credentials or one-time sensitive info is needed, call ask_user(prompt, secret=true) and then continue.",
                    "Never store credentials. Use them only for the immediate browser action.",
                ]
            )
        )

        def ask_user_handler(args: JsonDict) -> str:
            prompt = args.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                return json.dumps({"ok": False, "error": "prompt must be a non-empty string"})
            secret = bool(args.get("secret", False))
            try:
                val = self._ask_bridge.ask(prompt.strip(), secret=secret)
            except Exception as e:  # noqa: BLE001
                return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})
            return json.dumps({"ok": True, "value": val})

        self._session.registry.add(tool_spec=_tool_spec_ask_user(), handler=ask_user_handler)

        # Browser-first UX: keep a single rolling screenshot file (avoid endless PNG accumulation).
        try:
            pw_cmd = ["cmd.exe", "/c", "npx", "-y", "@playwright/mcp@latest"]
            if bool(self._session.cfg.playwright_headless):
                pw_cmd += ["--headless"]
            img_dir = (self._repo_root / "chat_history" / "browser_helper").resolve()
            pw_handler, pw_shutdown = make_playwright_browser_handler(
                cmd=pw_cmd,
                repo_root=self._repo_root,
                image_out_dir=img_dir,
                call_timeout_s=240.0,
                fixed_image_name="latest",
            )
            # Override the default handler (same tool spec).
            self._session.registry.handlers["playwright_browser"] = pw_handler
            # Keep restart_playwright working for this UI.
            self._session._playwright_shutdown = pw_shutdown  # type: ignore[attr-defined]
        except Exception:
            # If this fails, we fall back to the default handler created by ChatSession.
            pass

        self._busy = False
        self._worker: _ChatWorker | None = None
        self._assistant_live_id: int | None = None
        self._pw_handler = self._session.registry.handlers.get("playwright_browser")
        self._browser_timer: QtCore.QTimer | None = None
        self._last_screenshot_path: Path | None = None
        self._last_pixmap: QtGui.QPixmap | None = None
        self._browser_thread: QtCore.QThread | None = None
        self._browser_worker: _BrowserWorker | None = None

        self._build_ui()
        QtCore.QTimer.singleShot(0, self._start_browser_worker)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)

        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QtWidgets.QHBoxLayout()

        title = QtWidgets.QLabel("Browser Helper")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        header.addWidget(title)

        header.addStretch(1)

        self._model_lbl = QtWidgets.QLabel(f"Model: {self._session.cfg.model}")
        self._model_lbl.setStyleSheet("opacity: 0.85;")
        header.addWidget(self._model_lbl)

        self._chk_tool_events = QtWidgets.QCheckBox("Show tool events")
        self._chk_tool_events.setChecked(True)
        header.addWidget(self._chk_tool_events)

        btn_restart = QtWidgets.QPushButton("Restart browser")
        btn_restart.clicked.connect(self._restart_browser)
        header.addWidget(btn_restart)

        layout.addLayout(header)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)

        # Browser pane (dominant)
        browser_wrap = QtWidgets.QWidget()
        browser_layout = QtWidgets.QVBoxLayout(browser_wrap)
        browser_layout.setContentsMargins(0, 0, 0, 0)
        browser_layout.setSpacing(8)

        nav_row = QtWidgets.QHBoxLayout()
        self._btn_home = QtWidgets.QPushButton("Home")
        self._btn_home.clicked.connect(lambda: self._nav_to(self._default_home_url))
        nav_row.addWidget(self._btn_home)

        self._btn_back = QtWidgets.QPushButton("Back")
        self._btn_back.clicked.connect(self._nav_back)
        nav_row.addWidget(self._btn_back)

        self._btn_refresh = QtWidgets.QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self._request_refresh)
        nav_row.addWidget(self._btn_refresh)

        self._addr = QtWidgets.QLineEdit()
        self._addr.setPlaceholderText("Enter URL and press Enter…")
        self._addr.returnPressed.connect(self._nav_from_bar)
        nav_row.addWidget(self._addr, 1)

        self._lbl_status = QtWidgets.QLabel("Browser: starting…")
        self._lbl_status.setStyleSheet("opacity: 0.8;")
        nav_row.addWidget(self._lbl_status)

        browser_layout.addLayout(nav_row)

        self._browser_view = QtWidgets.QLabel()
        self._browser_view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._browser_view.setText("Waiting for first screenshot…")
        self._browser_view.setStyleSheet(
            "background: #0b0d10; color: #c9d2df; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;"
        )

        self._browser_scroll = QtWidgets.QScrollArea()
        self._browser_scroll.setWidgetResizable(True)
        self._browser_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._browser_scroll.setWidget(self._browser_view)
        browser_layout.addWidget(self._browser_scroll, 1)

        split.addWidget(browser_wrap)

        # Chat pane
        chat_wrap = QtWidgets.QWidget()
        chat_layout = QtWidgets.QVBoxLayout(chat_wrap)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(8)

        self._chat_view = QtWidgets.QTextBrowser()
        self._chat_view.setOpenExternalLinks(True)
        self._chat_view.setStyleSheet(
            """
            QTextBrowser { background: #0b0d10; color: #e8eef7; border: 1px solid rgba(255,255,255,0.08);
                           border-radius: 14px; padding: 12px; font-size: 14px; }
            """
        )
        chat_layout.addWidget(self._chat_view, 1)

        input_row = QtWidgets.QHBoxLayout()
        self._input = QtWidgets.QPlainTextEdit()
        self._input.setPlaceholderText("Ask Browser Helper what to do…")
        self._input.setFixedHeight(68)
        self._input.setStyleSheet(
            """
            QPlainTextEdit { background: rgba(255,255,255,0.06); color: #eef3fb; border: 1px solid rgba(255,255,255,0.12);
                             border-radius: 12px; padding: 10px; font-size: 14px; }
            """
        )
        input_row.addWidget(self._input, 1)

        self._btn_stop = QtWidgets.QPushButton("Stop")
        self._btn_stop.clicked.connect(self._stop)
        self._btn_stop.setEnabled(False)
        input_row.addWidget(self._btn_stop)

        btn_send = QtWidgets.QPushButton("Send")
        btn_send.clicked.connect(self._send)
        input_row.addWidget(btn_send)

        chat_layout.addLayout(input_row)
        split.addWidget(chat_wrap)

        # Events pane
        ev_wrap = QtWidgets.QWidget()
        ev_layout = QtWidgets.QVBoxLayout(ev_wrap)
        ev_layout.setContentsMargins(0, 0, 0, 0)
        ev_layout.setSpacing(8)

        ev_title = QtWidgets.QLabel("Events")
        ev_title.setStyleSheet("font-weight: 700;")
        ev_layout.addWidget(ev_title)

        self._ev_view = QtWidgets.QPlainTextEdit()
        self._ev_view.setReadOnly(True)
        self._ev_view.setStyleSheet(
            """
            QPlainTextEdit { background: #0b0d10; color: #c9d2df; border: 1px solid rgba(255,255,255,0.08);
                             border-radius: 14px; padding: 10px; font-size: 12px; }
            """
        )
        ev_layout.addWidget(self._ev_view, 1)
        split.addWidget(ev_wrap)

        split.setSizes([1200, 520, 340])

        root.setStyleSheet("QMainWindow, QWidget { background: #07090c; } QPushButton { padding: 8px 12px; }")

        self._append("assistant", "Hi—tell me what you want to do online and I’ll drive the browser.")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._stop_browser_worker()
        super().closeEvent(event)

    def _append(self, role: str, text: str) -> None:
        role_lbl = "You" if role == "user" else ("Assistant" if role == "assistant" else "Event")
        color = "#7db7ff" if role == "user" else ("#a6ffcb" if role == "assistant" else "#c9d2df")
        body = _render_text_with_images(text, repo_root=self._repo_root)
        block = (
            f"<div style='margin:10px 0;'>"
            f"<div style='font-weight:700; color:{color}; margin-bottom:4px'>{html.escape(role_lbl)}</div>"
            f"{body}"
            f"</div>"
        )
        self._chat_view.append(block)
        self._chat_view.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def _append_event(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._ev_view.appendPlainText(f"{ts} {text}")
        sb = self._ev_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _stop_browser_worker(self) -> None:
        w = self._browser_worker
        t = self._browser_thread
        self._browser_worker = None
        self._browser_thread = None
        if w is not None:
            try:
                w.stop()
            except Exception:
                pass
        if t is not None:
            try:
                t.quit()
                t.wait(1500)
            except Exception:
                pass

    def _start_browser_worker(self) -> None:
        # IMPORTANT: never call Playwright tools on the GUI thread (they can block for long navigations).
        self._stop_browser_worker()
        if self._pw_handler is None:
            self._append_event("playwright_browser tool not available")
            self._lbl_status.setText("Browser: error")
            return

        self._addr.setText(self._default_home_url)
        self._browser_thread = QtCore.QThread(self)
        self._browser_worker = _BrowserWorker(pw_handler=self._pw_handler, home_url=self._default_home_url)
        self._browser_worker.moveToThread(self._browser_thread)
        self._browser_thread.started.connect(self._browser_worker.run)  # type: ignore[arg-type]
        self._browser_worker.status.connect(self._lbl_status.setText)
        self._browser_worker.event_line.connect(self._append_event)
        self._browser_worker.error.connect(lambda msg: self._append_event(f"browser error: {msg}"))
        self._browser_worker.screenshot_path.connect(self._on_screenshot_path)
        self._browser_thread.start()

        if self._browser_timer is None:
            tmr = QtCore.QTimer(self)
            tmr.setInterval(900)
            tmr.timeout.connect(self._request_refresh)
            tmr.start()
            self._browser_timer = tmr
        QtCore.QTimer.singleShot(0, self._request_refresh)

    def _restart_browser(self) -> None:
        try:
            self._session.restart_playwright()
            self._append_event("playwright: restarted")
            self._lbl_status.setText("Browser: restarted")
            QtCore.QTimer.singleShot(0, self._start_browser_worker)
        except Exception as e:  # noqa: BLE001
            self._append_event(f"playwright restart failed: {type(e).__name__}: {e}")

    def _call_pw(self, action: str, params: dict[str, Any] | None = None) -> JsonDict:
        if self._pw_handler is None:
            raise RuntimeError("playwright_browser tool not installed")
        out_s = self._pw_handler({"action": action, "params": params or {}})
        d = json.loads(out_s)
        if not isinstance(d, dict) or not d.get("ok", False):
            raise RuntimeError(str(d))
        return d

    def _ensure_browser_ready(self) -> None:
        # Ensure the browser session exists and is on a useful start page.
        try:
            self._lbl_status.setText("Browser: initializing…")
            self._append_event("playwright: init")
            self._call_pw("browser_install", {})
            self._call_pw("browser_tabs", {"action": "new"})
            self._call_pw("browser_navigate", {"url": self._default_home_url})
            self._addr.setText(self._default_home_url)
            self._lbl_status.setText("Browser: ready")
        except Exception as e:  # noqa: BLE001
            self._lbl_status.setText("Browser: error")
            self._append_event(f"playwright init failed: {type(e).__name__}: {e}")
            return

        if self._browser_timer is None:
            t = QtCore.QTimer(self)
            t.setInterval(900)
            t.timeout.connect(self._refresh_view)
            t.start()
            self._browser_timer = t
        QtCore.QTimer.singleShot(0, self._refresh_view)

    def _nav_from_bar(self) -> None:
        raw = self._addr.text().strip()
        if not raw:
            return
        if "://" not in raw:
            raw = "https://" + raw
        self._nav_to(raw)

    def _nav_to(self, url: str) -> None:
        if self._browser_worker is None:
            self._append_event("navigate ignored: browser not ready")
            return
        self._addr.setText(url)
        self._append_event(f"navigate: {url}")
        self._browser_worker.enqueue("browser_navigate", {"url": url})
        self._browser_worker.request_screenshot()

    def _nav_back(self) -> None:
        if self._browser_worker is None:
            self._append_event("back ignored: browser not ready")
            return
        self._append_event("back")
        self._browser_worker.enqueue("browser_navigate_back", {})
        self._browser_worker.request_screenshot()

    def _refresh_view(self) -> None:
        # Backwards-compatible name for any existing signal connections.
        self._request_refresh()

    def _request_refresh(self) -> None:
        if self._browser_worker is not None:
            self._browser_worker.request_screenshot()

    @QtCore.Slot(str)
    def _on_screenshot_path(self, rel_path: str) -> None:
        try:
            abs_path = (self._repo_root / rel_path).resolve()
            if not abs_path.exists():
                return
            self._last_screenshot_path = abs_path
            pm = QtGui.QPixmap(str(abs_path))
            if pm.isNull():
                return
            self._last_pixmap = pm
            viewport_w = max(400, int(self._browser_scroll.viewport().width()) - 24)
            scaled = pm.scaledToWidth(viewport_w, QtCore.Qt.TransformationMode.SmoothTransformation)
            self._browser_view.setPixmap(scaled)
            self._browser_view.setMinimumSize(scaled.size())
        except Exception as e:  # noqa: BLE001
            self._append_event(f"render screenshot failed: {type(e).__name__}: {e}")

    def _stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._append_event("stop requested")

    def _send(self) -> None:
        if self._busy:
            return
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.setPlainText("")
        self._append("user", text)

        self._worker = _ChatWorker(session=self._session, user_text=text, show_tool_events=bool(self._chk_tool_events.isChecked()))
        self._worker.busy.connect(self._on_busy)
        self._worker.assistant_delta.connect(self._on_assistant_delta)
        self._worker.assistant_done.connect(self._on_assistant_done)
        self._worker.event_line.connect(self._append_event)
        self._worker.error.connect(self._on_error)
        self._assistant_live_id = None
        self._worker.start()

    @QtCore.Slot(bool)
    def _on_busy(self, b: bool) -> None:
        self._busy = bool(b)
        self._btn_stop.setEnabled(self._busy)

    @QtCore.Slot(str)
    def _on_assistant_delta(self, text: str) -> None:
        # Update last assistant message in-place by clearing and re-appending a "live" block.
        # For simplicity (and robustness), we just append the partial as a new block rarely.
        # The final message will be appended on done.
        pass

    @QtCore.Slot(str)
    def _on_assistant_done(self, text: str) -> None:
        self._append("assistant", text)

    @QtCore.Slot(str)
    def _on_error(self, msg: str) -> None:
        self._append("assistant", f"Error: {msg}")


def main() -> int:
    app = QtWidgets.QApplication([])
    w = BrowserHelperWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
