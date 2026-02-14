"""desktop_agent.chat_ui

Standalone chat UI for talking to a model with tool calling + web search.

Run:
    python -m desktop_agent.chat_ui
or:
    python scripts/chat_ui.py
"""

from __future__ import annotations

import html
import os
import re
import sys
import time
import logging
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json

from PySide6 import QtCore, QtGui, QtWidgets

from .chat_session import ChatConfig, ChatSession
from .chat_store import delete_chat, list_chats, load_chat, new_chat_id, prune_empty_chats, save_chat
from .config import DEFAULT_MODEL, SUPPORTED_MODELS, load_config
from .agent_hub import AgentHub, AgentHubError, load_agents_config, save_agents_config
from .tools import create_peer_agent_tool_spec, make_create_peer_agent_handler


_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")
_THINK_CAPTURE_RE = re.compile(r"(?is)<think>(.*?)</think>")
_HASH_CMD_CAPTURE_RE = re.compile(r"#(\S+)")
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_TOKEN_RE = re.compile(r"(?is)<think>(.*?)</think>|\[\[image:(.+?)\]\]|!\[[^\]]*\]\(([^)]+)\)")


def _extract_hash_commands(text: str) -> tuple[str, set[str]]:
    """Extract `#commands` (no space) and return (clean_text, commands_set)."""

    t = str(text or "")
    if "#" not in t:
        return t, set()
    cmds = {m.group(1).strip().casefold() for m in _HASH_CMD_CAPTURE_RE.finditer(t) if (m.group(1) or "").strip()}
    # Remove tokens like #skip; keep markdown headings "# Heading" (hash + space).
    cleaned = re.sub(r"#(\S+)", "", t)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), cmds


def _remove_think_blocks(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text or "")


def _render_markdown_with_think(text: str) -> str:
    """Render markdown, but wrap <think>...</think> content in green HTML spans.

    This hides the tags and colors only the content.
    """

    t = text or ""
    if "<think>" not in t.lower():
        return t
    out: list[str] = []
    last = 0
    for m in _THINK_CAPTURE_RE.finditer(t):
        out.append(t[last : m.start()])
        inner = m.group(1) or ""
        inner_html = _esc(inner).replace("\n", "<br/>")
        out.append(f"\n\n<span style=\"color:#31d158\">{inner_html}</span>\n\n")
        last = m.end()
    out.append(t[last:])
    return "".join(out)


def _render_html_with_think(text: str) -> str:
    """Render HTML safely, coloring <think>...</think> content in green.

    Also supports embedding local images via:
      - [[image:relative/path.png]]
      - Markdown images: ![](relative/path.png)
    """

    t = text or ""

    low = t.lower()
    if ("<think>" not in low) and ("[[image:" not in t) and ("![" not in t):
        return f"<div class='msg'>{_esc(t).replace('\\n', '<br/>')}</div>"

    root = _repo_root()

    def to_img_html(raw_path: str) -> str:
        p = str(raw_path or "").strip().strip("\"'")
        if not p:
            return ""
        rel = Path(p)
        if rel.is_absolute():
            return _esc(f"[image skipped: absolute path not allowed: {p}]")
        abs_path = (root / rel).resolve()
        try:
            if not abs_path.is_relative_to(root):
                return _esc(f"[image skipped: path escapes repo root: {p}]")
        except AttributeError:
            # best-effort fallback
            if str(abs_path).lower().find(str(root).lower()) != 0:
                return _esc(f"[image skipped: path escapes repo root: {p}]")
        if not abs_path.exists() or not abs_path.is_file():
            return _esc(f"[image missing: {p}]")
        try:
            url = abs_path.as_uri()
        except Exception:
            url = str(abs_path).replace("\\", "/")
        return (
            "<div style=\"margin-top:8px\">"
            f"<img src=\"{_esc(url)}\" style=\"max-width:100%; height:auto; border-radius:12px;\"/>"
            "</div>"
        )

    out: list[str] = []
    last = 0
    for m in _TOKEN_RE.finditer(t):
        out.append(_esc(t[last : m.start()]).replace("\n", "<br/>"))
        if m.group(1) is not None:
            inner = m.group(1) or ""
            inner_html = _esc(inner).replace("\n", "<br/>")
            out.append(f"<span style=\"color:#31d158\">{inner_html}</span>")
        else:
            img_path = (m.group(2) or m.group(3) or "").strip()
            out.append(to_img_html(img_path))
        last = m.end()
    out.append(_esc(t[last:]).replace("\n", "<br/>"))
    return "<div class='msg'>" + "".join(out) + "</div>"


def _strip_think(text: str) -> str:
    return _remove_think_blocks(text).strip()


def _repo_root() -> Path:
    # src/desktop_agent/chat_ui.py -> desktop_agent -> src -> repo root
    return Path(__file__).resolve().parents[2]

def _norm_name_ui(name: str) -> str:
    return " ".join(str(name or "").strip().split()).casefold()


_MENTION_TOKEN_RE = re.compile(r"(?i)(?:^|\\s)@([a-z0-9_\\-]+)")


def _mention_aliases(name: str) -> set[str]:
    """Compute mention-friendly aliases for an agent name."""

    base = _norm_name_ui(name)
    if not base:
        return set()
    out = {base}
    out.add(base.replace(" ", ""))
    out.add(base.replace(" ", "_"))
    return {x for x in out if x}


_LOG = logging.getLogger("desktop_agent.chat_ui")
_LOG_INITIALIZED = False


def _setup_run_logging() -> Path:
    """Configure terminal + file logging for debugging (clears file on startup)."""

    global _LOG_INITIALIZED  # noqa: PLW0603
    log_path = (_repo_root() / "chat_history" / "chat_ui.log").resolve()
    if _LOG_INITIALIZED:
        return log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False

    # Clear the log on every app start.
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    sh = logging.StreamHandler(stream=sys.stderr)
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    _LOG.handlers.clear()
    _LOG.addHandler(fh)
    _LOG.addHandler(sh)
    _LOG_INITIALIZED = True

    _LOG.info("=== Chat UI start ===")
    _LOG.info("log_path=%s", str(log_path))
    return log_path


def _load_qss() -> str:
    qss_path = _repo_root() / "ui" / "chat" / "style.qss"
    try:
        return qss_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _esc(s: str) -> str:
    return html.escape(s or "")


class _Signals(QtCore.QObject):
    append_user = QtCore.Signal(str)
    append_assistant = QtCore.Signal(str)
    append_tool = QtCore.Signal(str)
    set_busy = QtCore.Signal(bool)
    set_status = QtCore.Signal(str)
    set_tokens = QtCore.Signal(str)


@dataclass(frozen=True)
class ChatUIConfig:
    width: int = 1560
    height: int = 860


class AutoSizeTextBrowser(QtWidgets.QTextBrowser):
    """A QTextBrowser that grows vertically to fit its document.

    This avoids per-bubble scrollbars and prevents layout overlap issues that
    happen when fixed heights are used.
    """

    def __init__(self) -> None:
        super().__init__()
        sp = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        self.setOpenExternalLinks(True)
        self.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAnywhere)
        self.setViewportMargins(0, 0, 0, 0)
        try:
            self.document().setDocumentMargin(0.0)
        except Exception:
            pass

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, w: int) -> int:  # noqa: N802
        w = int(w)
        if w <= 0:
            return super().heightForWidth(w)

        # QTextDocument uses a "text width" for layout; approximate by subtracting
        # a small padding so wrapping matches the viewport width.
        doc = self.document()
        doc.setTextWidth(max(50.0, float(w - 2)))
        try:
            h = float(doc.documentLayout().documentSize().height())
        except Exception:
            h = float(doc.size().height())
        return max(24, int(h + 2))

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        # Provide a tight size hint so bubbles don't become overly tall.
        vw = int(self.viewport().width()) if self.viewport() is not None else 0
        w = vw if vw > 0 else 320
        return QtCore.QSize(w, self.heightForWidth(w))

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(120, 24)


class Bubble(QtWidgets.QFrame):
    def __init__(self, *, text: str, kind: str) -> None:
        super().__init__()
        self.setObjectName(f"bubble_{kind}")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._kind = kind

        self._label = AutoSizeTextBrowser()
        self._label.setObjectName("bubble_text")
        self._label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            | QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self._label.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.set_text(text)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.addWidget(self._label)

    def set_text(self, text: str) -> None:
        t = text or ""
        has_img = ("[[image:" in t) or ("![" in t and "](" in t)
        # Prefer Qt's Markdown renderer if available so **bold** etc. show nicely,
        # but use HTML when embedding local images (Qt Markdown image support is inconsistent).
        if (not has_img) and hasattr(self._label, "setMarkdown"):
            try:
                self._label.setMarkdown(_render_markdown_with_think(t))
                self._label.updateGeometry()
                self.updateGeometry()
                return
            except Exception:
                pass
        self._label.setHtml(_render_html_with_think(t))
        self._label.updateGeometry()
        self.updateGeometry()
        QtCore.QTimer.singleShot(0, self._label.updateGeometry)

    def set_streaming_text(self, text: str) -> None:
        # Streaming path: avoid re-parsing markdown on every tiny delta.
        t = text or ""
        self._label.setHtml(_render_html_with_think(t))
        self._label.updateGeometry()
        self.updateGeometry()


class ChatWindow(QtWidgets.QMainWindow):
    def __init__(self, *, cfg: Optional[ChatUIConfig] = None) -> None:
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        super().__init__()

        self._log_path = _setup_run_logging()
        self._cfg = cfg or ChatUIConfig()
        self._signals = _Signals()
        self._signals.append_user.connect(lambda t: self._append_message(t, kind="user"))
        self._signals.append_assistant.connect(lambda t: self._append_message(t, kind="assistant"))
        self._signals.append_tool.connect(lambda t: self._append_message(t, kind="tool"))
        self._signals.set_busy.connect(self._set_busy)
        self._signals.set_status.connect(self._set_status)
        self._signals.set_tokens.connect(self._set_tokens)

        self._session = self._make_session()
        self._show_tool_events = True
        self._hide_think = False
        self._workers: set[_Worker] = set()
        self._bubbles: list[Bubble] = []
        self._title_wrap: Optional[QtWidgets.QWidget] = None
        self._store_root = _repo_root()
        self._active_chat_id: str = self._session.chat_id
        self._switching_chat: bool = False
        self._active_submodel_id: str | None = None
        self._active_agent_id: str = str(self._session.chat_id)
        self._hub = AgentHub(base_config=self._session.cfg, make_session=self._instantiate_session, repo_root=self._store_root)
        self._hub.set_main(agent_id=str(self._session.chat_id), name="Main", session=self._session)
        # User-gated arming window for the main agent to create same-level agents.
        self._create_peer_armed_until: float = 0.0
        self._register_create_peer_agent_tool()
        self._event_ring: deque[str] = deque(maxlen=80)
        self._agent_busy: dict[str, bool] = {}

        self._build_ui()
        self._apply_style()

        self.resize(self._cfg.width, self._cfg.height)
        self.setMinimumSize(420, 640)
        self.setWindowTitle("Desktop Agent • Chat")

        self._append_message(
            "Tip: You can ask it to use web search, read files, or even create+register new tools during the chat.",
            kind="tool",
        )
        try:
            n = prune_empty_chats(self._store_root, keep_chat_id=None)
            if n:
                _LOG.info("pruned_empty_chats count=%d", int(n))
        except Exception:
            pass
        self._refresh_chat_list(select_chat_id=self._active_chat_id)
        self._refresh_submodel_list()
        self._set_tokens(self._session.usage_ratio_text())
        self._start_mention_timer()

        _LOG.info("ui_ready chat_id=%s model=%s", str(self._active_chat_id), str(self._session.cfg.model))

        self._peer_queue: list[str] = []
        self._last_user_text: str = ""
        self._last_main_text: str = ""
        self._skip_mode: bool = False
        self._last_speaker_id: str | None = None
        self._last_speaker_name: str = ""
        self._last_speaker_text: str = ""
        self._next_override_agent_id: str | None = None
        self._resume_next_agent_id: str | None = None
        self._resume_next_peer_id: str | None = None
        self._mention_popup: QtWidgets.QFrame | None = None
        self._mention_list: QtWidgets.QListWidget | None = None
        self._mention_all_names: list[str] = []
        self._mention_timer: QtCore.QTimer | None = None

    def exec(self) -> int:
        self.show()
        return int(self._app.exec())

    # ---- session ----

    def _instantiate_session(self, cfg: ChatConfig) -> ChatSession:
        app_cfg = load_config()
        return ChatSession(api_key=app_cfg.openai_api_key, config=cfg)

    def _make_session(self) -> ChatSession:
        app_cfg = load_config()
        # Chat uses the same env + model defaults as the main app.
        model = os.environ.get("OPENAI_MODEL", app_cfg.openai_model or DEFAULT_MODEL)
        raw_vs = (os.environ.get("OPENAI_VECTOR_STORE_IDS") or os.environ.get("OPENAI_VECTOR_STORE_ID") or "").strip()
        vs_ids = [x.strip() for x in raw_vs.split(",") if x.strip()]
        ccfg = ChatConfig(
            model=model,
            enable_web_search=True,
            web_search_context_size="medium",
            enable_file_search=bool(vs_ids),
            file_search_vector_store_ids=list(vs_ids),
            file_search_max_num_results=None,
            include_file_search_results=False,
            tool_base_dir=_repo_root(),
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=1024,
            max_tool_calls=8,
            allow_model_set_system_prompt=True,
            allow_model_create_tools=True,
            allow_model_propose_tools=True,
            allow_read_file=True,
            hide_think=False,
        )
        s = ChatSession(api_key=app_cfg.openai_api_key, config=ccfg)
        # Default system prompt: a chatty but tool-aware assistant.
        s.set_system_prompt(
            "You are a helpful assistant.\n"
            "You may use tools when beneficial (including web_search).\n"
            "When calling tools, be deliberate: call the minimum necessary tools.\n"
            "If you create a new tool, include a couple of self-tests.\n"
            "If the user asks you to create/generate a new agent, first ask for: name, model, system prompt, and optional memory file. "
            "Only then call create_peer_agent.\n"
        )
        return s

    def _register_create_peer_agent_tool(self) -> None:
        """Register a UI-only tool that lets Main create same-level agents (friends).

        This is intentionally gated: the user must explicitly request it in chat
        (we arm it for a short window after detecting a trigger phrase).
        """

        def is_armed() -> bool:
            return time.monotonic() <= float(self._create_peer_armed_until or 0.0)

        def disarm() -> None:
            self._create_peer_armed_until = 0.0

        def create_peer(name: str, model: str, system_prompt: str, memory_path: str | None) -> dict[str, Any]:
            a = self._hub.create_peer(name=name, model=model, system_prompt=system_prompt, memory_path=memory_path)
            return {"agent_id": a.agent_id, "name": a.name, "model": a.model, "memory_path": a.memory_path}

        try:
            self._session.registry.add(
                tool_spec=create_peer_agent_tool_spec(),
                handler=make_create_peer_agent_handler(is_armed=is_armed, disarm=disarm, create_peer=create_peer),
            )
        except Exception:
            # Best-effort: avoid crashing the UI if the tool already exists.
            pass

    # ---- UI ----

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        # Header
        header = QtWidgets.QFrame()
        header.setObjectName("header")
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(14, 12, 14, 12)
        hl.setSpacing(10)

        title = QtWidgets.QLabel("Chat")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel("Tools + Web Search")
        subtitle.setObjectName("subtitle")
        title_wrap = QtWidgets.QWidget()
        title_wrap.setObjectName("title_wrap")
        title_wrap.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self._title_wrap = title_wrap
        title_box = QtWidgets.QVBoxLayout(title_wrap)
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(2)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hl.addWidget(title_wrap, 0)
        hl.addStretch(1)

        # Right-side controls: keep these as a fixed-size cluster so they never overlap.
        right = QtWidgets.QWidget()
        right.setObjectName("header_right")
        right.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        hr = QtWidgets.QHBoxLayout(right)
        hr.setContentsMargins(0, 0, 0, 0)
        hr.setSpacing(8)

        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.setObjectName("model_combo")
        self._model_combo.addItems(list(SUPPORTED_MODELS))
        self._model_combo.setEditable(False)
        # Keep width bounded; elide long model IDs.
        self._model_combo.setMinimumContentsLength(18)
        self._model_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        # Allow shrinking in narrow windows; otherwise Qt may lay out neighbors
        # as-if the combo was smaller, but the widget will still enforce its
        # minimum size and overlap them.
        self._model_combo.setMinimumWidth(0)
        self._model_combo.setMaximumWidth(240)
        # Try to set current model if present.
        idx = self._model_combo.findText(self._session.cfg.model)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        hr.addWidget(self._model_combo)

        self._btn_system = QtWidgets.QPushButton("Sys")
        self._btn_system.setObjectName("btn_ghost")
        self._btn_system.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self._btn_system.setToolTip("System prompt")
        self._btn_system.setMaximumWidth(64)
        self._btn_system.clicked.connect(self._open_system_dialog)
        hr.addWidget(self._btn_system)

        self._btn_controls = QtWidgets.QPushButton("Ctrl")
        self._btn_controls.setObjectName("btn_ghost")
        self._btn_controls.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self._btn_controls.setToolTip("Controls")
        self._btn_controls.setMaximumWidth(72)
        self._btn_controls.clicked.connect(self._open_controls_dialog)
        hr.addWidget(self._btn_controls)

        # Let the model picker take whatever space remains.
        hr.setStretch(0, 1)
        hr.setStretch(1, 0)
        hr.setStretch(2, 0)

        hl.addWidget(right, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        outer.addWidget(header)

        # Main area: sidebar + chat
        main = QtWidgets.QHBoxLayout()
        main.setSpacing(12)
        outer.addLayout(main, 1)

        # Sidebar (chat history)
        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(210)
        sidebar.setMaximumWidth(300)
        sbl = QtWidgets.QVBoxLayout(sidebar)
        sbl.setContentsMargins(12, 12, 12, 12)
        sbl.setSpacing(10)

        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel("Chats")
        lbl.setObjectName("sidebar_title")
        row.addWidget(lbl)
        row.addStretch(1)
        self._btn_new_chat = QtWidgets.QPushButton("New")
        self._btn_new_chat.setObjectName("btn_ghost")
        self._btn_new_chat.clicked.connect(self._new_chat)
        row.addWidget(self._btn_new_chat)
        sbl.addLayout(row)

        self._chat_list = QtWidgets.QListWidget()
        self._chat_list.setObjectName("chat_list")
        self._chat_list.itemSelectionChanged.connect(self._on_chat_selected)
        sbl.addWidget(self._chat_list, 1)

        side_btns = QtWidgets.QHBoxLayout()
        self._btn_rename = QtWidgets.QPushButton("Rename")
        self._btn_rename.setObjectName("btn_ghost")
        self._btn_rename.clicked.connect(self._rename_chat)
        self._btn_delete = QtWidgets.QPushButton("Delete")
        self._btn_delete.setObjectName("btn_ghost")
        self._btn_delete.clicked.connect(self._delete_chat)
        side_btns.addWidget(self._btn_rename)
        side_btns.addWidget(self._btn_delete)
        sbl.addLayout(side_btns)

        main.addWidget(sidebar, 0)

        # Chat area
        chat_col = QtWidgets.QVBoxLayout()
        chat_col.setSpacing(12)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setObjectName("scroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._chat = QtWidgets.QWidget()
        self._chat.setObjectName("chat")
        self._chat_layout = QtWidgets.QVBoxLayout(self._chat)
        self._chat_layout.setContentsMargins(6, 10, 6, 10)
        self._chat_layout.setSpacing(10)
        self._chat_layout.addStretch(1)

        self._scroll.setWidget(self._chat)
        chat_col.addWidget(self._scroll, 1)

        # Tokens indicator (bottom-right of chat area)
        token_row = QtWidgets.QHBoxLayout()
        token_row.setContentsMargins(0, 0, 0, 0)
        token_row.addStretch(1)
        self._tokens = QtWidgets.QLabel("")
        self._tokens.setObjectName("tokens")
        self._tokens.setToolTip("Approx prompt tokens used (includes system prompt) / context window")
        token_row.addWidget(self._tokens)
        chat_col.addLayout(token_row)

        # Composer
        composer = QtWidgets.QFrame()
        composer.setObjectName("composer")
        cl = QtWidgets.QHBoxLayout(composer)
        cl.setContentsMargins(12, 12, 12, 12)
        cl.setSpacing(10)

        self._input = QtWidgets.QTextEdit()
        self._input.setObjectName("input")
        self._input.setPlaceholderText("Message… (Shift+Enter for newline)")
        self._input.setAcceptRichText(False)
        self._input.setFixedHeight(90)
        self._input.installEventFilter(self)
        cl.addWidget(self._input, 1)
        self._install_mention_popup()

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(8)

        self._btn_send = QtWidgets.QPushButton("Send")
        self._btn_send.setObjectName("btn_primary")
        self._btn_send.clicked.connect(self._on_send)
        send_row = QtWidgets.QHBoxLayout()
        send_row.setSpacing(8)
        send_row.addWidget(self._btn_send, 1)

        self._btn_skip = QtWidgets.QPushButton("Skip")
        self._btn_skip.setCheckable(True)
        self._btn_skip.setObjectName("btn_toggle")
        self._btn_skip.setToolTip("When enabled, agents continue round-robin without waiting for user replies.")
        self._btn_skip.toggled.connect(self._on_skip_toggled)
        send_row.addWidget(self._btn_skip, 0)

        right.addLayout(send_row)

        self._chk_tools = QtWidgets.QCheckBox("Tool events")
        self._chk_tools.setChecked(True)
        self._chk_tools.stateChanged.connect(lambda _: self._set_show_tool_events(self._chk_tools.isChecked()))
        right.addWidget(self._chk_tools)

        self._chk_think = QtWidgets.QCheckBox("Hide <think>")
        self._chk_think.setChecked(False)
        self._chk_think.stateChanged.connect(lambda _: self._set_hide_think(self._chk_think.isChecked()))
        right.addWidget(self._chk_think)

        self._status = QtWidgets.QLabel("")
        self._status.setObjectName("status")
        right.addWidget(self._status)
        right.addStretch(1)

        cl.addLayout(right)
        chat_col.addWidget(composer)
        main.addLayout(chat_col, 1)

        # Right sidebar (agents tree: main + sub-agents/submodels)
        subbar = QtWidgets.QFrame()
        subbar.setObjectName("subbar")
        subbar.setMinimumWidth(240)
        subbar.setMaximumWidth(360)
        rbl = QtWidgets.QVBoxLayout(subbar)
        rbl.setContentsMargins(12, 12, 12, 12)
        rbl.setSpacing(10)

        lbl2 = QtWidgets.QLabel("Agents")
        lbl2.setObjectName("sidebar_title")
        rbl.addWidget(lbl2)

        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(8)

        self._btn_new_agent = QtWidgets.QPushButton("New")
        self._btn_new_agent.setObjectName("btn_ghost")
        self._btn_new_agent.clicked.connect(self._new_agent_dialog)
        self._btn_new_agent.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        row2.addWidget(self._btn_new_agent)

        self._btn_load_agents = QtWidgets.QPushButton("Load")
        self._btn_load_agents.setObjectName("btn_ghost")
        self._btn_load_agents.clicked.connect(self._load_agents_config_ui)
        self._btn_load_agents.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        row2.addWidget(self._btn_load_agents)

        self._btn_save_agents = QtWidgets.QPushButton("Save")
        self._btn_save_agents.setObjectName("btn_ghost")
        self._btn_save_agents.clicked.connect(self._save_agents_config_ui)
        self._btn_save_agents.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        row2.addWidget(self._btn_save_agents)

        self._btn_edit_agent = QtWidgets.QPushButton("Edit")
        self._btn_edit_agent.setObjectName("btn_ghost")
        self._btn_edit_agent.clicked.connect(self._edit_selected_agent)
        self._btn_edit_agent.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        row2.addWidget(self._btn_edit_agent)

        self._btn_delete_agent = QtWidgets.QPushButton("Delete")
        self._btn_delete_agent.setObjectName("btn_ghost")
        self._btn_delete_agent.clicked.connect(self._delete_selected_agent)
        self._btn_delete_agent.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        row2.addWidget(self._btn_delete_agent)

        rbl.addLayout(row2)

        self._agents_tree = QtWidgets.QTreeWidget()
        self._agents_tree.setObjectName("agents_tree")
        self._agents_tree.setHeaderHidden(True)
        # We'll render our own +/- icons in a dedicated column for consistent UX across platforms.
        self._agents_tree.setRootIsDecorated(False)
        self._agents_tree.setAnimated(True)
        self._agents_tree.setIndentation(18)
        self._agents_tree.setColumnCount(2)
        try:
            self._agents_tree.header().setStretchLastSection(True)
            self._agents_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self._agents_tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        except Exception:
            pass
        self._agents_tree.setColumnWidth(0, 22)
        self._agents_tree.setIconSize(QtCore.QSize(14, 14))
        self._agent_icon_closed: QtGui.QIcon | None = None
        self._agent_icon_open: QtGui.QIcon | None = None
        try:
            icons_dir = _repo_root() / "ui" / "chat" / "icons"
            self._agent_icon_closed = QtGui.QIcon(str((icons_dir / "branch_closed.svg").resolve()))
            self._agent_icon_open = QtGui.QIcon(str((icons_dir / "branch_open.svg").resolve()))
        except Exception:
            self._agent_icon_closed = None
            self._agent_icon_open = None
        self._agents_tree.itemDoubleClicked.connect(lambda *_: self._open_selected_submodel())
        self._agents_tree.itemExpanded.connect(self._update_agents_tree_labels)
        self._agents_tree.itemCollapsed.connect(self._update_agents_tree_labels)
        self._agents_tree.itemClicked.connect(self._on_agents_tree_clicked)
        self._agents_tree.itemSelectionChanged.connect(self._on_agent_selection_changed)
        rbl.addWidget(self._agents_tree, 1)

        sub_btns = QtWidgets.QHBoxLayout()
        self._btn_open_sub = QtWidgets.QPushButton("Open")
        self._btn_open_sub.setObjectName("btn_ghost")
        self._btn_open_sub.clicked.connect(self._open_selected_submodel)
        self._btn_close_sub = QtWidgets.QPushButton("Close")
        self._btn_close_sub.setObjectName("btn_ghost")
        self._btn_close_sub.clicked.connect(self._close_selected_submodel)
        sub_btns.addWidget(self._btn_open_sub)
        sub_btns.addWidget(self._btn_close_sub)
        rbl.addLayout(sub_btns)

        main.addWidget(subbar, 0)

    def _apply_style(self) -> None:
        qss = _load_qss()
        if qss:
            self._app.setStyleSheet(qss)
        font = QtGui.QFont()
        font.setFamilies(["SF Pro Display", "Segoe UI Variable", "Segoe UI", "Inter", "Arial"])
        font.setPointSize(10)
        self._app.setFont(font)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if obj is self._input and event.type() in (
            QtCore.QEvent.Type.FocusOut,
            QtCore.QEvent.Type.WindowDeactivate,
        ):
            self._hide_mention_popup()
        if obj is self._input and event.type() == QtCore.QEvent.Type.KeyPress:
            e = event  # type: ignore[assignment]
            if isinstance(e, QtGui.QKeyEvent):
                if self._mention_popup is not None and self._mention_popup.isVisible():
                    if e.key() == QtCore.Qt.Key.Key_Escape:
                        self._hide_mention_popup()
                        return True
                    if e.key() in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down):
                        self._move_mention_selection(-1 if e.key() == QtCore.Qt.Key.Key_Up else 1)
                        return True
                    if e.key() in (
                        QtCore.Qt.Key.Key_Return,
                        QtCore.Qt.Key.Key_Enter,
                        QtCore.Qt.Key.Key_Tab,
                        QtCore.Qt.Key.Key_Backtab,
                    ):
                        sel = self._selected_mention_name()
                        if sel:
                            self._insert_mention_completion(sel)
                        self._hide_mention_popup()
                        return True
                if e.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    if e.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                        return False
                    self._on_send()
                    return True
                # Update @-mentions dropdown after edits.
                if e.key() in (
                    QtCore.Qt.Key.Key_At,
                    QtCore.Qt.Key.Key_Backspace,
                    QtCore.Qt.Key.Key_Delete,
                    QtCore.Qt.Key.Key_Space,
                ) or (e.text() and not e.text().isspace()):
                    QtCore.QTimer.singleShot(0, self._update_mention_popup)
        if obj is self._input and event.type() in (QtCore.QEvent.Type.KeyRelease, QtCore.QEvent.Type.InputMethod):
            QtCore.QTimer.singleShot(0, self._update_mention_popup)
        return super().eventFilter(obj, event)

    # ---- actions ----

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def _set_tokens(self, text: str) -> None:
        self._tokens.setText(text)

    def _set_busy(self, busy: bool) -> None:
        # Keep Send/Input enabled so the user can interrupt by sending.
        self._model_combo.setEnabled(not busy)
        self._btn_system.setEnabled(not busy)
        self._set_status("Thinking…" if busy else "")
        _LOG.info("busy=%s chat_id=%s model=%s", bool(busy), str(self._active_chat_id), str(self._session.cfg.model))

    def _on_skip_toggled(self, on: bool) -> None:
        self._skip_mode = bool(on)
        # If we just enabled skip and nothing is running, continue the round-robin.
        if self._skip_mode and not self._workers:
            QtCore.QTimer.singleShot(0, self._auto_continue_next)

    def _interrupt_all_workers(self) -> None:
        # Best-effort: request interruption so streaming loops can stop.
        for w in list(self._workers):
            try:
                w.requestInterruption()
            except Exception:
                pass
            try:
                w.quit()
            except Exception:
                pass
        for w in list(self._workers):
            try:
                w.wait(300)
            except Exception:
                pass
        self._workers.clear()
        self._peer_queue = []
        self._set_status("")

    def _maybe_set_next_override_from_text(self, text: str) -> None:
        """If `text` contains @AgentName, schedule that agent to speak next."""

        raw = str(text or "")
        if "@" not in raw:
            return

        agents = self._hub.list_agents()
        if not agents:
            return

        alias_to_id: dict[str, str] = {}
        for a in agents:
            for al in _mention_aliases(a.name):
                alias_to_id[al] = a.agent_id

        # Token mentions: @friend
        for m in _MENTION_TOKEN_RE.finditer(raw):
            tok = _norm_name_ui(m.group(1) or "")
            if not tok:
                continue
            aid = alias_to_id.get(tok)
            if isinstance(aid, str) and aid:
                self._next_override_agent_id = aid
                return

        # Full-name mentions: "@Tel Aviv weather"
        low = raw.casefold()
        for a in agents:
            nm = str(a.name or "").strip()
            if not nm:
                continue
            if f"@{nm.casefold()}" in low:
                self._next_override_agent_id = a.agent_id
                return

    def _set_show_tool_events(self, enabled: bool) -> None:
        self._show_tool_events = bool(enabled)

    def _set_hide_think(self, enabled: bool) -> None:
        self._hide_think = bool(enabled)
        self._session.cfg.hide_think = self._hide_think

    def _on_model_changed(self, model: str) -> None:
        self._session.cfg.model = str(model)
        # Keep hub main model in sync for exports.
        try:
            self._hub.get(self._hub.main_agent_id()).model = str(model)
        except Exception:
            pass

    def _append_message(self, text: str, *, kind: str) -> Bubble:
        try:
            preview = (text or "").replace("\r", "").strip()
            if len(preview) > 500:
                preview = preview[:500] + " …"
            self._event_ring.append(f"{kind}: {preview}")
            _LOG.info("chat_event chat_id=%s kind=%s text=%s", str(self._active_chat_id), str(kind), preview)
        except Exception:
            pass
        bubble = Bubble(text=text, kind=kind)
        self._bubbles.append(bubble)
        container = QtWidgets.QWidget()
        container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        if kind == "user":
            row.addStretch(1)
            row.addWidget(bubble, 0)
        else:
            row.addWidget(bubble, 0)
            row.addStretch(1)

        # Insert before the stretch at the end.
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, container)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
        QtCore.QTimer.singleShot(0, self._update_bubble_widths)
        QtCore.QTimer.singleShot(0, self._refresh_submodel_list)
        return bubble

    # ---- @mentions autocomplete ----

    def _start_mention_timer(self) -> None:
        t = QtCore.QTimer(self)
        t.setInterval(150)
        t.timeout.connect(self._mention_timer_tick)
        t.start()
        self._mention_timer = t

    def _mention_timer_tick(self) -> None:
        try:
            if self._input.hasFocus() or (self._mention_popup is not None and self._mention_popup.isVisible()):
                self._update_mention_popup()
        except Exception:
            pass

    def _install_mention_popup(self) -> None:
        # QCompleter can be unreliable with QTextEdit on some platforms; implement a
        # small custom popup instead.
        # NOTE: make it a top-level tool window so it doesn't steal focus from the editor
        # (Popup windows can be flaky across platforms and sometimes instantly dismiss).
        pop = QtWidgets.QFrame(None)
        pop.setObjectName("mention_popup_frame")
        pop.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        pop.setWindowFlags(
            QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        pop.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        lay = QtWidgets.QVBoxLayout(pop)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lw = QtWidgets.QListWidget()
        lw.setObjectName("mention_popup")
        lw.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        lw.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lw.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        lw.itemClicked.connect(lambda it: (self._insert_mention_completion(it.text()), self._hide_mention_popup()))
        lay.addWidget(lw)
        self._mention_popup = pop
        self._mention_list = lw
        self._input.textChanged.connect(lambda: QtCore.QTimer.singleShot(0, self._update_mention_popup))
        self._input.cursorPositionChanged.connect(lambda: QtCore.QTimer.singleShot(0, self._update_mention_popup))
        self._update_agent_mentions_model()

    def _update_agent_mentions_model(self) -> None:
        try:
            names = [str(a.name) for a in self._hub.list_agents() if str(a.name).strip()]
        except Exception:
            names = []
        seen: set[str] = set()
        out: list[str] = []
        for n in names:
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
        self._mention_all_names = out

    def _mention_context(self) -> tuple[int, str] | None:
        """Return (at_index, prefix_after_at) if cursor is in an @mention prefix."""

        try:
            cur = self._input.textCursor()
            pos = int(cur.position())
        except Exception:
            return None
        text = self._input.toPlainText()
        if pos < 0 or pos > len(text):
            return None
        before = text[:pos]
        at = before.rfind("@")
        if at < 0:
            return None
        prefix = before[at + 1 :]
        # Only show completion while still typing the mention token (no whitespace/newline yet).
        if any(ch.isspace() for ch in prefix):
            return None
        return (at, prefix)

    def _update_mention_popup(self) -> None:
        if self._mention_popup is None or self._mention_list is None:
            return
        ctx = self._mention_context()
        if ctx is None:
            self._hide_mention_popup()
            return
        _, prefix = ctx
        try:
            _LOG.info("mention_popup_update prefix=%r", str(prefix))
        except Exception:
            pass
        self._update_agent_mentions_model()
        pref = _norm_name_ui(prefix)
        items: list[str] = []
        if not pref:
            items = list(self._mention_all_names)
        else:
            for n in self._mention_all_names:
                if pref in _norm_name_ui(n):
                    items.append(n)

        if not items:
            self._hide_mention_popup()
            return

        self._mention_list.clear()
        self._mention_list.addItems(items)
        self._mention_list.setCurrentRow(0)

        self._show_mention_popup()

    def _show_mention_popup(self) -> None:
        if self._mention_popup is None or self._mention_list is None:
            return
        r = self._input.cursorRect()
        try:
            pt = self._input.viewport().mapToGlobal(r.bottomLeft())
        except Exception:
            pt = self._input.mapToGlobal(r.bottomLeft())
        width = max(260, min(420, int(self._input.width() * 0.75)))
        try:
            row_h = int(self._mention_list.sizeHintForRow(0))
        except Exception:
            row_h = 0
        if row_h <= 0:
            row_h = 22
        height = min(260, max(80, int(row_h * min(10, self._mention_list.count()) + 16)))
        self._mention_popup.setGeometry(pt.x(), pt.y() + 6, width, height)
        self._mention_popup.show()
        self._mention_popup.raise_()
        try:
            _LOG.info("mention_popup_show items=%d prefix_ok", int(self._mention_list.count()))
        except Exception:
            pass

    def _hide_mention_popup(self) -> None:
        try:
            if self._mention_popup is not None:
                self._mention_popup.hide()
                _LOG.info("mention_popup_hide")
        except Exception:
            pass

    def _selected_mention_name(self) -> str:
        if self._mention_list is None:
            return ""
        it = self._mention_list.currentItem()
        return str(it.text() or "") if it is not None else ""

    def _move_mention_selection(self, delta: int) -> None:
        if self._mention_list is None:
            return
        n = int(self._mention_list.count())
        if n <= 0:
            return
        cur = int(self._mention_list.currentRow())
        nxt = max(0, min(n - 1, cur + int(delta)))
        self._mention_list.setCurrentRow(nxt)

    def _insert_mention_completion(self, completion: str) -> None:
        ctx = self._mention_context()
        if ctx is None:
            return
        at, _ = ctx
        cur = self._input.textCursor()
        try:
            pos = int(cur.position())
        except Exception:
            return
        cur.setPosition(at + 1)
        cur.setPosition(pos, QtGui.QTextCursor.MoveMode.KeepAnchor)
        cur.insertText(str(completion) + " ")
        self._input.setTextCursor(cur)

    def _append_agent_message(self, *, agent_name: str, text: str) -> None:
        self._append_message(f"**{agent_name}:**\n\n{text}", kind="assistant")

    def _find_agent_by_name(self, name: str) -> str | None:
        nn = str(name or "").strip()
        if not nn:
            return None
        for a in self._hub.list_agents():
            if _norm_name_ui(a.name) == _norm_name_ui(nn):
                return a.agent_id
        return None

    def _on_agent_selection_changed(self) -> None:
        it = self._agents_tree.currentItem()
        if it is None:
            return
        data = it.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return
        kind = data.get("kind")
        if kind == "agent":
            aid = data.get("id")
            if isinstance(aid, str) and aid:
                self._active_agent_id = aid
        elif kind == "submodel":
            # Keep active agent as the parent; selection is a submodel.
            pid = data.get("agent_id")
            if isinstance(pid, str) and pid:
                self._active_agent_id = pid

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._title_wrap is not None:
            # Keep the title region from starving the right-side controls in narrow windows.
            max_w = max(140, min(260, int(self.width() * 0.38)))
            self._title_wrap.setMaximumWidth(max_w)
        self._update_bubble_widths()

    def _update_bubble_widths(self) -> None:
        try:
            viewport_w = int(self._scroll.viewport().width())
        except Exception:
            viewport_w = int(self.width())
        max_w = max(280, int(viewport_w * 0.72))
        for b in self._bubbles[-200:]:
            b.setMaximumWidth(max_w)
            b.updateGeometry()

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _open_system_dialog(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("System Prompt")
        dlg.setObjectName("dialog")
        dlg.resize(720, 520)

        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        label = QtWidgets.QLabel("Edit the system prompt used for future turns (history preserved).")
        label.setObjectName("dialog_label")
        lay.addWidget(label)

        edit = QtWidgets.QTextEdit()
        edit.setObjectName("system_edit")
        edit.setPlainText(self._session.system_prompt)
        lay.addWidget(edit, 1)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_reset = QtWidgets.QPushButton("Reset chat")
        btn_reset.setObjectName("btn_ghost")
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setObjectName("btn_primary")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setObjectName("btn_ghost")
        row.addWidget(btn_reset)
        row.addWidget(btn_cancel)
        row.addWidget(btn_apply)
        lay.addLayout(row)

        def do_reset() -> None:
            self._session.reset(keep_system_prompt=True)
            self._append_message("Chat history cleared (system prompt preserved).", kind="tool")
            dlg.accept()

        btn_reset.clicked.connect(do_reset)
        btn_cancel.clicked.connect(dlg.reject)

        def do_apply() -> None:
            try:
                mem = self._hub.get(self._hub.main_agent_id()).memory_path
            except Exception:
                mem = None
            self._hub.update_main(model=str(self._session.cfg.model), system_prompt=edit.toPlainText(), memory_path=mem)
            self._append_message("System prompt updated.", kind="tool")
            dlg.accept()

        btn_apply.clicked.connect(do_apply)

        dlg.exec()

    def _open_controls_dialog(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Controls")
        dlg.setObjectName("dialog")
        dlg.resize(760, 640)

        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("tabs")
        lay.addWidget(tabs, 1)

        # --- System tab ---
        sys_tab = QtWidgets.QWidget()
        sys_l = QtWidgets.QVBoxLayout(sys_tab)
        sys_l.setContentsMargins(12, 12, 12, 12)
        sys_l.setSpacing(10)

        lbl = QtWidgets.QLabel("System prompt (applies to future turns; conversation preserved).")
        lbl.setObjectName("dialog_label")
        sys_l.addWidget(lbl)

        sys_edit = QtWidgets.QTextEdit()
        sys_edit.setObjectName("system_edit")
        sys_edit.setPlainText(self._session.system_prompt)
        sys_l.addWidget(sys_edit, 1)

        sys_row = QtWidgets.QHBoxLayout()
        sys_row.addStretch(1)
        btn_clear = QtWidgets.QPushButton("Clear chat")
        btn_clear.setObjectName("btn_ghost")
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setObjectName("btn_primary")
        sys_row.addWidget(btn_clear)
        sys_row.addWidget(btn_apply)
        sys_l.addLayout(sys_row)

        def do_clear() -> None:
            self._session.reset(keep_system_prompt=True)
            self._append_message("Chat history cleared (system prompt preserved).", kind="tool")

        btn_clear.clicked.connect(do_clear)

        def do_apply() -> None:
            try:
                mem = self._hub.get(self._hub.main_agent_id()).memory_path
            except Exception:
                mem = None
            self._hub.update_main(model=str(self._session.cfg.model), system_prompt=sys_edit.toPlainText(), memory_path=mem)
            self._append_message("System prompt updated.", kind="tool")

        btn_apply.clicked.connect(do_apply)

        tabs.addTab(sys_tab, "System")

        # --- Generation tab ---
        gen_tab = QtWidgets.QWidget()
        gl = QtWidgets.QFormLayout(gen_tab)
        gl.setContentsMargins(12, 12, 12, 12)
        gl.setSpacing(10)

        temp = QtWidgets.QDoubleSpinBox()
        temp.setRange(0.0, 2.0)
        temp.setSingleStep(0.05)
        temp.setValue(float(self._session.cfg.temperature or 0.7))

        top_p = QtWidgets.QDoubleSpinBox()
        top_p.setRange(0.0, 1.0)
        top_p.setSingleStep(0.01)
        top_p.setValue(float(self._session.cfg.top_p or 0.95))

        mot = QtWidgets.QSpinBox()
        mot.setRange(1, 32_768)
        mot.setValue(int(self._session.cfg.max_output_tokens or 1024))

        mtc = QtWidgets.QSpinBox()
        mtc.setRange(1, 50)
        mtc.setValue(int(self._session.cfg.max_tool_calls or 8))

        ctxw = QtWidgets.QSpinBox()
        ctxw.setRange(1_024, 1_000_000)
        ctxw.setSingleStep(1024)
        ctxw.setValue(int(self._session.cfg.context_window_tokens))

        gl.addRow("Temperature", temp)
        gl.addRow("Top-p", top_p)
        gl.addRow("Context window (tok)", ctxw)
        gl.addRow("Max output tokens", mot)
        gl.addRow("Max tool calls", mtc)

        def apply_gen() -> None:
            self._session.cfg.temperature = float(temp.value())
            self._session.cfg.top_p = float(top_p.value())
            self._session.cfg.context_window_tokens = int(ctxw.value())
            self._session.cfg.max_output_tokens = int(mot.value())
            self._session.cfg.max_tool_calls = int(mtc.value())
            self._append_message("Generation settings updated.", kind="tool")
            self._set_tokens(self._session.usage_ratio_text())

        btn_apply_gen = QtWidgets.QPushButton("Apply generation settings")
        btn_apply_gen.setObjectName("btn_primary")
        btn_apply_gen.clicked.connect(apply_gen)
        gl.addRow(btn_apply_gen)

        tabs.addTab(gen_tab, "Generation")

        # --- Tools tab ---
        tools_tab = QtWidgets.QWidget()
        tl = QtWidgets.QFormLayout(tools_tab)
        tl.setContentsMargins(12, 12, 12, 12)
        tl.setSpacing(10)

        chk_web = QtWidgets.QCheckBox("Enable web search")
        chk_web.setChecked(bool(self._session.cfg.enable_web_search))

        ctx = QtWidgets.QComboBox()
        ctx.addItems(["low", "medium", "high"])
        i = ctx.findText(str(self._session.cfg.web_search_context_size))
        if i >= 0:
            ctx.setCurrentIndex(i)

        chk_read = QtWidgets.QCheckBox("Allow read_file")
        chk_read.setChecked(bool(self._session.cfg.allow_read_file))

        chk_file = QtWidgets.QCheckBox("Enable file search (vector stores)")
        chk_file.setChecked(bool(getattr(self._session.cfg, "enable_file_search", False)))

        vs_edit = QtWidgets.QLineEdit()
        vs_edit.setObjectName("vector_store_ids")
        vs_edit.setPlaceholderText("vs_... , vs_... (comma-separated)")
        try:
            vs_edit.setText(",".join(getattr(self._session.cfg, "file_search_vector_store_ids", []) or []))
        except Exception:
            vs_edit.setText("")

        fs_max = QtWidgets.QSpinBox()
        fs_max.setRange(0, 50)
        fs_max.setToolTip("0 = default")
        cur_mn = getattr(self._session.cfg, "file_search_max_num_results", None)
        fs_max.setValue(int(cur_mn) if cur_mn is not None else 0)

        chk_fs_results = QtWidgets.QCheckBox("Include file search results (uses more tokens)")
        chk_fs_results.setChecked(bool(getattr(self._session.cfg, "include_file_search_results", False)))

        chk_write = QtWidgets.QCheckBox("Allow write/append files (memory.md, chat_history/*)")
        chk_write.setChecked(bool(self._session.cfg.allow_write_files))

        chk_py = QtWidgets.QCheckBox("Allow python sandbox (numpy/pandas/matplotlib)")
        chk_py.setChecked(bool(getattr(self._session.cfg, "allow_python_sandbox", False)))

        py_timeout = QtWidgets.QDoubleSpinBox()
        py_timeout.setRange(0.5, 120.0)
        py_timeout.setSingleStep(0.5)
        py_timeout.setValue(float(getattr(self._session.cfg, "python_sandbox_timeout_s", 12.0)))

        chk_setsys = QtWidgets.QCheckBox("Allow model set_system_prompt")
        chk_setsys.setChecked(bool(self._session.cfg.allow_model_set_system_prompt))

        chk_propose = QtWidgets.QCheckBox("Allow model propose tools")
        chk_propose.setChecked(bool(self._session.cfg.allow_model_propose_tools))

        chk_create = QtWidgets.QCheckBox("Allow model create+register tools")
        chk_create.setChecked(bool(self._session.cfg.allow_model_create_tools))

        chk_submodels = QtWidgets.QCheckBox("Allow submodels (spawn/run helper instances)")
        chk_submodels.setChecked(bool(self._session.cfg.allow_submodels))

        max_sub = QtWidgets.QSpinBox()
        max_sub.setRange(0, 32)
        max_sub.setValue(int(self._session.cfg.max_submodels))

        max_depth = QtWidgets.QSpinBox()
        max_depth.setRange(0, 5)
        max_depth.setValue(int(self._session.cfg.max_submodel_depth))

        ping_s = QtWidgets.QDoubleSpinBox()
        ping_s.setRange(0.2, 10.0)
        ping_s.setSingleStep(0.2)
        ping_s.setValue(float(self._session.cfg.submodel_ping_s))

        tl.addRow(chk_web)
        tl.addRow("Search context size", ctx)
        tl.addRow(chk_read)
        tl.addRow(chk_file)
        tl.addRow("Vector store IDs", vs_edit)
        tl.addRow("File search max results", fs_max)
        tl.addRow(chk_fs_results)
        tl.addRow(chk_write)
        tl.addRow(chk_py)
        tl.addRow("Python sandbox timeout (s)", py_timeout)
        tl.addRow(chk_setsys)
        tl.addRow(chk_propose)
        tl.addRow(chk_create)
        tl.addRow(chk_submodels)
        tl.addRow("Max submodels", max_sub)
        tl.addRow("Max submodel depth", max_depth)
        tl.addRow("Ping interval (s)", ping_s)

        def apply_tools() -> None:
            self._session.cfg.enable_web_search = bool(chk_web.isChecked())
            self._session.cfg.web_search_context_size = str(ctx.currentText())
            self._session.cfg.allow_read_file = bool(chk_read.isChecked())
            self._session.cfg.enable_file_search = bool(chk_file.isChecked())
            raw = str(vs_edit.text() or "").strip()
            self._session.cfg.file_search_vector_store_ids = [x.strip() for x in raw.split(",") if x.strip()]
            mn = int(fs_max.value())
            self._session.cfg.file_search_max_num_results = mn if mn > 0 else None
            self._session.cfg.include_file_search_results = bool(chk_fs_results.isChecked())
            self._session.cfg.allow_write_files = bool(chk_write.isChecked())
            self._session.cfg.allow_python_sandbox = bool(chk_py.isChecked())
            self._session.cfg.python_sandbox_timeout_s = float(py_timeout.value())
            self._session.cfg.allow_model_set_system_prompt = bool(chk_setsys.isChecked())
            self._session.cfg.allow_model_propose_tools = bool(chk_propose.isChecked())
            self._session.cfg.allow_model_create_tools = bool(chk_create.isChecked())
            self._session.cfg.allow_submodels = bool(chk_submodels.isChecked())
            self._session.cfg.max_submodels = int(max_sub.value())
            self._session.cfg.max_submodel_depth = int(max_depth.value())
            self._session.cfg.submodel_ping_s = float(ping_s.value())
            self._append_message("Tool settings updated.", kind="tool")
            self._refresh_submodel_list()

        btn_apply_tools = QtWidgets.QPushButton("Apply tool settings")
        btn_apply_tools.setObjectName("btn_primary")
        btn_apply_tools.clicked.connect(apply_tools)
        tl.addRow(btn_apply_tools)

        tabs.addTab(tools_tab, "Tools")

        # --- Display tab ---
        disp_tab = QtWidgets.QWidget()
        dl = QtWidgets.QFormLayout(disp_tab)
        dl.setContentsMargins(12, 12, 12, 12)
        dl.setSpacing(10)

        chk_tool_events = QtWidgets.QCheckBox("Show tool events")
        chk_tool_events.setChecked(bool(self._show_tool_events))
        chk_hide_think = QtWidgets.QCheckBox("Hide <think> blocks")
        chk_hide_think.setChecked(bool(self._hide_think))
        dl.addRow(chk_tool_events)
        dl.addRow(chk_hide_think)

        def apply_disp() -> None:
            self._set_show_tool_events(chk_tool_events.isChecked())
            self._set_hide_think(chk_hide_think.isChecked())
            self._append_message("Display settings updated.", kind="tool")

        btn_apply_disp = QtWidgets.QPushButton("Apply display settings")
        btn_apply_disp.setObjectName("btn_primary")
        btn_apply_disp.clicked.connect(apply_disp)
        dl.addRow(btn_apply_disp)

        tabs.addTab(disp_tab, "Display")

        dlg.exec()

    def _on_send(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        if self._workers:
            self._interrupt_all_workers()
        _LOG.info("user_send chat_id=%s model=%s chars=%d", str(self._active_chat_id), str(self._session.cfg.model), len(text))
        self._input.clear()

        self._last_user_text = text
        self._last_speaker_id = "user"
        self._last_speaker_name = "User"
        self._last_speaker_text = text
        self._maybe_set_next_override_from_text(text)
        self._append_message(text, kind="user")
        assistant_bubble: Bubble | None = None

        # Arm the create-agent tool if the user explicitly asked to create/generate an agent.
        if re.search(r"(?i)\b(generate|create|make|add)\s+(a\s+)?(new\s+)?agent\b", text):
            self._create_peer_armed_until = time.monotonic() + 10 * 60.0  # 10 minutes

        # Optional direct message: "@AgentName message"
        direct_target: str | None = None
        direct_text: str | None = None
        if text.startswith("@") and " " in text:
            maybe, rest = text[1:].split(" ", 1)
            aid = self._find_agent_by_name(maybe)
            if aid is not None:
                direct_target = aid
                direct_text = rest.strip()

        if direct_target is not None and direct_text:
            # Send only to that agent.
            try:
                agent = self._hub.get(direct_target)
            except Exception:
                agent = self._hub.get(self._hub.main_agent_id())
            self._send_via_worker(session=agent.session, agent_name=agent.name, agent_id=agent.agent_id, text=direct_text)
            return

        # Default: main agent responds, then peer agents respond sequentially.
        self._peer_queue = [a.agent_id for a in self._hub.list_agents() if a.agent_id != self._hub.main_agent_id()]

        def _after_main_final(t: str) -> None:
            self._last_main_text = str(t or "")
            QtCore.QTimer.singleShot(0, self._maybe_run_next_peer)

        self._send_via_worker(
            session=self._session,
            agent_name="Main",
            agent_id=self._hub.main_agent_id(),
            text=text,
            on_main_final=_after_main_final,
            stream_into=assistant_bubble,
        )

    def _maybe_run_next_peer(self) -> None:
        if not self._peer_queue:
            if self._skip_mode:
                QtCore.QTimer.singleShot(0, self._auto_continue_next)
            return
        base_next = self._peer_queue[0] if self._peer_queue else None
        aid: str | None = None

        if isinstance(self._resume_next_peer_id, str) and self._resume_next_peer_id:
            if self._resume_next_peer_id in self._peer_queue:
                self._peer_queue.remove(self._resume_next_peer_id)
                aid = self._resume_next_peer_id
            self._resume_next_peer_id = None
        elif isinstance(self._next_override_agent_id, str) and self._next_override_agent_id:
            if self._next_override_agent_id in self._peer_queue and self._next_override_agent_id != self._last_speaker_id:
                if isinstance(base_next, str) and base_next and base_next != self._next_override_agent_id:
                    self._resume_next_peer_id = base_next
                self._peer_queue.remove(self._next_override_agent_id)
                aid = self._next_override_agent_id
            self._next_override_agent_id = None

        if aid is None:
            aid = self._peer_queue.pop(0)
        try:
            agent = self._hub.get(aid)
        except Exception:
            QtCore.QTimer.singleShot(0, self._maybe_run_next_peer)
            return
        prompt = (
            "Context:\n"
            f"- User: {self._last_user_text.strip()}\n"
            f"- Main agent reply: {_strip_think(self._last_main_text)}\n\n"
            "Now respond with your perspective as a peer agent. Be concise.\n"
        )
        self._send_via_worker(
            session=agent.session,
            agent_name=agent.name,
            agent_id=agent.agent_id,
            text=prompt,
            on_done=lambda: QtCore.QTimer.singleShot(0, self._maybe_run_next_peer),
        )

    def _auto_continue_next(self) -> None:
        if not self._skip_mode:
            return
        if self._workers:
            return

        agents = self._hub.list_agents()
        if not agents:
            return
        order = [a.agent_id for a in agents]
        main_id = self._hub.main_agent_id()

        last_id = self._last_speaker_id
        if not isinstance(last_id, str) or not last_id or last_id == "user" or last_id not in order:
            base_next_id = main_id
        else:
            try:
                i = order.index(last_id)
                base_next_id = order[(i + 1) % len(order)]
            except Exception:
                base_next_id = main_id

        if isinstance(self._resume_next_agent_id, str) and self._resume_next_agent_id:
            next_id = self._resume_next_agent_id
            self._resume_next_agent_id = None
        elif isinstance(self._next_override_agent_id, str) and self._next_override_agent_id:
            if self._next_override_agent_id != last_id:
                next_id = self._next_override_agent_id
                if base_next_id != next_id:
                    self._resume_next_agent_id = base_next_id
            else:
                next_id = base_next_id
            self._next_override_agent_id = None
        else:
            next_id = base_next_id

        try:
            nxt = self._hub.get(next_id)
        except Exception:
            nxt = self._hub.get(main_id)

        prev_name = self._last_speaker_name or "Previous agent"
        prev_text = self._last_speaker_text or ""
        if last_id and last_id != nxt.agent_id:
            prev_text = _strip_think(prev_text)

        topic = (self._last_user_text or "").strip()
        prompt = (
            "You are in a round-robin multi-agent discussion.\n"
            f"User topic: {topic}\n\n"
            f"Previous speaker ({prev_name}) said:\n{prev_text}\n\n"
            "Continue the discussion with your next turn. Be concise and useful.\n"
        )

        self._send_via_worker(
            session=nxt.session,
            agent_name=nxt.name,
            agent_id=nxt.agent_id,
            text=prompt,
            on_done=lambda: QtCore.QTimer.singleShot(0, self._auto_continue_next),
        )

    def _send_via_worker(
        self,
        *,
        session: ChatSession,
        agent_name: str,
        agent_id: str | None = None,
        text: str,
        on_main_final: Optional[callable] = None,
        on_done: Optional[callable] = None,
        stream_into: Bubble | None = None,
    ) -> None:
        worker = _Worker(
            session=session,
            text=text,
            show_tool_events=self._show_tool_events,
            hide_think=self._hide_think,
            parent=self,
        )
        worker.signals.tool.connect(self._signals.append_tool.emit)

        assistant_bubble: Bubble | None = stream_into

        def on_partial(t: str) -> None:
            nonlocal assistant_bubble
            if assistant_bubble is None:
                assistant_bubble = self._append_message("", kind="assistant")
            assistant_bubble.set_streaming_text(f"**{agent_name}:**\n\n" + (t if isinstance(t, str) else str(t)))
            QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
            QtCore.QTimer.singleShot(0, self._update_bubble_widths)

        def on_final(t: str) -> None:
            nonlocal assistant_bubble
            if assistant_bubble is None:
                assistant_bubble = self._append_message("", kind="assistant")
            raw = t if isinstance(t, str) else str(t)
            cleaned, cmds = _extract_hash_commands(raw)
            did_skip = ("skip" in cmds) and (not cleaned.strip())
            if did_skip:
                assistant_bubble.set_text(f"**{agent_name}:**\n\n_(skipped)_")
                self._signals.append_tool.emit(f"[skip] {agent_name} passed.")
            else:
                assistant_bubble.set_text(f"**{agent_name}:**\n\n" + cleaned)
            QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
            QtCore.QTimer.singleShot(0, self._update_bubble_widths)
            # Refresh Agents tree after each completed turn (captures any submodel changes)
            QtCore.QTimer.singleShot(0, self._refresh_submodel_list)
            if agent_id:
                self._last_speaker_id = str(agent_id)
                self._last_speaker_name = str(agent_name or "")
                self._last_speaker_text = "" if did_skip else cleaned
                if not did_skip:
                    self._maybe_set_next_override_from_text(self._last_speaker_text)
            if on_main_final is not None:
                try:
                    on_main_final(cleaned)
                except Exception:
                    pass
            if on_done is not None:
                try:
                    on_done()
                except Exception:
                    pass

        worker.signals.assistant_partial.connect(on_partial)
        worker.signals.assistant_final.connect(on_final)
        worker.signals.busy.connect(self._signals.set_busy.emit)

        def _on_busy(busy: bool) -> None:
            if not agent_id:
                return
            self._agent_busy[str(agent_id)] = bool(busy)
            QtCore.QTimer.singleShot(0, self._refresh_submodel_list)

        worker.signals.busy.connect(_on_busy)

        def _on_err(m: str) -> None:
            try:
                ctx = "\n".join(list(self._event_ring)[-25:])
            except Exception:
                ctx = ""
            _LOG.error(
                "llm_error chat_id=%s model=%s error=%s\nrecent_events:\n%s",
                str(self._active_chat_id),
                str(getattr(session.cfg, "model", "")),
                str(m),
                ctx,
            )
            self._signals.append_tool.emit(f"Error: {m}")
            if agent_id:
                try:
                    self._agent_busy[str(agent_id)] = False
                except Exception:
                    pass
                QtCore.QTimer.singleShot(0, self._refresh_submodel_list)

        worker.signals.error.connect(_on_err)
        worker.signals.tokens.connect(self._signals.set_tokens.emit)
        worker.finished.connect(lambda: self._workers.discard(worker))
        worker.finished.connect(self._persist_current_chat)
        if agent_id:
            worker.finished.connect(lambda: self._agent_busy.__setitem__(str(agent_id), False))
            worker.finished.connect(lambda: QtCore.QTimer.singleShot(0, self._refresh_submodel_list))
        self._workers.add(worker)
        worker.start()

    def _agents_config_dir(self) -> Path:
        d = (self._store_root / "chat_history" / "agents_configs").resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _selected_agent_id(self) -> str | None:
        it = self._agents_tree.currentItem()
        if it is None:
            return None
        data = it.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return None
        if data.get("kind") == "agent":
            aid = data.get("id")
        elif data.get("kind") == "submodel":
            aid = data.get("agent_id")
        else:
            return None
        return aid if isinstance(aid, str) and aid else None

    def _new_agent_dialog(self) -> None:
        self._agent_properties_dialog(agent_id=None)

    def _edit_selected_agent(self) -> None:
        aid = self._selected_agent_id()
        if not aid:
            return
        self._agent_properties_dialog(agent_id=aid)

    def _agent_properties_dialog(self, *, agent_id: str | None) -> None:
        is_new = agent_id is None
        cur_name = "Friend"
        cur_model = str(self._session.cfg.model)
        cur_prompt = "You are a helpful assistant."
        cur_mem: str | None = ""
        mem_enabled = False
        if not is_new:
            a = self._hub.get(str(agent_id))
            cur_name = a.name
            cur_model = a.model
            cur_prompt = a.system_prompt
            cur_mem = a.memory_path or ""
            mem_enabled = a.memory_path is not None
            if str(agent_id) == self._hub.main_agent_id():
                # Keep the Main agent name stable in the UI.
                cur_name = "Main"

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("New agent" if is_new else "Edit agent")
        dlg.setObjectName("dialog")
        dlg.setMinimumWidth(720)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(6)

        form = QtWidgets.QFormLayout()
        form.setSpacing(6)
        lay.addLayout(form)

        name_edit = QtWidgets.QLineEdit()
        name_edit.setText(cur_name)
        if not is_new and str(agent_id) == self._hub.main_agent_id():
            name_edit.setReadOnly(True)
        form.addRow("Name", name_edit)

        model_combo = QtWidgets.QComboBox()
        model_combo.addItems(list(SUPPORTED_MODELS))
        idx = model_combo.findText(cur_model)
        if idx >= 0:
            model_combo.setCurrentIndex(idx)
        form.addRow("Model", model_combo)

        # Keep all one-line fields compact; only the system prompt should be multi-line.
        try:
            one_line_h = int(model_combo.sizeHint().height())
            name_edit.setFixedHeight(one_line_h)
            model_combo.setFixedHeight(one_line_h)
        except Exception:
            one_line_h = None

        mem_row = QtWidgets.QWidget()
        mem_l = QtWidgets.QHBoxLayout(mem_row)
        mem_l.setContentsMargins(0, 0, 0, 0)
        mem_l.setSpacing(8)
        mem_chk = QtWidgets.QCheckBox()
        mem_chk.setToolTip("Enable memory file")
        mem_chk.setChecked(bool(mem_enabled))
        mem_edit = QtWidgets.QLineEdit()
        mem_edit.setText(cur_mem or "")
        mem_edit.setEnabled(mem_chk.isChecked())
        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.setObjectName("btn_ghost")
        btn_browse.setEnabled(mem_chk.isChecked())
        mem_chk.toggled.connect(lambda on: (mem_edit.setEnabled(bool(on)), btn_browse.setEnabled(bool(on))))
        mem_l.addWidget(mem_chk, 0)
        mem_l.addWidget(mem_edit, 1)
        mem_l.addWidget(btn_browse, 0)
        try:
            h = int(one_line_h or model_combo.sizeHint().height())
            for w in (mem_row, mem_chk, mem_edit, btn_browse):
                w.setFixedHeight(h)
        except Exception:
            pass
        form.addRow("Memory", mem_row)

        prompt = QtWidgets.QTextEdit()
        prompt.setObjectName("system_edit")
        prompt.setPlainText(cur_prompt)
        prompt.setMinimumHeight(220)
        prompt.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        form.addRow("System prompt", prompt)

        def browse() -> None:
            start = str((self._store_root / "chat_history").resolve())
            path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Select memory file", start, "Markdown (*.md);;All (*.*)")
            if path:
                try:
                    rp = Path(path).resolve().relative_to(self._store_root.resolve())
                    mem_edit.setText(rp.as_posix())
                except Exception:
                    mem_edit.setText("")

        btn_browse.clicked.connect(browse)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setObjectName("btn_ghost")
        btn_ok = QtWidgets.QPushButton("Save")
        btn_ok.setObjectName("btn_primary")
        row.addWidget(btn_cancel)
        row.addWidget(btn_ok)
        lay.addLayout(row)

        def save() -> None:
            nm = str(name_edit.text() or "").strip()
            mdl = str(model_combo.currentText() or "").strip()
            sp = str(prompt.toPlainText() or "").strip()
            mem = str(mem_edit.text() or "").strip()
            mem_val: str | None
            if not mem_chk.isChecked():
                mem_val = None
            else:
                mem_val = mem  # empty -> default path
            try:
                if is_new:
                    self._hub.create_peer(name=nm, model=mdl, system_prompt=sp, memory_path=mem_val)
                    self._append_message(f"Created agent: {nm}", kind="tool")
                else:
                    if str(agent_id) == self._hub.main_agent_id():
                        # Main agent is editable via a dedicated update path.
                        if not nm:
                            nm = "Main"
                        self._hub.update_main(model=mdl, system_prompt=sp, memory_path=mem_val)
                        self._append_message("Main agent updated.", kind="tool")
                    else:
                        self._hub.update_peer(agent_id=str(agent_id), name=nm, model=mdl, system_prompt=sp, memory_path=mem_val)
                        self._append_message("Agent updated.", kind="tool")
            except AgentHubError as e:
                QtWidgets.QMessageBox.warning(dlg, "Invalid agent", str(e))
                return
            except Exception as e:
                QtWidgets.QMessageBox.warning(dlg, "Error", f"{type(e).__name__}: {e}")
                return
            self._refresh_submodel_list()
            dlg.accept()

        btn_cancel.clicked.connect(dlg.reject)
        btn_ok.clicked.connect(save)
        try:
            dlg.adjustSize()
            dlg.resize(max(720, dlg.sizeHint().width()), dlg.sizeHint().height())
        except Exception:
            pass
        dlg.exec()

    def _delete_selected_agent(self) -> None:
        aid = self._selected_agent_id()
        if not aid or aid == self._hub.main_agent_id():
            return
        try:
            a = self._hub.get(aid)
        except Exception:
            return
        resp = QtWidgets.QMessageBox.question(
            self,
            "Delete agent",
            f"Delete agent '{a.name}'?",
        )
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            self._hub.remove_peer(agent_id=aid)
            self._append_message(f"Deleted agent: {a.name}", kind="tool")
        except Exception as e:
            self._append_message(f"Failed to delete agent: {type(e).__name__}: {e}", kind="tool")
        self._refresh_submodel_list()

    def _save_agents_config_ui(self) -> None:
        d = self._agents_config_dir()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save agents configuration",
            str((d / "agents_config.json").resolve()),
            "JSON (*.json);;All (*.*)",
        )
        if not path:
            return
        try:
            save_agents_config(Path(path), self._hub.export_config())
            self._append_message(f"Saved agents config: {Path(path).name}", kind="tool")
        except Exception as e:
            self._append_message(f"Failed to save config: {type(e).__name__}: {e}", kind="tool")

    def _load_agents_config_ui(self) -> None:
        d = self._agents_config_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load agents configuration",
            str(d.resolve()),
            "JSON (*.json);;All (*.*)",
        )
        if not path:
            return
        try:
            data = load_agents_config(Path(path))
        except Exception as e:
            self._append_message(f"Failed to read config: {type(e).__name__}: {e}", kind="tool")
            return

        # Let the user choose which agents to import from the file.
        main_d = data.get("main") if isinstance(data.get("main"), dict) else None
        friends_d = data.get("friends") if isinstance(data.get("friends"), list) else []

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Import agents")
        dlg.setModal(True)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QtWidgets.QLabel(f"Import agents from: {Path(path).name}")
        header.setStyleSheet("font-weight: 700;")
        layout.addWidget(header)

        select_all = QtWidgets.QCheckBox("Select all")
        select_all.setChecked(True)
        layout.addWidget(select_all)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout.addWidget(scroll, 1)

        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        inner_layout = QtWidgets.QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(6)

        checks: list[tuple[str, QtWidgets.QCheckBox]] = []
        if main_d is not None:
            cb = QtWidgets.QCheckBox("Main")
            cb.setChecked(True)
            inner_layout.addWidget(cb)
            checks.append(("main", cb))
        for i, f in enumerate(friends_d):
            if not isinstance(f, dict):
                continue
            nm = str(f.get("name") or f"Friend {i+1}").strip() or f"Friend {i+1}"
            cb = QtWidgets.QCheckBox(nm)
            cb.setChecked(True)
            inner_layout.addWidget(cb)
            checks.append((f"friend:{i}", cb))
        inner_layout.addStretch(1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_import = QtWidgets.QPushButton("Import")
        btn_import.setDefault(True)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_import)
        layout.addLayout(btn_row)

        def set_all(state: bool) -> None:
            for _, cb in checks:
                cb.blockSignals(True)
                try:
                    cb.setChecked(state)
                finally:
                    cb.blockSignals(False)

        def sync_select_all() -> None:
            if not checks:
                select_all.setChecked(False)
                return
            all_on = all(cb.isChecked() for _, cb in checks)
            select_all.blockSignals(True)
            try:
                select_all.setChecked(all_on)
            finally:
                select_all.blockSignals(False)

        select_all.toggled.connect(lambda v: set_all(bool(v)))
        for _, cb in checks:
            cb.toggled.connect(lambda _v: sync_select_all())

        chosen: dict | None = None

        def do_import() -> None:
            nonlocal chosen
            out: dict = {}
            # main
            if main_d is not None:
                for key, cb in checks:
                    if key == "main":
                        if cb.isChecked():
                            out["main"] = dict(main_d)
                        break
            # friends
            out_friends: list[dict] = []
            for key, cb in checks:
                if not key.startswith("friend:"):
                    continue
                if not cb.isChecked():
                    continue
                idx_s = key.split(":", 1)[1]
                try:
                    idx = int(idx_s)
                except Exception:
                    continue
                if idx < 0 or idx >= len(friends_d):
                    continue
                f = friends_d[idx]
                if isinstance(f, dict):
                    out_friends.append(dict(f))
            out["friends"] = out_friends

            # Confirm replacement semantics.
            resp = QtWidgets.QMessageBox.question(
                dlg,
                "Import agents",
                "Replace current friend agents with the selected agents?",
            )
            if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            chosen = out
            dlg.accept()

        btn_cancel.clicked.connect(dlg.reject)
        btn_import.clicked.connect(do_import)

        try:
            dlg.adjustSize()
            dlg.resize(max(520, dlg.sizeHint().width()), min(640, max(360, dlg.sizeHint().height())))
        except Exception:
            pass
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted or chosen is None:
            return

        try:
            self._hub.load_config(chosen)  # type: ignore[arg-type]
            n_f = len(chosen.get("friends") or []) if isinstance(chosen, dict) else 0
            self._append_message(f"Loaded agents config: {Path(path).name} (friends: {n_f})", kind="tool")
        except Exception as e:
            self._append_message(f"Failed to load config: {type(e).__name__}: {e}", kind="tool")
            return
        # Sync main-model selector with loaded configuration.
        try:
            mdl = str(self._session.cfg.model or "")
            idx = self._model_combo.findText(mdl)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        except Exception:
            pass
        self._refresh_submodel_list()

    # ---- chat history ----

    def _refresh_chat_list(self, *, select_chat_id: str | None = None) -> None:
        metas = list_chats(self._store_root)
        self._chat_list.blockSignals(True)
        try:
            self._chat_list.clear()
            for m in metas:
                item = QtWidgets.QListWidgetItem(m.title)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, m.chat_id)
                item.setToolTip(f"{m.chat_id}\nUpdated: {m.updated_at}")
                self._chat_list.addItem(item)

            # Ensure current chat exists on disk.
            self._persist_current_chat()
            if select_chat_id:
                for i in range(self._chat_list.count()):
                    it = self._chat_list.item(i)
                    if it.data(QtCore.Qt.ItemDataRole.UserRole) == select_chat_id:
                        self._chat_list.setCurrentItem(it)
                        break
        finally:
            self._chat_list.blockSignals(False)
        self._refresh_submodel_list()

    def _persist_current_chat(self) -> None:
        try:
            rec = self._session.to_record()
            # Avoid creating lots of empty "New chat" files; we prune them on startup too.
            conv = rec.get("conversation")
            meaningful = False
            if isinstance(conv, list):
                for it in conv:
                    if not isinstance(it, dict):
                        continue
                    if it.get("role") not in {"user", "assistant"}:
                        continue
                    content = it.get("content")
                    if not isinstance(content, list) or not content:
                        continue
                    t = content[0].get("text") if isinstance(content[0], dict) else None
                    if isinstance(t, str) and t.strip():
                        meaningful = True
                        break
            try:
                agents_state = self._hub.export_state()
                rec["agents_state"] = agents_state
            except Exception:
                agents_state = None
            # If there are friend agents, persist the chat even before messages exist.
            has_agents = False
            if isinstance(agents_state, dict):
                friends = agents_state.get("friends")
                has_agents = isinstance(friends, list) and len(friends) > 0
            if not meaningful and not has_agents:
                return
            save_chat(self._store_root, rec)
        except Exception:
            pass

    def _clear_chat_view(self) -> None:
        self._bubbles.clear()
        # Remove all widgets except the trailing stretch.
        while self._chat_layout.count() > 1:
            it = self._chat_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

    def _load_chat_into_ui(self, chat_id: str) -> None:
        self._switching_chat = True
        try:
            data = load_chat(self._store_root, chat_id)
            self._session.load_record(data)
            self._active_chat_id = self._session.chat_id

            # Recreate the agent hub for this chat so per-chat agent crews can differ.
            self._hub = AgentHub(base_config=self._session.cfg, make_session=self._instantiate_session, repo_root=self._store_root)
            self._hub.set_main(agent_id=str(self._session.chat_id), name="Main", session=self._session)
            agents_state = data.get("agents_state")
            if isinstance(agents_state, dict):
                try:
                    self._hub.load_state(agents_state)
                except Exception:
                    pass
            else:
                # Back-compat: load old-style config if present.
                agents_cfg = data.get("agents_config")
                if isinstance(agents_cfg, dict):
                    try:
                        self._hub.load_config(agents_cfg)
                    except Exception:
                        pass

            # Sync model picker.
            idx = self._model_combo.findText(self._session.cfg.model)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)

            self._clear_chat_view()

            # Replay conversation items into bubbles (user/assistant only).
            for item in self._session.to_record().get("conversation", []):
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                if role in {"user", "assistant"}:
                    parts = item.get("content") or []
                    if isinstance(parts, list) and parts:
                        t = parts[0].get("text")
                        if isinstance(t, str) and t.strip():
                            self._append_message(t.strip(), kind="user" if role == "user" else "assistant")
            self._refresh_submodel_list()
        finally:
            self._switching_chat = False

    def _on_chat_selected(self) -> None:
        if self._switching_chat:
            return
        it = self._chat_list.currentItem()
        if it is None:
            return
        chat_id = it.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(chat_id, str) or not chat_id:
            return
        if chat_id == self._active_chat_id:
            return
        self._persist_current_chat()
        self._load_chat_into_ui(chat_id)

    def _new_chat(self) -> None:
        self._persist_current_chat()
        self._session.chat_id = new_chat_id()
        self._session.title = "New chat"
        self._session.reset(keep_system_prompt=True)
        self._active_chat_id = self._session.chat_id
        self._clear_chat_view()
        self._append_message("New chat created.", kind="tool")
        self._persist_current_chat()
        self._refresh_chat_list(select_chat_id=self._active_chat_id)
        self._refresh_submodel_list()

    def _rename_chat(self) -> None:
        it = self._chat_list.currentItem()
        if it is None:
            return
        chat_id = it.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(chat_id, str):
            return
        cur = it.text()
        text, ok = QtWidgets.QInputDialog.getText(self, "Rename chat", "Title:", text=cur)
        if not ok:
            return
        new_title = (text or "").strip()
        if not new_title:
            return
        if chat_id == self._active_chat_id:
            self._session.title = new_title
        try:
            data = load_chat(self._store_root, chat_id)
            data["title"] = new_title
            save_chat(self._store_root, data)
        except Exception:
            pass
        self._refresh_chat_list(select_chat_id=self._active_chat_id)

    def _delete_chat(self) -> None:
        it = self._chat_list.currentItem()
        if it is None:
            return
        chat_id = it.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(chat_id, str):
            return
        if chat_id == self._active_chat_id:
            # Don't delete the active chat; create a new one instead.
            self._new_chat()
        try:
            delete_chat(self._store_root, chat_id)
        except Exception:
            pass
        self._refresh_chat_list(select_chat_id=self._active_chat_id)
        self._refresh_submodel_list()

    # ---- submodels UI ----

    def _refresh_submodel_list(self) -> None:
        # Tree structure: agents (top-level) -> submodels (children).
        self._agents_tree.blockSignals(True)
        try:
            expanded_by_id: dict[str, bool] = {}
            for i in range(self._agents_tree.topLevelItemCount()):
                it0 = self._agents_tree.topLevelItem(i)
                if it0 is None:
                    continue
                d0 = it0.data(0, QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(d0, dict) and d0.get("kind") == "agent":
                    aid0 = d0.get("id")
                    if isinstance(aid0, str) and aid0:
                        expanded_by_id[aid0] = bool(it0.isExpanded())

            cur_kind: str | None = None
            cur_id: str | None = None
            cur_agent: str | None = None
            it = self._agents_tree.currentItem()
            if it is not None:
                data = it.data(0, QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(data, dict):
                    cur_kind = str(data.get("kind") or "")
                    if cur_kind == "submodel":
                        sid = data.get("id")
                        cur_id = str(sid) if isinstance(sid, str) and sid else None
                        pid = data.get("agent_id")
                        cur_agent = str(pid) if isinstance(pid, str) and pid else None
                    elif cur_kind == "agent":
                        pid = data.get("id")
                        cur_agent = str(pid) if isinstance(pid, str) and pid else None

            self._agents_tree.clear()

            # Build tree per agent.
            for a in self._hub.list_agents():
                is_busy = bool(self._agent_busy.get(a.agent_id, False))
                label_name = f"{a.name} [talking]" if is_busy else a.name
                label = f"{label_name}\n{a.model}" if a.model else label_name
                top = QtWidgets.QTreeWidgetItem(["", label])
                top.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "agent", "id": a.agent_id})
                if is_busy:
                    font = top.font(1)
                    font.setBold(True)
                    top.setFont(1, font)
                    top.setForeground(1, QtGui.QBrush(QtGui.QColor("#e9f0ff")))
                    top.setBackground(1, QtGui.QBrush(QtGui.QColor(95, 135, 255, 40)))
                self._agents_tree.addTopLevelItem(top)

                try:
                    metas = a.session.list_submodels()
                except Exception:
                    metas = []
                for m in metas:
                    sid = str(m.get("id") or "")
                    title = str(m.get("title") or sid)
                    model = str(m.get("model") or "")
                    depth = int(m.get("depth", 0))
                    state = str(m.get("state") or "")
                    child_label = f"{title}"
                    if model:
                        child_label += f"\n{model}"
                    meta_bits = []
                    if depth:
                        meta_bits.append(f"d{depth}")
                    if state:
                        meta_bits.append(state)
                    if meta_bits:
                        child_label += f"  ({', '.join(meta_bits)})"
                    child = QtWidgets.QTreeWidgetItem(["", child_label])
                    child.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "submodel", "id": sid, "agent_id": a.agent_id})
                    top.addChild(child)

                if a.agent_id in expanded_by_id:
                    top.setExpanded(bool(expanded_by_id[a.agent_id]))
                else:
                    top.setExpanded(True if a.agent_id == self._hub.main_agent_id() else False)
            self._update_agents_tree_labels()
            self._update_agent_mentions_model()

            # Restore selection if possible
            if isinstance(cur_agent, str) and cur_agent:
                for i in range(self._agents_tree.topLevelItemCount()):
                    top = self._agents_tree.topLevelItem(i)
                    if top is None:
                        continue
                    dtop = top.data(0, QtCore.Qt.ItemDataRole.UserRole)
                    if not isinstance(dtop, dict) or dtop.get("kind") != "agent" or dtop.get("id") != cur_agent:
                        continue
                    if cur_kind == "submodel" and isinstance(cur_id, str) and cur_id:
                        for j in range(top.childCount()):
                            it2 = top.child(j)
                            d = it2.data(0, QtCore.Qt.ItemDataRole.UserRole)
                            if isinstance(d, dict) and d.get("kind") == "submodel" and d.get("id") == cur_id:
                                self._agents_tree.setCurrentItem(it2)
                                return
                    # Fallback: keep agent selected
                    self._agents_tree.setCurrentItem(top)
                    return
        finally:
            self._agents_tree.blockSignals(False)

    def _update_agents_tree_labels(self) -> None:
        """Update +/- icon for each top-level agent item."""

        for i in range(self._agents_tree.topLevelItemCount()):
            top = self._agents_tree.topLevelItem(i)
            if top is None:
                continue
            if top.childCount() > 0:
                icon = self._agent_icon_open if bool(top.isExpanded()) else self._agent_icon_closed
                if icon is not None:
                    top.setIcon(0, icon)
            else:
                top.setIcon(0, QtGui.QIcon())

    def _on_agents_tree_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        # Clicking the icon column toggles expand/collapse like a file explorer.
        try:
            if int(column) != 0:
                return
        except Exception:
            return
        if item is None or item.childCount() <= 0:
            return
        if item.isExpanded():
            item.setExpanded(False)
        else:
            item.setExpanded(True)
        self._update_agents_tree_labels()

    def _selected_submodel_id(self) -> str | None:
        it = self._agents_tree.currentItem()
        if it is None:
            return None
        data = it.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict) or data.get("kind") != "submodel":
            return None
        sid = data.get("id")
        return sid if isinstance(sid, str) and sid else None

    def _selected_submodel_agent_id(self) -> str | None:
        it = self._agents_tree.currentItem()
        if it is None:
            return None
        data = it.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict) or data.get("kind") != "submodel":
            return None
        aid = data.get("agent_id")
        return aid if isinstance(aid, str) and aid else None

    def _open_selected_submodel(self) -> None:
        sid = self._selected_submodel_id()
        aid = self._selected_submodel_agent_id()
        if not sid or not aid:
            return
        try:
            sess = self._hub.get(aid).session
        except Exception:
            sess = self._session
        sm = sess.get_submodel(sid)
        if sm is None:
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Submodel • {sm.title}")
        dlg.setObjectName("dialog")
        dlg.resize(860, 700)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        header = QtWidgets.QLabel(f"{sm.title} — {sm.model}")
        header.setObjectName("dialog_label")
        lay.addWidget(header)

        sys = QtWidgets.QTextEdit()
        sys.setReadOnly(True)
        sys.setObjectName("system_edit")
        sys.setPlainText(sm.system_prompt)
        sys.setFixedHeight(120)
        lay.addWidget(sys)

        view = QtWidgets.QTextBrowser()
        view.setObjectName("bubble_text")
        view.setOpenExternalLinks(True)
        view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        view.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAnywhere)

        def render() -> None:
            parts: list[str] = []
            for item in sm.transcript:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                if role == "main":
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(f"### Main → Submodel\n{txt}\n")
                elif role == "assistant":
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(f"### Submodel\n{txt}\n")
                elif role == "tool":
                    parts.append("### Tool event\n```json\n" + json.dumps(item.get("item"), ensure_ascii=False)[:4000] + "\n```\n")
                elif role == "error":
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(f"### Error\n{txt}\n")
            transcript = "\n".join(parts)
            if hasattr(view, "setMarkdown"):
                view.setMarkdown(transcript)
            else:
                view.setHtml(f"<pre>{_esc(transcript)}</pre>")

        render()
        lay.addWidget(view, 1)

        timer = QtCore.QTimer(dlg)
        timer.setInterval(500)
        timer.timeout.connect(render)
        timer.start()

        dlg.exec()

    def _close_selected_submodel(self) -> None:
        sid = self._selected_submodel_id()
        aid = self._selected_submodel_agent_id()
        if not sid or not aid:
            return
        try:
            sess = self._hub.get(aid).session
        except Exception:
            sess = self._session
        sess.close_submodel(sid)
        self._refresh_submodel_list()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        # Best-effort shutdown of background threads to avoid "QThread destroyed" warnings.
        _LOG.info("ui_close chat_id=%s model=%s", str(self._active_chat_id), str(self._session.cfg.model))
        try:
            if self._mention_timer is not None:
                self._mention_timer.stop()
        except Exception:
            pass
        self._hide_mention_popup()
        for w in list(self._workers):
            try:
                w.requestInterruption()
            except Exception:
                pass
            try:
                w.quit()
            except Exception:
                pass
        for w in list(self._workers):
            try:
                w.wait(500)
            except Exception:
                pass
        self._workers.clear()
        try:
            self._session.shutdown()
        except Exception:
            pass
        self._persist_current_chat()
        super().closeEvent(event)


class _WorkerSignals(QtCore.QObject):
    assistant_partial = QtCore.Signal(str)
    assistant_final = QtCore.Signal(str)
    tool = QtCore.Signal(str)
    error = QtCore.Signal(str)
    busy = QtCore.Signal(bool)
    tokens = QtCore.Signal(str)


class _Worker(QtCore.QThread):
    def __init__(
        self,
        *,
        session: ChatSession,
        text: str,
        show_tool_events: bool,
        hide_think: bool,
        parent: Optional[QtCore.QObject],
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.text = text
        self.show_tool_events = show_tool_events
        self.hide_think = hide_think
        self.signals = _WorkerSignals()

    def run(self) -> None:  # type: ignore[override]
        self.signals.busy.emit(True)
        try:
            if self.isInterruptionRequested():
                return
            full_raw = ""
            last_emit = 0.0
            last_tok_emit = 0.0
            new_items: list[dict[str, Any]] = []
            prompt_tok_est = self.session.estimate_prompt_tokens(user_text=self.text)
            max_ctx = int(getattr(self.session.cfg, "context_window_tokens", 0) or 0)

            for ev in self.session.send_stream(self.text):
                if self.isInterruptionRequested():
                    return
                et = ev.get("type")
                if et == "assistant_delta":
                    d = ev.get("delta")
                    if isinstance(d, str) and d:
                        full_raw += d
                    now = time.monotonic()
                    if now - last_emit >= 0.05:
                        shown = _strip_think(full_raw) if self.hide_think else full_raw
                        self.signals.assistant_partial.emit(shown)
                        last_emit = now
                    if now - last_tok_emit >= 0.20:
                        out_tok_est = self.session.estimate_tokens(full_raw)
                        used_est = int(prompt_tok_est + out_tok_est)
                        if max_ctx > 0:
                            pct = (used_est / max_ctx) * 100.0
                            self.signals.tokens.emit(f"~{used_est:,}/{max_ctx:,} tok ({pct:.1f}%)")
                        else:
                            self.signals.tokens.emit(f"~{used_est:,} tok")
                        last_tok_emit = now
                elif et == "turn_done":
                    ni = ev.get("new_items")
                    if isinstance(ni, list):
                        new_items = [x for x in ni if isinstance(x, dict)]

            if self.isInterruptionRequested():
                return

            if self.show_tool_events:
                for item in new_items:
                    t = item.get("type")
                    if t == "function_call":
                        name = item.get("name", "")
                        args = item.get("arguments", "")
                        self.signals.tool.emit(f"[tool call] {name}({args})")
                    elif t == "function_call_output":
                        # If the tool produced images (e.g., python_sandbox / render_plot),
                        # emit image markers so bubbles can render them inline.
                        try:
                            out_full = item.get("output", "")
                            if isinstance(out_full, str) and out_full.strip().startswith("{"):
                                d = json.loads(out_full)
                                if isinstance(d, dict):
                                    imgs = d.get("image_paths")
                                    if isinstance(imgs, list):
                                        for p in imgs[:8]:
                                            if isinstance(p, str) and p.strip():
                                                self.signals.tool.emit(f"[[image:{p.strip()}]]")
                        except Exception:
                            pass
                        out = item.get("output", "")
                        out_s = out if isinstance(out, str) else str(out)
                        if len(out_s) > 1200:
                            out_s = out_s[:1200] + " …"
                        self.signals.tool.emit(f"[tool output] {out_s}")
                    elif t in {"reasoning", "reasoning_summary"}:
                        # These items are output-only; we show a short summary (not raw JSON),
                        # and we never replay them back as input.
                        summary_txt = ""
                        s = item.get("summary")
                        if isinstance(s, str):
                            summary_txt = s.strip()
                        elif isinstance(s, list):
                            parts: list[str] = []
                            for x in s:
                                if isinstance(x, str) and x.strip():
                                    parts.append(x.strip())
                                elif isinstance(x, dict):
                                    tt = x.get("text")
                                    if isinstance(tt, str) and tt.strip():
                                        parts.append(tt.strip())
                            summary_txt = "\n".join(parts).strip()

                        if summary_txt:
                            if len(summary_txt) > 1200:
                                summary_txt = summary_txt[:1200] + " …"
                            self.signals.tool.emit(f"[reasoning_summary] {summary_txt}")
                        else:
                            self.signals.tool.emit("[reasoning] (omitted; not replayed)")
                    elif t == "web_search_call":
                        action = item.get("action") or {}
                        q = action.get("query") or ""
                        sources = action.get("sources") or []
                        msg = f"[web_search] query={q!r} sources={len(sources)}"
                        self.signals.tool.emit(msg)
                        shown = 0
                        for s in sources:
                            if not isinstance(s, dict):
                                continue
                            url = s.get("url")
                            title = s.get("title")
                            if not isinstance(url, str) or not url:
                                continue
                            shown += 1
                            ttxt = f" — {title}" if isinstance(title, str) and title else ""
                            self.signals.tool.emit(f"  {shown}. {url}{ttxt}")
                            if shown >= 5:
                                break
                    elif t == "file_search_call":
                        q = item.get("query") or ""
                        status = item.get("status") or ""
                        msg = f"[file_search] query={q!r} status={status!r}"
                        self.signals.tool.emit(msg)
                        results = item.get("results") or item.get("search_results") or []
                        if isinstance(results, list) and results:
                            self.signals.tool.emit(f"  results={len(results)}")
                            for i, r in enumerate(results[:5], 1):
                                if not isinstance(r, dict):
                                    continue
                                fname = r.get("filename") or r.get("file_name") or ""
                                score = r.get("score")
                                if fname:
                                    self.signals.tool.emit(f"  {i}. {fname} score={score}")

            out = (_strip_think(full_raw) if self.hide_think else full_raw).strip()
            self.signals.assistant_final.emit(out)
            self.signals.tokens.emit(self.session.usage_ratio_text())
        except Exception as e:
            try:
                preview = (self.text or "").replace("\r", "").strip()
                if len(preview) > 800:
                    preview = preview[:800] + " …"
                _LOG.error(
                    "worker_exception chat_id=%s model=%s error=%s\nprompt_preview:\n%s\ntraceback:\n%s",
                    str(getattr(self.session, "chat_id", "")),
                    str(getattr(getattr(self.session, "cfg", None), "model", "")),
                    f"{type(e).__name__}: {e}",
                    preview,
                    traceback.format_exc(),
                )
            except Exception:
                pass
            self.signals.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.signals.busy.emit(False)


def main() -> None:
    _setup_run_logging()
    w = ChatWindow()
    raise SystemExit(w.exec())


if __name__ == "__main__":
    main()
