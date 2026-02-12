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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json

from PySide6 import QtCore, QtGui, QtWidgets

from .chat_session import ChatConfig, ChatSession
from .chat_store import delete_chat, list_chats, load_chat, new_chat_id, save_chat
from .config import DEFAULT_MODEL, SUPPORTED_MODELS, load_config


def _strip_think(text: str) -> str:
    if not text:
        return text
    low = text.lower()
    if "</think>" in low:
        i = low.find("</think>")
        return text[i + len("</think>") :].lstrip()
    if low.lstrip().startswith("<think>"):
        return ""
    return text


def _repo_root() -> Path:
    # src/desktop_agent/chat_ui.py -> desktop_agent -> src -> repo root
    return Path(__file__).resolve().parents[2]


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
        # Prefer Qt's Markdown renderer if available so **bold** etc. show nicely.
        if hasattr(self._label, "setMarkdown"):
            try:
                self._label.setMarkdown(t)
                self._label.updateGeometry()
                self.updateGeometry()
                return
            except Exception:
                pass
        self._label.setHtml(f"<div class='msg'>{_esc(t).replace('\\n', '<br/>')}</div>")
        self._label.updateGeometry()
        self.updateGeometry()

    def set_streaming_text(self, text: str) -> None:
        # Streaming path: avoid re-parsing markdown on every tiny delta.
        t = text or ""
        self._label.setHtml(f"<div class='msg'>{_esc(t).replace('\\n', '<br/>')}</div>")
        self._label.updateGeometry()
        self.updateGeometry()


class ChatWindow(QtWidgets.QMainWindow):
    def __init__(self, *, cfg: Optional[ChatUIConfig] = None) -> None:
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        super().__init__()

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

        self._build_ui()
        self._apply_style()

        self.resize(self._cfg.width, self._cfg.height)
        self.setMinimumSize(420, 640)
        self.setWindowTitle("Desktop Agent • Chat")

        self._append_message(
            "Tip: You can ask it to use web search, read files, or even create+register new tools during the chat.",
            kind="tool",
        )
        self._refresh_chat_list(select_chat_id=self._active_chat_id)
        self._refresh_submodel_list()
        self._set_tokens(self._session.usage_ratio_text())

        # Keep the Agents tree fresh without requiring manual refresh.
        # If we miss an event (e.g. a tool spawned/closed a sub-agent), this timer
        # will reconcile within ~1s.
        self._agents_refresh_timer = QtCore.QTimer(self)
        self._agents_refresh_timer.setInterval(1000)
        self._agents_refresh_timer.timeout.connect(self._refresh_submodel_list)
        self._agents_refresh_timer.start()

    def exec(self) -> int:
        self.show()
        return int(self._app.exec())

    # ---- session ----

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
        )
        return s

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

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(8)

        self._btn_send = QtWidgets.QPushButton("Send")
        self._btn_send.setObjectName("btn_primary")
        self._btn_send.clicked.connect(self._on_send)
        right.addWidget(self._btn_send)

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

        row2 = QtWidgets.QHBoxLayout()
        lbl2 = QtWidgets.QLabel("Agents")
        lbl2.setObjectName("sidebar_title")
        row2.addWidget(lbl2)
        row2.addStretch(1)
        self._btn_refresh_sub = QtWidgets.QPushButton("Refresh")
        self._btn_refresh_sub.setObjectName("btn_ghost")
        self._btn_refresh_sub.clicked.connect(self._refresh_submodel_list)
        row2.addWidget(self._btn_refresh_sub)
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
        if obj is self._input and event.type() == QtCore.QEvent.Type.KeyPress:
            e = event  # type: ignore[assignment]
            if isinstance(e, QtGui.QKeyEvent):
                if e.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    if e.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                        return False
                    self._on_send()
                    return True
        return super().eventFilter(obj, event)

    # ---- actions ----

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def _set_tokens(self, text: str) -> None:
        self._tokens.setText(text)

    def _set_busy(self, busy: bool) -> None:
        self._btn_send.setEnabled(not busy)
        self._input.setEnabled(not busy)
        self._model_combo.setEnabled(not busy)
        self._btn_system.setEnabled(not busy)
        self._set_status("Thinking…" if busy else "")

    def _set_show_tool_events(self, enabled: bool) -> None:
        self._show_tool_events = bool(enabled)

    def _set_hide_think(self, enabled: bool) -> None:
        self._hide_think = bool(enabled)
        self._session.cfg.hide_think = self._hide_think

    def _on_model_changed(self, model: str) -> None:
        self._session.cfg.model = str(model)

    def _append_message(self, text: str, *, kind: str) -> Bubble:
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
            self._session.set_system_prompt(edit.toPlainText())
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
            self._session.set_system_prompt(sys_edit.toPlainText())
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
        mtc.setRange(0, 50)
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
        self._input.clear()

        self._append_message(text, kind="user")
        assistant_bubble: Bubble | None = None

        worker = _Worker(
            session=self._session,
            text=text,
            show_tool_events=self._show_tool_events,
            hide_think=self._hide_think,
            parent=self,
        )
        worker.signals.tool.connect(self._signals.append_tool.emit)

        def on_partial(t: str) -> None:
            nonlocal assistant_bubble
            if assistant_bubble is None:
                assistant_bubble = self._append_message("", kind="assistant")
            assistant_bubble.set_streaming_text(t if isinstance(t, str) else str(t))
            QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
            QtCore.QTimer.singleShot(0, self._update_bubble_widths)

        def on_final(t: str) -> None:
            nonlocal assistant_bubble
            if assistant_bubble is None:
                assistant_bubble = self._append_message("", kind="assistant")
            assistant_bubble.set_text(t if isinstance(t, str) else str(t))
            QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
            QtCore.QTimer.singleShot(0, self._update_bubble_widths)

        worker.signals.assistant_partial.connect(on_partial)
        worker.signals.assistant_final.connect(on_final)
        worker.signals.busy.connect(self._signals.set_busy.emit)
        worker.signals.error.connect(lambda m: self._signals.append_tool.emit(f"Error: {m}"))
        worker.signals.tokens.connect(self._signals.set_tokens.emit)
        worker.finished.connect(lambda: self._workers.discard(worker))
        worker.finished.connect(self._persist_current_chat)
        self._workers.add(worker)
        worker.start()

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
        try:
            metas = self._session.list_submodels()
        except Exception:
            metas = []

        # Tree structure: Main agent (expand/collapse like a folder) -> submodels.
        self._agents_tree.blockSignals(True)
        try:
            prev_expanded: bool | None = None
            if self._agents_tree.topLevelItemCount() > 0:
                try:
                    prev_expanded = bool(self._agents_tree.topLevelItem(0).isExpanded())
                except Exception:
                    prev_expanded = None

            cur_id: str | None = None
            it = self._agents_tree.currentItem()
            if it is not None:
                data = it.data(0, QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(data, dict) and data.get("kind") == "submodel":
                    sid = data.get("id")
                    cur_id = str(sid) if isinstance(sid, str) and sid else None

            self._agents_tree.clear()

            # Main "folder"
            main_label = "Main"
            try:
                mm = str(getattr(self._session.cfg, "model", "") or "")
                if mm:
                    main_label += f"\n{mm}"
            except Exception:
                pass
            main_item = QtWidgets.QTreeWidgetItem(["", main_label])
            main_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "main"})
            self._agents_tree.addTopLevelItem(main_item)

            for m in metas:
                sid = str(m.get("id") or "")
                title = str(m.get("title") or sid)
                model = str(m.get("model") or "")
                depth = int(m.get("depth", 0))
                state = str(m.get("state") or "")
                label = f"{title}"
                if model:
                    label += f"\n{model}"
                meta_bits = []
                if depth:
                    meta_bits.append(f"d{depth}")
                if state:
                    meta_bits.append(state)
                if meta_bits:
                    label += f"  ({', '.join(meta_bits)})"
                child = QtWidgets.QTreeWidgetItem(["", label])
                child.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "submodel", "id": sid})
                main_item.addChild(child)

            if prev_expanded is None:
                main_item.setExpanded(True)
            else:
                main_item.setExpanded(bool(prev_expanded))
            self._update_agents_tree_labels()

            # Restore selection if possible
            if isinstance(cur_id, str) and cur_id:
                for i in range(main_item.childCount()):
                    it2 = main_item.child(i)
                    d = it2.data(0, QtCore.Qt.ItemDataRole.UserRole)
                    if isinstance(d, dict) and d.get("kind") == "submodel" and d.get("id") == cur_id:
                        self._agents_tree.setCurrentItem(it2)
                        break
        finally:
            self._agents_tree.blockSignals(False)

    def _update_agents_tree_labels(self) -> None:
        """Keep the Main label updated (model name may change)."""

        try:
            top = self._agents_tree.topLevelItem(0)
        except Exception:
            top = None
        if top is None:
            return
        try:
            base = "Main"
            mm = str(getattr(self._session.cfg, "model", "") or "")
            if mm:
                base += f"\n{mm}"
        except Exception:
            base = "Main"
        top.setText(1, base)
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

    def _open_selected_submodel(self) -> None:
        sid = self._selected_submodel_id()
        if not sid:
            return
        sm = self._session.get_submodel(sid)
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
        if not sid:
            return
        self._session.close_submodel(sid)
        self._refresh_submodel_list()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        # Best-effort shutdown of background threads to avoid "QThread destroyed" warnings.
        try:
            self._agents_refresh_timer.stop()
        except Exception:
            pass
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
                        out = item.get("output", "")
                        out_s = out if isinstance(out, str) else str(out)
                        if len(out_s) > 1200:
                            out_s = out_s[:1200] + " …"
                        self.signals.tool.emit(f"[tool output] {out_s}")
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
            self.signals.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.signals.busy.emit(False)


def main() -> None:
    w = ChatWindow()
    raise SystemExit(w.exec())


if __name__ == "__main__":
    main()
