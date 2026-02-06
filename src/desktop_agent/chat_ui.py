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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .chat_session import ChatConfig, ChatSession
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


@dataclass(frozen=True)
class ChatUIConfig:
    width: int = 520
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

        self._session = self._make_session()
        self._show_tool_events = True
        self._hide_think = False
        self._workers: set[_Worker] = set()
        self._bubbles: list[Bubble] = []

        self._build_ui()
        self._apply_style()

        self.resize(self._cfg.width, self._cfg.height)
        self.setMinimumSize(420, 640)
        self.setWindowTitle("Desktop Agent • Chat")

        self._append_message(
            "Tip: You can ask it to use web search, read files, or even create+register new tools during the chat.",
            kind="tool",
        )

    def exec(self) -> int:
        self.show()
        return int(self._app.exec())

    # ---- session ----

    def _make_session(self) -> ChatSession:
        app_cfg = load_config()
        # Chat uses the same env + model defaults as the main app.
        model = os.environ.get("OPENAI_MODEL", app_cfg.openai_model or DEFAULT_MODEL)
        ccfg = ChatConfig(
            model=model,
            enable_web_search=True,
            web_search_context_size="medium",
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

        title = QtWidgets.QLabel("Chat")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel("Tools + Web Search + Self-Extending")
        subtitle.setObjectName("subtitle")
        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(2)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hl.addLayout(title_box)
        hl.addStretch(1)

        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.setObjectName("model_combo")
        self._model_combo.addItems(list(SUPPORTED_MODELS))
        # Try to set current model if present.
        idx = self._model_combo.findText(self._session.cfg.model)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        hl.addWidget(self._model_combo)

        self._btn_system = QtWidgets.QPushButton("System")
        self._btn_system.setObjectName("btn_ghost")
        self._btn_system.clicked.connect(self._open_system_dialog)
        hl.addWidget(self._btn_system)

        self._btn_controls = QtWidgets.QPushButton("Controls")
        self._btn_controls.setObjectName("btn_ghost")
        self._btn_controls.clicked.connect(self._open_controls_dialog)
        hl.addWidget(self._btn_controls)

        outer.addWidget(header)

        # Chat area
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
        outer.addWidget(self._scroll, 1)

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
        outer.addWidget(composer)

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

    def _append_message(self, text: str, *, kind: str) -> None:
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

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
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

        gl.addRow("Temperature", temp)
        gl.addRow("Top-p", top_p)
        gl.addRow("Max output tokens", mot)
        gl.addRow("Max tool calls", mtc)

        def apply_gen() -> None:
            self._session.cfg.temperature = float(temp.value())
            self._session.cfg.top_p = float(top_p.value())
            self._session.cfg.max_output_tokens = int(mot.value())
            self._session.cfg.max_tool_calls = int(mtc.value())
            self._append_message("Generation settings updated.", kind="tool")

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

        chk_setsys = QtWidgets.QCheckBox("Allow model set_system_prompt")
        chk_setsys.setChecked(bool(self._session.cfg.allow_model_set_system_prompt))

        chk_propose = QtWidgets.QCheckBox("Allow model propose tools")
        chk_propose.setChecked(bool(self._session.cfg.allow_model_propose_tools))

        chk_create = QtWidgets.QCheckBox("Allow model create+register tools")
        chk_create.setChecked(bool(self._session.cfg.allow_model_create_tools))

        tl.addRow(chk_web)
        tl.addRow("Search context size", ctx)
        tl.addRow(chk_read)
        tl.addRow(chk_setsys)
        tl.addRow(chk_propose)
        tl.addRow(chk_create)

        def apply_tools() -> None:
            self._session.cfg.enable_web_search = bool(chk_web.isChecked())
            self._session.cfg.web_search_context_size = str(ctx.currentText())
            self._session.cfg.allow_read_file = bool(chk_read.isChecked())
            self._session.cfg.allow_model_set_system_prompt = bool(chk_setsys.isChecked())
            self._session.cfg.allow_model_propose_tools = bool(chk_propose.isChecked())
            self._session.cfg.allow_model_create_tools = bool(chk_create.isChecked())
            self._append_message("Tool settings updated.", kind="tool")

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
        self._signals.append_user.emit(text)

        worker = _Worker(session=self._session, text=text, show_tool_events=self._show_tool_events, parent=self)
        worker.signals.tool.connect(self._signals.append_tool.emit)
        worker.signals.assistant.connect(self._signals.append_assistant.emit)
        worker.signals.busy.connect(self._signals.set_busy.emit)
        worker.signals.error.connect(lambda m: self._signals.append_tool.emit(f"Error: {m}"))
        worker.finished.connect(lambda: self._workers.discard(worker))
        self._workers.add(worker)
        worker.start()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        # Best-effort shutdown of background threads to avoid "QThread destroyed" warnings.
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
        super().closeEvent(event)


class _WorkerSignals(QtCore.QObject):
    assistant = QtCore.Signal(str)
    tool = QtCore.Signal(str)
    error = QtCore.Signal(str)
    busy = QtCore.Signal(bool)


class _Worker(QtCore.QThread):
    def __init__(self, *, session: ChatSession, text: str, show_tool_events: bool, parent: Optional[QtCore.QObject]) -> None:
        super().__init__(parent)
        self.session = session
        self.text = text
        self.show_tool_events = show_tool_events
        self.signals = _WorkerSignals()

    def run(self) -> None:  # type: ignore[override]
        self.signals.busy.emit(True)
        try:
            if self.isInterruptionRequested():
                return
            delta = self.session.send(self.text)
            if self.isInterruptionRequested():
                return
            if self.show_tool_events:
                for item in delta.new_items:
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
                        # Emit a short clickable-looking list.
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
            out = delta.assistant_text.strip()
            if self.session.cfg.hide_think:
                out = _strip_think(out)
            self.signals.assistant.emit(out)
        except Exception as e:
            self.signals.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.signals.busy.emit(False)


def main() -> None:
    w = ChatWindow()
    raise SystemExit(w.exec())


if __name__ == "__main__":
    main()
