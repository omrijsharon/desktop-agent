from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .automated_calibration_config import build_analysis_system_prompt, load_run_config, save_run_config
from .chat_session import ChatConfig, ChatSession
from .config import DEFAULT_MODEL, load_config
from .chat_ui import Bubble, _strip_think  # reuse bubble rendering
from .tools import list_saved_analysis_tool_names


JsonDict = dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class _Signals(QtCore.QObject):
    tool = QtCore.Signal(str)
    assistant_partial = QtCore.Signal(str)
    assistant_final = QtCore.Signal(str)
    busy = QtCore.Signal(bool)
    tokens = QtCore.Signal(str)


class _Worker(QtCore.QThread):
    def __init__(self, *, session: ChatSession, text: str, hide_think: bool, parent: Optional[QtCore.QObject]) -> None:
        super().__init__(parent)
        self.session = session
        self.text = text
        self.hide_think = hide_think
        self.signals = _Signals()

    def run(self) -> None:  # type: ignore[override]
        self.signals.busy.emit(True)
        try:
            full_raw = ""
            last_emit = 0.0
            last_tok = 0.0
            prompt_tok_est = self.session.estimate_prompt_tokens(user_text=self.text)
            max_ctx = int(getattr(self.session.cfg, "context_window_tokens", 0) or 0)

            new_items: list[JsonDict] = []
            try:
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
                        if now - last_tok >= 0.20:
                            out_tok_est = self.session.estimate_tokens(full_raw)
                            used_est = int(prompt_tok_est + out_tok_est)
                            if max_ctx > 0:
                                pct = (used_est / max_ctx) * 100.0
                                self.signals.tokens.emit(f"~{used_est:,}/{max_ctx:,} tok ({pct:.1f}%)")
                            else:
                                self.signals.tokens.emit(f"~{used_est:,} tok")
                            last_tok = now
                    elif et == "error":
                        err = ev.get("error")
                        err_s = str(err) if err is not None else "unknown error"
                        self.signals.assistant_final.emit(f"[error] {err_s}")
                        return
                    elif et == "turn_done":
                        ni = ev.get("new_items")
                        if isinstance(ni, list):
                            new_items = [x for x in ni if isinstance(x, dict)]
            except Exception as e:  # noqa: BLE001
                # Last-resort guard: don't crash the QThread override (prints scary tracebacks).
                self.signals.assistant_final.emit(f"[error] {type(e).__name__}: {e}")
                return

            # Emit tool events after completion (keeps UI simpler).
            for item in new_items:
                t = item.get("type")
                if t == "function_call":
                    name = item.get("name", "")
                    args = item.get("arguments", "")
                    self.signals.tool.emit(f"[tool call] {name}({args})")
                elif t == "function_call_output":
                    out = item.get("output", "")
                    try:
                        if isinstance(out, str) and out.strip().startswith("{"):
                            d = json.loads(out)
                            if isinstance(d, dict):
                                imgs = d.get("image_paths")
                                if isinstance(imgs, list):
                                    for p in imgs[:8]:
                                        if isinstance(p, str) and p.strip():
                                            self.signals.tool.emit(f"[[image:{p.strip()}]]")
                    except Exception:
                        pass
                    out_s = out if isinstance(out, str) else str(out)
                    if len(out_s) > 1200:
                        out_s = out_s[:1200] + " …"
                    self.signals.tool.emit(f"[tool output] {out_s}")

            out = (_strip_think(full_raw) if self.hide_think else full_raw).strip()
            self.signals.assistant_final.emit(out)
            self.signals.tokens.emit(self.session.usage_ratio_text())
        finally:
            self.signals.busy.emit(False)


class CalibrationAnalysisWindow(QtWidgets.QMainWindow):
    def __init__(self, *, run_config_path: Path) -> None:
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        super().__init__()
        self._run_config_path = Path(run_config_path)
        self._cfg = load_run_config(self._run_config_path)

        self._session: ChatSession | None = None
        self._workers: set[_Worker] = set()
        self._bubbles: list[Bubble] = []
        self._hide_think = True

        self._build_ui()
        self._apply_style()

        self.resize(1420, 860)
        self.setWindowTitle("Calibration Analysis")
        self._tokens.setText("")

        # Seed file list from config, else from local logs folder if present.
        self._load_files_from_config()

    def exec(self) -> int:
        self.show()
        return int(self._app.exec())

    # ---- session ----

    def _make_session(self) -> ChatSession:
        app_cfg = load_config()
        api_key = app_cfg.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        base_dir = Path(self._cfg.analysis.analysis_base_dir).resolve()
        saved_tools = list_saved_analysis_tool_names(scripts_root=_repo_root() / "ui" / "automated_calibration" / "analysis_tools")
        ccfg = ChatConfig(
            model=(self._cfg.analysis.model or DEFAULT_MODEL),
            enable_web_search=True,
            web_search_context_size="medium",
            enable_file_search=False,
            tool_base_dir=base_dir,
            allow_read_file=False,
            allow_write_files=False,
            allow_model_set_system_prompt=False,
            allow_model_propose_tools=False,
            allow_model_create_tools=True,
            allow_python_sandbox=True,
            python_sandbox_timeout_s=20.0,
            allow_model_create_analysis_tools=True,
            hide_think=True,
            allow_submodels=False,
        )
        s = ChatSession(api_key=api_key, config=ccfg)
        s.set_system_prompt(build_analysis_system_prompt(cfg=self._cfg.analysis, saved_analysis_tools=saved_tools))
        return s

    def _ensure_session(self) -> ChatSession:
        if self._session is None:
            self._session = self._make_session()
            try:
                self._tokens.setText(self._session.usage_ratio_text())
            except Exception:
                pass
        return self._session

    # ---- UI ----

    def _apply_style(self) -> None:
        # Reuse the main chat style for now.
        qss_path = _repo_root() / "ui" / "chat" / "style.qss"
        try:
            self._app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        font = QtGui.QFont()
        font.setFamilies(["SF Pro Display", "Segoe UI Variable", "Segoe UI", "Inter", "Arial"])
        font.setPointSize(10)
        self._app.setFont(font)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        outer = QtWidgets.QHBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # Left: file list
        left = QtWidgets.QFrame()
        left.setObjectName("panel")
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(12, 12, 12, 12)
        ll.setSpacing(8)

        lbl = QtWidgets.QLabel("Files to analyze")
        lbl.setObjectName("subtitle")
        ll.addWidget(lbl)

        self._file_list = QtWidgets.QListWidget()
        self._file_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        ll.addWidget(self._file_list, 1)

        row = QtWidgets.QHBoxLayout()
        self._btn_add_file = QtWidgets.QPushButton("Add")
        self._btn_add_file.setObjectName("btn_ghost")
        self._btn_add_folder = QtWidgets.QPushButton("Add folder")
        self._btn_add_folder.setObjectName("btn_ghost")
        self._btn_load_logs = QtWidgets.QPushButton("Load logs dir")
        self._btn_load_logs.setObjectName("btn_ghost")
        self._btn_rm_file = QtWidgets.QPushButton("Remove")
        self._btn_rm_file.setObjectName("btn_ghost")
        row.addWidget(self._btn_add_file)
        row.addWidget(self._btn_add_folder)
        row.addWidget(self._btn_load_logs)
        row.addWidget(self._btn_rm_file)
        ll.addLayout(row)

        self._btn_send_analysis = QtWidgets.QPushButton("Send analysis prompt")
        self._btn_send_analysis.setObjectName("btn_primary")
        ll.addWidget(self._btn_send_analysis)

        splitter.addWidget(left)

        # Center: chat
        mid = QtWidgets.QFrame()
        mid.setObjectName("panel")
        ml = QtWidgets.QVBoxLayout(mid)
        ml.setContentsMargins(12, 12, 12, 12)
        ml.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Analysis")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch(1)
        self._btn_config = QtWidgets.QPushButton("Config")
        self._btn_config.setObjectName("btn_ghost")
        header.addWidget(self._btn_config)
        ml.addLayout(header)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        ml.addWidget(self._scroll, 1)

        chat = QtWidgets.QWidget()
        self._chat_layout = QtWidgets.QVBoxLayout(chat)
        self._chat_layout.setContentsMargins(0, 0, 0, 0)
        self._chat_layout.setSpacing(10)
        self._chat_layout.addStretch(1)
        self._scroll.setWidget(chat)

        bottom = QtWidgets.QHBoxLayout()
        self._input = QtWidgets.QTextEdit()
        self._input.setPlaceholderText("Message… (Shift+Enter for newline)")
        self._input.setFixedHeight(92)
        self._btn_send = QtWidgets.QPushButton("Send")
        self._btn_send.setObjectName("btn_primary")
        bottom.addWidget(self._input, 1)
        bottom.addWidget(self._btn_send)
        ml.addLayout(bottom)

        self._tokens = QtWidgets.QLabel("")
        self._tokens.setObjectName("subtitle")
        self._tokens.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        ml.addWidget(self._tokens)

        splitter.addWidget(mid)

        splitter.setSizes([360, 1060])

        # wiring
        self._btn_send.clicked.connect(self._on_send)
        self._btn_send_analysis.clicked.connect(self._send_analysis_prompt)
        self._btn_add_file.clicked.connect(self._add_files)
        self._btn_add_folder.clicked.connect(self._add_folder)
        self._btn_load_logs.clicked.connect(self._load_logs_dir)
        self._btn_rm_file.clicked.connect(self._remove_files)
        self._btn_config.clicked.connect(self._open_config_dialog)

        self._input.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if obj is self._input and event.type() == QtCore.QEvent.Type.KeyPress:
            e = event  # type: ignore[assignment]
            if isinstance(e, QtGui.QKeyEvent) and e.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                if e.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    return False
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    # ---- files ----

    def _load_files_from_config(self) -> None:
        self._file_list.clear()
        for f in self._cfg.analysis.files or []:
            self._file_list.addItem(f)

    def _normalize_path(self, p: str) -> str:
        try:
            base = Path(self._cfg.analysis.analysis_base_dir).resolve()
            rp = Path(p).resolve()
            return rp.relative_to(base).as_posix()
        except Exception:
            return str(p)

    def _add_files(self) -> None:
        start = str(Path(self._cfg.local.logs_dir).resolve()) if self._cfg.local.logs_dir else str(_repo_root())
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add log files", start, "Log files (*.jsonl *.json *.txt);;All files (*)")
        for p in paths:
            np = self._normalize_path(p)
            if np and not self._has_file(np):
                self._file_list.addItem(np)

    def _add_folder(self) -> None:
        start = str(Path(self._cfg.local.logs_dir).resolve()) if self._cfg.local.logs_dir else str(_repo_root())
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Add folder", start)
        np = self._normalize_path(p)
        if np and not self._has_file(np):
            self._file_list.addItem(np)

    def _load_logs_dir(self) -> None:
        logs_dir = str(Path(self._cfg.local.logs_dir).resolve())
        np = self._normalize_path(logs_dir)
        if np and not self._has_file(np):
            self._file_list.addItem(np)

    def _has_file(self, p: str) -> bool:
        for i in range(self._file_list.count()):
            if self._file_list.item(i).text() == p:
                return True
        return False

    def _remove_files(self) -> None:
        for it in self._file_list.selectedItems():
            self._file_list.takeItem(self._file_list.row(it))

    # ---- config dialog ----

    def _open_config_dialog(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Configuration")
        dlg.setObjectName("dialog")
        dlg.resize(880, 720)

        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("tabs")
        lay.addWidget(tabs, 1)

        # --- Prompt tab ---
        ptab = QtWidgets.QWidget()
        pl = QtWidgets.QFormLayout(ptab)
        pl.setContentsMargins(12, 12, 12, 12)
        pl.setSpacing(10)

        system = QtWidgets.QTextEdit()
        system.setPlainText(self._cfg.analysis.system_prompt)
        system.setMinimumHeight(160)
        pl.addRow("System prompt", system)

        debug = QtWidgets.QTextEdit()
        debug.setPlainText("\n".join(self._cfg.analysis.debug_mapping or []))
        debug.setMinimumHeight(120)
        pl.addRow("Debug mapping", debug)

        extra = QtWidgets.QTextEdit()
        extra.setPlainText(self._cfg.analysis.extra_context or "")
        extra.setMinimumHeight(140)
        pl.addRow("Extra context", extra)

        tabs.addTab(ptab, "Prompt")

        # --- Betaflight snippet tab ---
        btab = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(btab)
        bl.setContentsMargins(12, 12, 12, 12)
        bl.setSpacing(8)
        bl.addWidget(QtWidgets.QLabel("Paste the relevant Betaflight code snippet(s) here."))
        bf = QtWidgets.QTextEdit()
        bf.setPlainText(self._cfg.analysis.betaflight_snippet or "")
        bl.addWidget(bf, 1)
        tabs.addTab(btab, "Betaflight")

        # --- Control params tab ---
        ctab = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(ctab)
        cl.setContentsMargins(12, 12, 12, 12)
        cl.setSpacing(8)
        cl.addWidget(QtWidgets.QLabel("Control parameters / gains (JSON object). The model can recommend updates."))
        ctrl = QtWidgets.QTextEdit()
        ctrl.setPlainText(json.dumps(self._cfg.analysis.control_params or {}, indent=2, ensure_ascii=False))
        cl.addWidget(ctrl, 1)
        tabs.addTab(ctab, "Params")

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setObjectName("btn_ghost")
        btn_apply = QtWidgets.QPushButton("Apply + save")
        btn_apply.setObjectName("btn_primary")
        row.addWidget(btn_cancel)
        row.addWidget(btn_apply)
        lay.addLayout(row)

        btn_cancel.clicked.connect(dlg.reject)

        def apply() -> None:
            sys_prompt = system.toPlainText()
            debug_lines = [ln.strip() for ln in debug.toPlainText().splitlines() if ln.strip()]
            extra_txt = extra.toPlainText()
            bf_txt = bf.toPlainText()
            try:
                params = json.loads(ctrl.toPlainText() or "{}")
                if not isinstance(params, dict):
                    raise ValueError("control_params must be a JSON object")
            except Exception as e:
                raise RuntimeError(f"Invalid control_params JSON: {e}") from e

            raw = json.loads(Path(self._run_config_path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise RuntimeError("run_config.json must be an object")
            analysis = raw.get("analysis") or {}
            if not isinstance(analysis, dict):
                analysis = {}
            analysis["system_prompt"] = sys_prompt
            analysis["debug_mapping"] = debug_lines
            analysis["extra_context"] = extra_txt
            analysis["betaflight_snippet"] = bf_txt
            analysis["control_params"] = params
            # Keep files list in sync too
            analysis["files"] = [self._file_list.item(i).text() for i in range(self._file_list.count())]
            raw["analysis"] = analysis
            save_run_config(self._run_config_path, raw)

            # Reload strongly typed config
            self._cfg = load_run_config(self._run_config_path)
            if self._session is not None:
                saved_tools = list_saved_analysis_tool_names(scripts_root=_repo_root() / "ui" / "automated_calibration" / "analysis_tools")
                self._session.set_system_prompt(build_analysis_system_prompt(cfg=self._cfg.analysis, saved_analysis_tools=saved_tools))
            self._tool_message("Saved run_config.json and updated system prompt.")
            dlg.accept()

        btn_apply.clicked.connect(apply)
        dlg.exec()

    # ---- chat ----

    def _tool_message(self, text: str) -> None:
        self._append_message(text, kind="tool")

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
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, container)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
        return bubble

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_busy(self, busy: bool) -> None:
        self._btn_send.setEnabled(not busy)
        self._input.setEnabled(not busy)
        self._btn_send_analysis.setEnabled(not busy)
        self._btn_config.setEnabled(not busy)

    def _on_send(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        session = self._ensure_session()
        self._input.clear()
        self._append_message(text, kind="user")
        assistant: Bubble | None = None

        worker = _Worker(session=session, text=text, hide_think=self._hide_think, parent=self)
        worker.signals.busy.connect(self._set_busy)
        worker.signals.tokens.connect(self._tokens.setText)
        worker.signals.tool.connect(lambda t: self._append_message(t, kind="tool"))

        def on_partial(t: str) -> None:
            nonlocal assistant
            if assistant is None:
                assistant = self._append_message("", kind="assistant")
            assistant.set_streaming_text(t)
            QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

        def on_final(t: str) -> None:
            nonlocal assistant
            if assistant is None:
                assistant = self._append_message("", kind="assistant")
            assistant.set_text(t)
            QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

        worker.signals.assistant_partial.connect(on_partial)
        worker.signals.assistant_final.connect(on_final)
        worker.finished.connect(lambda: self._workers.discard(worker))
        self._workers.add(worker)
        worker.start()

    def _send_analysis_prompt(self) -> None:
        # Send only file paths; analysis instructions live in the system prompt.
        files = [self._file_list.item(i).text() for i in range(self._file_list.count())]
        if not files:
            self._tool_message("No files selected.")
            return

        base_dir = Path(self._cfg.analysis.analysis_base_dir).resolve()
        rels: list[str] = []
        for f in files:
            try:
                fp = Path(f)
                rp = (base_dir / fp).resolve() if not fp.is_absolute() else fp.resolve()
                rels.append(rp.relative_to(base_dir).as_posix())
            except Exception:
                # Keep as-is; python_sandbox can only copy within base_dir.
                rels.append(f)

        msg = "Files to analyze (repo-relative if possible):\n" + "\n".join(f"- {p}" for p in rels) + "\n"
        self._input.setPlainText(msg)
        self._on_send()


def main() -> None:
    # Default run_config.json under ui/automated_calibration/
    p = _repo_root() / "ui" / "automated_calibration" / "run_config.json"
    w = CalibrationAnalysisWindow(run_config_path=p)
    raise SystemExit(w.exec())


if __name__ == "__main__":
    main()
