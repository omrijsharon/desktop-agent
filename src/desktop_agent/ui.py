"""desktop_agent.ui

PySide6 UI for MVP v0.

This replaces the earlier Tkinter UI to provide a more modern native feel on
Windows.

Public API (used by `desktop_agent.main`):
- post_message(text)
- post_status(state=..., high_level=...)
- set_controls_enabled(running=...)
- show()
- exec()
- on_close(callback)

Threading:
- UI runs on the main thread (Qt event loop)
- Agent/planner runs on a worker thread
- UI updates are queued via Qt signals (thread-safe)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import html

from PySide6 import QtCore, QtGui, QtWidgets

from .config import DEFAULT_MODEL, SUPPORTED_MODELS


@dataclass(frozen=True)
class UIConfig:
    width: int = 420
    height: int = 720
    always_on_top: bool = False
    dock_right: bool = True
    snap_threshold_px: int = 24


class _Signals(QtCore.QObject):
    message = QtCore.Signal(str)
    status = QtCore.Signal(str, str)
    controls = QtCore.Signal(bool)


class DesktopAgentUI(QtWidgets.QMainWindow):
    def __init__(
        self,
        *,
        config: UIConfig | None = None,
        on_run: Callable[[str, bool, str], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        on_approve: Callable[[], None] | None = None,
    ) -> None:
        # Ensure a QApplication exists *before* constructing any QWidget.
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        super().__init__()

        self._cfg = config or UIConfig()
        self._on_run = on_run
        self._on_stop = on_stop
        self._on_approve = on_approve

        self._signals = _Signals()

        self._signals.message.connect(self._append_message)
        self._signals.status.connect(self._set_status)
        self._signals.controls.connect(self._set_running)

        self._close_cb: Optional[Callable[[], None]] = None

        self._theme: str = "light"  # "light" | "dark"

        self._build_ui()
        self._apply_style()

        # Keep the window in front, but try not to steal focus.
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        self.setMinimumSize(360, 420)
        self.resize(self._cfg.width, self._cfg.height)

        if self._cfg.dock_right:
            self._dock_right_initial()

        self._set_running(False)

    # ---- public API (thread-safe) ----

    def post_message(self, text: str) -> None:
        self._signals.message.emit(str(text))

    def post_status(self, *, state: str, high_level: str = "") -> None:
        self._signals.status.emit(str(state), str(high_level))

    def set_controls_enabled(self, *, running: bool) -> None:
        self._signals.controls.emit(bool(running))

    def on_close(self, cb: Callable[[], None]) -> None:
        self._close_cb = cb

    def exec(self) -> int:
        return int(self._app.exec())

    def show(self) -> None:  # type: ignore[override]
        # Show in front without activating (best-effort; varies by Windows policy).
        super().show()
        try:
            self.raise_()
            self.clearFocus()
        except Exception:
            pass

    # ---- UI ----

    def _build_ui(self) -> None:
        self.setWindowTitle("Desktop Agent")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # Header row
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(8)

        # (No big title; keep the header compact.)
        header.addStretch(1)

        self._mode_btn = QtWidgets.QPushButton("Dark mode")
        self._mode_btn.setObjectName("SecondaryButton")
        self._mode_btn.setToolTip("Toggle dark/light mode")
        self._mode_btn.clicked.connect(self._toggle_theme)

        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.setObjectName("ModelCombo")
        self._model_combo.addItems(list(SUPPORTED_MODELS))
        self._model_combo.setEditable(False)
        self._model_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self._model_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        idx = max(0, self._model_combo.findText(DEFAULT_MODEL))
        self._model_combo.setCurrentIndex(idx)
        self._model_combo.currentIndexChanged.connect(lambda _i: self._model_combo.update())

        self._step_cb = QtWidgets.QCheckBox("Step")
        self._step_cb.setObjectName("StepCheck")

        self._approve_btn = QtWidgets.QPushButton("Approve")
        self._approve_btn.setObjectName("SecondaryButton")
        self._approve_btn.clicked.connect(self._handle_approve)

        self._stop_btn = QtWidgets.QPushButton("Stop")
        self._stop_btn.setObjectName("DangerButton")
        self._stop_btn.clicked.connect(self._handle_stop)

        header.addWidget(self._mode_btn)
        header.addWidget(self._model_combo)
        header.addWidget(self._step_cb)
        header.addWidget(self._approve_btn)
        header.addWidget(self._stop_btn)

        root.addLayout(header)

        # Status row
        status_row = QtWidgets.QHBoxLayout()
        status_row.setSpacing(8)

        self._status_lbl = QtWidgets.QLabel("IDLE")
        self._status_lbl.setObjectName("Status")
        self._high_level_lbl = QtWidgets.QLabel("")
        self._high_level_lbl.setObjectName("HighLevel")
        self._high_level_lbl.setWordWrap(True)

        status_row.addWidget(self._status_lbl)
        status_row.addWidget(self._high_level_lbl, 1)
        root.addLayout(status_row)

        # Chat area
        self._chat = QtWidgets.QTextBrowser()
        self._chat.setObjectName("Chat")
        self._chat.setOpenExternalLinks(True)
        self._chat.setReadOnly(True)

        root.addWidget(self._chat, 1)

        # Input row
        input_row = QtWidgets.QHBoxLayout()
        input_row.setSpacing(8)

        self._entry = QtWidgets.QLineEdit()
        self._entry.setObjectName("GoalEntry")
        self._entry.setPlaceholderText("Type a goal…")
        self._entry.returnPressed.connect(self._handle_run)

        self._run_btn = QtWidgets.QPushButton("")
        self._run_btn.setAccessibleName("Run")
        self._run_btn.setToolTip("Run")
        self._run_btn.setObjectName("PrimaryButton")
        self._run_btn.setIcon(self._paper_plane_icon(color=QtGui.QColor("white")))
        self._run_btn.setIconSize(QtCore.QSize(18, 18))
        self._run_btn.setFixedWidth(44)
        self._run_btn.clicked.connect(self._handle_run)

        input_row.addWidget(self._entry, 1)
        input_row.addWidget(self._run_btn)
        root.addLayout(input_row)

    def _toggle_theme(self) -> None:
        self._theme = "dark" if self._theme == "light" else "light"
        self._mode_btn.setText("Bright mode" if self._theme == "dark" else "Dark mode")
        self._apply_style()

    def _apply_style(self) -> None:
        # Runtime theme switching via Qt stylesheet.
        if self._theme == "dark":
            self.setStyleSheet(
                """
                QMainWindow { background: #0B0B0F; }
                QLabel { color: #F2F2F7; font-family: 'Segoe UI'; font-size: 10pt; }
                QLabel#Status { color: #A1A1AA; font-weight: 600; }
                QLabel#HighLevel { color: #F2F2F7; }

                QTextBrowser#Chat {
                    background: #14141A;
                    color: #F2F2F7;
                    border: 1px solid #2A2A33;
                    border-radius: 12px;
                    padding: 10px;
                    font-family: 'Segoe UI';
                    font-size: 10pt;
                }
                QLineEdit#GoalEntry {
                    background: #14141A;
                    color: #F2F2F7;
                    selection-background-color: #0A84FF;
                    selection-color: #FFFFFF;
                    border: 1px solid #2A2A33;
                    border-radius: 12px;
                    padding: 10px 12px;
                    font-family: 'Segoe UI';
                    font-size: 10pt;
                }
                QLineEdit#GoalEntry::placeholder { color: #A1A1AA; }

                QComboBox#ModelCombo {
                    background: #14141A;
                    color: #F2F2F7;
                    border: 1px solid #2A2A33;
                    border-radius: 12px;
                    padding: 6px 10px;
                    min-width: 150px;
                    font-family: 'Segoe UI';
                    font-size: 10pt;
                }
                QComboBox#ModelCombo::drop-down { border: 0px; width: 24px; }
                QComboBox QAbstractItemView {
                    background: #14141A;
                    color: #F2F2F7;
                    selection-background-color: #26324A;
                    selection-color: #F2F2F7;
                    outline: 0;
                }

                QPushButton {
                    border-radius: 12px;
                    padding: 8px 12px;
                    font-family: 'Segoe UI';
                    font-size: 10pt;
                }
                QPushButton#PrimaryButton { background: #0A84FF; color: white; border: 0px; padding: 8px; }
                QPushButton#PrimaryButton:hover { background: #2F95FF; }

                QPushButton#SecondaryButton { background: #1E1E25; color: #F2F2F7; border: 1px solid #2A2A33; }
                QPushButton#SecondaryButton:hover { background: #24242D; }

                QPushButton#DangerButton { background: #FF3B30; color: white; border: 0px; }
                QPushButton#DangerButton:hover { background: #FF5E57; }

                QCheckBox#StepCheck { font-family: 'Segoe UI'; font-size: 10pt; color: #F2F2F7; }
                """
            )
            return

        # light theme (default)
        self.setStyleSheet(
            """
            QMainWindow { background: #F5F5F7; }
            QLabel { color: #1C1C1E; font-family: 'Segoe UI'; font-size: 10pt; }
            QLabel#Status { color: #6E6E73; font-weight: 600; }
            QLabel#HighLevel { color: #1C1C1E; }

            QTextBrowser#Chat {
                background: #FFFFFF;
                color: #1C1C1E;
                border: 1px solid #E5E5EA;
                border-radius: 12px;
                padding: 10px;
                font-family: 'Segoe UI';
                font-size: 10pt;
            }
            QLineEdit#GoalEntry {
                background: #FFFFFF;
                color: #1C1C1E;
                selection-background-color: #0A84FF;
                selection-color: #FFFFFF;
                border: 1px solid #E5E5EA;
                border-radius: 12px;
                padding: 10px 12px;
                font-family: 'Segoe UI';
                font-size: 10pt;
            }
            QLineEdit#GoalEntry::placeholder { color: #6E6E73; }

            QComboBox#ModelCombo {
                background: #FFFFFF;
                color: #1C1C1E;
                border: 1px solid #E5E5EA;
                border-radius: 12px;
                padding: 6px 10px;
                min-width: 150px;
                font-family: 'Segoe UI';
                font-size: 10pt;
            }
            QComboBox#ModelCombo::drop-down { border: 0px; width: 24px; }
            QComboBox QAbstractItemView {
                background: #FFFFFF;
                color: #1C1C1E;
                selection-background-color: #E8F0FF;
                selection-color: #1C1C1E;
                outline: 0;
            }

            QPushButton {
                border-radius: 12px;
                padding: 8px 12px;
                font-family: 'Segoe UI';
                font-size: 10pt;
            }
            QPushButton#PrimaryButton { background: #0A84FF; color: white; border: 0px; padding: 8px; }
            QPushButton#PrimaryButton:hover { background: #2F95FF; }

            QPushButton#SecondaryButton { background: #EFEFF4; color: #1C1C1E; border: 1px solid #E5E5EA; }
            QPushButton#SecondaryButton:hover { background: #E5E5EA; }

            QPushButton#DangerButton { background: #FF3B30; color: white; border: 0px; }
            QPushButton#DangerButton:hover { background: #FF5E57; }

            QCheckBox#StepCheck { font-family: 'Segoe UI'; font-size: 10pt; color: #1C1C1E; }
            """
        )

    # ---- behavior ----

    def _dock_right_initial(self) -> None:
        screen = self.screen() or self._app.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()

        # Keep the configured width, but occupy the full available height.
        w = self.width()
        h = geo.height()
        self.resize(w, h)

        x = geo.x() + max(0, geo.width() - w)
        y = geo.y()
        self.move(x, y)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        try:
            if self._close_cb:
                self._close_cb()
        finally:
            event.accept()

    # ---- handlers ----

    def _handle_run(self) -> None:
        goal = (self._entry.text() or "").strip()
        if not goal:
            return

        self._append_user(goal)
        self._entry.clear()

        model = self._model_combo.currentText().strip() or DEFAULT_MODEL
        step = bool(self._step_cb.isChecked())

        if self._on_run:
            self._on_run(goal, step, model)

    def _handle_stop(self) -> None:
        if self._on_stop:
            self._on_stop()

    def _handle_approve(self) -> None:
        if self._on_approve:
            self._on_approve()

    # ---- render helpers ----

    def _append_html(self, html: str) -> None:
        self._chat.append(html)
        sb = self._chat.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _append_user(self, text: str) -> None:
        t = html.escape(text)
        self._append_html(
            "<div style='margin:6px 0; text-align:right;'>"
            "<span style='display:inline-block; background:#0A84FF; color:white; padding:8px 10px; border-radius:12px; max-width:90%;'>"
            f"{t}"
            "</span></div>"
        )

    def _append_agent(self, text: str) -> None:
        t = html.escape(text)
        self._append_html(
            "<div style='margin:6px 0; text-align:left;'>"
            "<span style='display:inline-block; background:#EFEFF4; color:#1C1C1E; padding:8px 10px; border-radius:12px; max-width:90%;'>"
            f"{t}"
            "</span></div>"
        )

    def _append_system(self, text: str) -> None:
        t = html.escape(text)
        self._append_html(
            "<div style='margin:6px 0; text-align:left;'>"
            "<span style='display:inline-block; background:#E8F0FF; color:#1C1C1E; padding:6px 10px; border-radius:10px; font-size:9pt;'>"
            f"{t}"
            "</span></div>"
        )

    @QtCore.Slot(str)
    def _append_message(self, text: str) -> None:
        txt = str(text)
        if txt.startswith("[SYSTEM]"):
            self._append_system(txt[len("[SYSTEM]") :].lstrip())
        else:
            self._append_agent(txt)

    @QtCore.Slot(str, str)
    def _set_status(self, state: str, high_level: str) -> None:
        self._status_lbl.setText(state)
        self._high_level_lbl.setText(high_level)

    @QtCore.Slot(bool)
    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._entry.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._approve_btn.setEnabled(running)

    def _paper_plane_icon(self, *, color: QtGui.QColor) -> QtGui.QIcon:
        """Create a simple paper-plane glyph.

        Shape: a right-pointing triangle with a smaller triangle cut out near
        the base to suggest a fold.
        """

        size = 20
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.GlobalColor.transparent)

        p = QtGui.QPainter(pm)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            p.setPen(QtCore.Qt.PenStyle.NoPen)

            big = QtGui.QPainterPath()
            # Big triangle: base on left, point to the right
            big.moveTo(3, 4)
            big.lineTo(17, 10)
            big.lineTo(3, 16)
            big.closeSubpath()

            cut = QtGui.QPainterPath()
            # Cut-out triangle sitting on the base
            cut.moveTo(3, 7)
            cut.lineTo(9, 10)
            cut.lineTo(3, 13)
            cut.closeSubpath()

            shape = big.subtracted(cut)

            p.setBrush(QtGui.QBrush(color))
            p.drawPath(shape)
        finally:
            p.end()

        return QtGui.QIcon(pm)
