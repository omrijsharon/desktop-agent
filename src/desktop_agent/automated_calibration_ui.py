from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .automated_calibration_config import build_analysis_system_prompt, default_run_config_path, example_run_config_path, load_run_config
from .automated_calibration_ops import (
    archive_dir,
    connect_wifi_windows,
    ensure_empty_dir,
    has_internet,
    list_wifi_ssids_windows,
    remote_dir_listing,
    scp_pull_dir,
    ssh_popen,
    ssh_run,
)
from .automated_calibration_analysis_ui import CalibrationAnalysisWindow
from .chat_session import ChatConfig, ChatSession
from .config import DEFAULT_MODEL, load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_qss() -> str:
    p = _repo_root() / "ui" / "automated_calibration" / "style.qss"
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


class _Signals(QtCore.QObject):
    log = QtCore.Signal(str)
    busy = QtCore.Signal(bool)
    set_step = QtCore.Signal(str)


class CalibrationRunnerWindow(QtWidgets.QMainWindow):
    def __init__(self, *, config_path: Optional[Path] = None) -> None:
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        super().__init__()

        self._signals = _Signals()
        self._signals.log.connect(self._append_log)
        self._signals.busy.connect(self._set_busy)
        self._signals.set_step.connect(self._set_step)

        self._config_path = Path(config_path) if config_path else default_run_config_path()
        if not self._config_path.exists():
            self._config_path = example_run_config_path()
        self._cfg = load_run_config(self._config_path)
        self._webapp_proc = None
        self._webapp_reader: Optional[threading.Thread] = None
        self._analysis_windows: set[QtWidgets.QWidget] = set()

        self._build_ui()
        self._apply_style()

        self.resize(980, 760)
        self.setWindowTitle("Automated Calibration")
        self._log(f"Config: {self._config_path}")
        if str(self._config_path).endswith("run_config.example.json"):
            self._log("NOTE: Using example config. Create ui/automated_calibration/run_config.json with your SSIDs/passwords.")

    def exec(self) -> int:
        self.show()
        return int(self._app.exec())

    # ---- UI ----

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        outer.addWidget(panel, 1)

        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        title = QtWidgets.QLabel("Automated Calibration")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel("Pi AP → Webapp → Logs → Internet → LLM+Python analysis")
        subtitle.setObjectName("subtitle")
        lay.addWidget(title)
        lay.addWidget(subtitle)

        self._step = QtWidgets.QLabel("")
        self._step.setObjectName("subtitle")
        lay.addWidget(self._step)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        lay.addLayout(row)

        self._btn_connect_pi = QtWidgets.QPushButton("1) Connect Pi AP")
        self._btn_connect_pi.setObjectName("btn_primary")
        self._btn_start_webapp = QtWidgets.QPushButton("2) Start webapp")
        self._btn_start_webapp.setObjectName("btn_ghost")
        self._btn_done_flying = QtWidgets.QPushButton("3) Finished flying (pull logs)")
        self._btn_done_flying.setObjectName("btn_ghost")
        self._btn_back_inet = QtWidgets.QPushButton("4) Back to internet")
        self._btn_back_inet.setObjectName("btn_primary")
        self._btn_analyze = QtWidgets.QPushButton("5) Analyze logs")
        self._btn_analyze.setObjectName("btn_ghost")

        row.addWidget(self._btn_connect_pi, 1)
        row.addWidget(self._btn_start_webapp, 1)
        row.addWidget(self._btn_done_flying, 1)
        row.addWidget(self._btn_back_inet, 1)
        row.addWidget(self._btn_analyze, 1)

        self._log_view = QtWidgets.QTextEdit()
        self._log_view.setObjectName("log")
        self._log_view.setReadOnly(True)
        lay.addWidget(self._log_view, 1)

        self._btn_connect_pi.clicked.connect(self._on_connect_pi)
        self._btn_start_webapp.clicked.connect(self._on_start_webapp)
        self._btn_done_flying.clicked.connect(self._on_done_flying)
        self._btn_back_inet.clicked.connect(self._on_back_internet)
        self._btn_analyze.clicked.connect(self._on_analyze)

        self._set_step("Ready")

    def _apply_style(self) -> None:
        qss = _load_qss()
        if qss:
            self._app.setStyleSheet(qss)
        font = QtGui.QFont()
        font.setFamilies(["SF Pro Display", "Segoe UI Variable", "Segoe UI", "Inter", "Arial"])
        font.setPointSize(10)
        self._app.setFont(font)

    # ---- logging/state ----

    def _log(self, msg: str) -> None:
        self._signals.log.emit(msg)

    def _append_log(self, msg: str) -> None:
        self._log_view.append(msg)
        self._log_view.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def _set_step(self, msg: str) -> None:
        self._step.setText(msg)

    def _set_busy(self, busy: bool) -> None:
        for b in (
            self._btn_connect_pi,
            self._btn_start_webapp,
            self._btn_done_flying,
            self._btn_back_inet,
            self._btn_analyze,
        ):
            b.setEnabled(not busy)

    def _run_bg(self, step: str, fn) -> None:
        def go() -> None:
            self._signals.busy.emit(True)
            self._signals.set_step.emit(step)
            try:
                fn()
            except Exception as e:
                self._signals.log.emit(f"ERROR: {type(e).__name__}: {e}")
            finally:
                self._signals.busy.emit(False)
                self._signals.set_step.emit("Ready")

        threading.Thread(target=go, daemon=True).start()

    # ---- actions ----

    def _on_connect_pi(self) -> None:
        def work() -> None:
            self._log(f"Connecting Wi-Fi to Pi AP: {self._cfg.pi_ap.ssid!r}")
            r = connect_wifi_windows(ssid=self._cfg.pi_ap.ssid, password=self._cfg.pi_ap.password)
            self._log(r.stdout.strip() or "Connected (netsh).")
            if not r.ok:
                raise RuntimeError(r.stderr.strip() or "Failed to connect Wi-Fi")
            self._log("Tip: if mDNS fails, we will fall back to 10.42.0.1.")

        self._run_bg("Connecting to Pi AP…", work)

    def _pick_ssh_host(self) -> str:
        return self._cfg.pi_ssh.host or self._cfg.pi_ssh.fallback_host

    def _open_webapp_url(self) -> None:
        url = "http://10.42.0.1:5000/"
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
            return
        except Exception:
            pass
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass

    def _on_start_webapp(self) -> None:
        # Open the web UI immediately for a smooth workflow; the page will load once the
        # Pi webapp is up (or show a connection error if it's not reachable yet).
        try:
            self._open_webapp_url()
        except Exception:
            pass

        def work() -> None:
            if self._webapp_proc is not None and self._webapp_proc.poll() is None:
                self._log("Webapp already running.")
                return
            host = self._pick_ssh_host()
            self._log(f"Starting webapp via SSH: {self._cfg.pi_ssh.user}@{host}")
            proc = ssh_popen(
                user=self._cfg.pi_ssh.user,
                host=host,
                remote_cmd=self._cfg.remote.webapp_cmd,
                x11=bool(self._cfg.pi_ssh.use_x11),
                cwd=str(_repo_root()),
            )
            self._webapp_proc = proc

            def reader() -> None:
                try:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        self._signals.log.emit(f"[pi] {line.rstrip()}")
                except Exception:
                    pass

            self._webapp_reader = threading.Thread(target=reader, daemon=True)
            self._webapp_reader.start()
            self._log("Webapp started. Fly the drone, then click “Finished flying (pull logs)”.")

        self._run_bg("Starting webapp…", work)

    def _on_done_flying(self) -> None:
        def work() -> None:
            host = self._pick_ssh_host()
            user = self._cfg.pi_ssh.user

            # Stop the webapp SSH session (best-effort). It may terminate the webapp if it runs in foreground.
            if self._webapp_proc is not None and self._webapp_proc.poll() is None:
                try:
                    self._log("Stopping webapp SSH session…")
                    self._webapp_proc.terminate()
                except Exception:
                    pass

            self._log("Checking remote logs directory before scp…")
            ok_ls, entries, raw = remote_dir_listing(user=user, host=host, remote_dir=self._cfg.remote.logs_dir)
            if not ok_ls or not entries:
                # Try fallback host in case mDNS is flaky.
                fb = self._cfg.pi_ssh.fallback_host
                if fb and fb != host:
                    self._log(f"Remote listing empty/failed; retrying with fallback host: {fb}")
                    ok_ls, entries, raw = remote_dir_listing(user=user, host=fb, remote_dir=self._cfg.remote.logs_dir)
                    if ok_ls:
                        host = fb
                if not ok_ls or not entries:
                    self._log("ERROR: No logs found on the Pi (remote logs directory is empty or missing).")
                    raise RuntimeError("No logs found on the Pi")
            self._log(f"Remote logs entries found: {len(entries)} (showing up to 10)")
            for e in entries[:10]:
                self._log(f"  - {e}")

            local_logs = Path(self._cfg.local.logs_dir)
            archive_root = Path(self._cfg.local.archive_root)

            self._log(f"Archiving existing local logs (if any): {local_logs}")
            archived = archive_dir(local_logs, archive_root)
            if archived is not None:
                self._log(f"Archived to: {archived}")

            self._log("Clearing local logs dir…")
            ensure_empty_dir(local_logs)

            self._log("Pulling logs via scp…")
            r = scp_pull_dir(user=user, host=host, remote_dir=self._cfg.remote.logs_dir, local_parent=str(local_logs.parent))
            self._log(r.stdout.strip() or "scp completed.")
            if not r.ok:
                # try fallback host
                fb = self._cfg.pi_ssh.fallback_host
                if fb and fb != host:
                    self._log(f"scp failed; retrying with fallback host: {fb}")
                    r2 = scp_pull_dir(user=user, host=fb, remote_dir=self._cfg.remote.logs_dir, local_parent=str(local_logs.parent))
                    self._log(r2.stdout.strip() or "scp completed.")
                    if not r2.ok:
                        raise RuntimeError(r2.stderr.strip() or "scp failed")
                else:
                    raise RuntimeError(r.stderr.strip() or "scp failed")

            if not local_logs.exists() or not any(local_logs.rglob("*.jsonl")):
                raise RuntimeError(f"No .jsonl logs found under {local_logs}")
            self._log("Local logs look good.")

            self._log("Deleting remote logs dir…")
            rm_cmd = f"rm -rf {self._cfg.remote.logs_dir}"
            rm = ssh_run(user=user, host=host, remote_cmd=rm_cmd, x11=False, timeout_s=30)
            if not rm.ok:
                fb = self._cfg.pi_ssh.fallback_host
                if fb and fb != host:
                    rm2 = ssh_run(user=user, host=fb, remote_cmd=rm_cmd, x11=False, timeout_s=30)
                    if not rm2.ok:
                        raise RuntimeError(rm2.stderr.strip() or "remote rm failed")
                else:
                    raise RuntimeError(rm.stderr.strip() or "remote rm failed")
            self._log("Remote logs deleted.")

        self._run_bg("Pulling logs…", work)

    def _choose_wifi_dialog(self, *, title: str) -> tuple[str | None, str | None]:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setObjectName("dialog")
        dlg.resize(560, 220)

        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        lbl = QtWidgets.QLabel("Pick a Wi‑Fi network to connect to.")
        lbl.setObjectName("subtitle")
        lay.addWidget(lbl)

        ssids = list_wifi_ssids_windows()
        if not ssids:
            ssids = [self._cfg.hotspot.ssid]

        combo = QtWidgets.QComboBox()
        combo.addItems(ssids)
        # Prefer configured hotspot if present
        i = combo.findText(self._cfg.hotspot.ssid)
        if i >= 0:
            combo.setCurrentIndex(i)
        lay.addWidget(combo)

        pw = QtWidgets.QLineEdit()
        pw.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        pw.setPlaceholderText("Password (only if needed)")
        lay.addWidget(pw)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setObjectName("btn_ghost")
        btn_ok = QtWidgets.QPushButton("Connect")
        btn_ok.setObjectName("btn_primary")
        row.addWidget(btn_cancel)
        row.addWidget(btn_ok)
        lay.addLayout(row)

        out_ssid: str | None = None
        out_pw: str | None = None

        def accept() -> None:
            nonlocal out_ssid, out_pw
            out_ssid = str(combo.currentText() or "").strip() or None
            out_pw = str(pw.text() or "").strip() or None
            dlg.accept()

        btn_ok.clicked.connect(accept)
        btn_cancel.clicked.connect(dlg.reject)
        dlg.exec()
        return out_ssid, out_pw

    def _on_back_internet(self) -> None:
        ssid, pw = self._choose_wifi_dialog(title="Back to internet")
        if not ssid:
            return

        def work() -> None:
            # Try connecting; if it fails and no password was provided, ask once more.
            password = pw or self._cfg.hotspot.password
            self._log(f"Connecting Wi-Fi: {ssid!r}")
            r = connect_wifi_windows(ssid=ssid, password=password)
            self._log(r.stdout.strip() or "netsh returned.")
            if not r.ok and not pw:
                self._log("Connect failed; prompting for password…")
                ssid2, pw2 = self._choose_wifi_dialog(title="Enter Wi‑Fi password")
                if not ssid2:
                    raise RuntimeError(r.stderr.strip() or "Failed to connect Wi-Fi")
                r = connect_wifi_windows(ssid=ssid2, password=pw2 or "")
                self._log(r.stdout.strip() or "netsh returned.")
            if not r.ok:
                raise RuntimeError(r.stderr.strip() or "Failed to connect Wi-Fi")

            self._log("Checking internet…")
            if not has_internet():
                raise RuntimeError("No internet connectivity detected")
            self._log("Internet OK.")

        self._run_bg("Connecting to internet…", work)

    def _on_analyze(self) -> None:
        try:
            w = CalibrationAnalysisWindow(run_config_path=self._config_path)
            # Keep a strong reference; otherwise the window may be garbage-collected and close immediately.
            w.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
            self._analysis_windows.add(w)
            w.destroyed.connect(lambda *_: self._analysis_windows.discard(w))
            w.show()
            self._log("Opened Analysis window.")
        except Exception as e:
            self._log(f"ERROR: {type(e).__name__}: {e}")

    # ---- analysis ----

    def _run_analysis(self) -> None:
        # Legacy path kept for troubleshooting; the main flow now uses CalibrationAnalysisWindow.
        app_cfg = load_config()
        api_key = app_cfg.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        base_dir = Path(self._cfg.analysis.analysis_base_dir).resolve()
        logs_dir = Path(self._cfg.local.logs_dir).resolve()
        if not logs_dir.exists():
            raise RuntimeError(f"logs_dir not found: {logs_dir}")

        # Analysis instructions live in the system prompt; user message includes only file paths.

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
            hide_think=True,
            allow_submodels=False,
        )
        session = ChatSession(api_key=api_key, config=ccfg)
        session.set_system_prompt(build_analysis_system_prompt(cfg=self._cfg.analysis))

        # Provide a repo-relative path so python_sandbox can `copy_from_repo`.
        try:
            rel_logs = logs_dir.relative_to(base_dir)
            rel_logs_s = rel_logs.as_posix()
        except Exception:
            rel_logs_s = logs_dir.as_posix()

        msg = f"Files/logs folder to analyze (repo-relative if possible):\n- {rel_logs_s}\n"

        delta = session.send(msg)
        out = (delta.assistant_text or "").strip()
        self._log("=== LLM RECOMMENDATION ===")
        self._log(out)


def main() -> None:
    w = CalibrationRunnerWindow()
    raise SystemExit(w.exec())


if __name__ == "__main__":
    main()
