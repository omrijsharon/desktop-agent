"""desktop_agent.terminal_agent_ui

Terminal Agent: a split-pane GUI with Chat + Terminal.

- The model chats in the left pane.
- When the model outputs <Terminal>...</Terminal> blocks, those commands are executed
  in the terminal pane automatically (PowerShell on Windows).
- Command output is streamed live to the terminal pane and summarized back into the
  model context so it can continue autonomously.

Run:
    python -m desktop_agent.terminal_agent_ui
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import secrets

from PySide6 import QtCore, QtGui, QtWidgets

from .chat_session import ChatConfig, ChatSession
from .config import DEFAULT_MODEL, load_config
from .chat_ui import Bubble, _strip_think  # reuse bubble + think styling


JsonDict = dict[str, Any]

_TERM_BLOCK_RE = re.compile(r"(?is)<terminal>(.*?)</terminal>")
_ANSI_RE = re.compile(r"(?s)\x1b\[[0-?]*[ -/]*[@-~]")
_OSC_RE = re.compile(r"(?s)\x1b\].*?(?:\x07|\x1b\\)")
_PROMPT_SENTINEL = "__TA_PROMPT__"
_DONE_RE = re.compile(r"__TA_DONE__[0-9a-f]{6,32}", re.IGNORECASE)
_LINUX_PROMPT_RE = re.compile(r"(?m)^[^\r\n]{0,80}@[^\r\n]{1,80}:[^\r\n]{0,200}[$#]\s*$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_qss() -> str:
    # Reuse the chat UI style and add terminal-specific bits.
    chat_qss = (_repo_root() / "ui" / "chat" / "style.qss").read_text(encoding="utf-8")
    terminal_qss = """
QPlainTextEdit#terminal_view {
    background: #0b0e14;
    color: #d6deeb;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 10px;
    font-family: Consolas, "Cascadia Mono", "SF Mono", monospace;
    font-size: 12px;
}
QLineEdit#terminal_input {
    background: rgba(255,255,255,0.06);
    color: #eaf2ff;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px;
    font-family: Consolas, "Cascadia Mono", "SF Mono", monospace;
    font-size: 12px;
}
QLabel#cwd_label {
    color: rgba(230,240,255,0.70);
}
"""
    return chat_qss + "\n" + terminal_qss


def extract_terminal_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    for m in _TERM_BLOCK_RE.finditer(text or ""):
        inner = (m.group(1) or "").strip()
        if inner:
            blocks.append(inner)
    return blocks


def strip_terminal_blocks(text: str) -> str:
    # Remove the blocks entirely; model explanations remain.
    t = _TERM_BLOCK_RE.sub("", text or "")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    elapsed_s: float
    cwd_after: str | None = None


class _Signals(QtCore.QObject):
    busy = QtCore.Signal(bool)
    chat_partial = QtCore.Signal(str)
    chat_final = QtCore.Signal(str)
    tool_msg = QtCore.Signal(str)
    terminal_append = QtCore.Signal(str)
    cwd_changed = QtCore.Signal(str)
    tokens = QtCore.Signal(str)
    error = QtCore.Signal(str)


class _TermSignals(QtCore.QObject):
    chunk = QtCore.Signal(str)


def _strip_ansi(text: str) -> str:
    if not text:
        return ""
    t = _OSC_RE.sub("", text)
    t = _ANSI_RE.sub("", t)
    return t


def _powershell_prompt_at_end(text: str) -> bool:
    """Heuristic: did we end at a PowerShell prompt (our sentinel prompt)?"""

    tail = (text or "")[-600:]
    # prompt() prints: "__TA_PROMPT__ <pwd>> "
    return bool(re.search(r"__TA_PROMPT__ .*?>\s*$", tail))


class _TerminalRunner:
    """Terminal runner with two modes:

    - One-shot PowerShell commands (default): runs and exits, returns an exit code.
    - Interactive SSH session (persistent): if the user/model runs `ssh ...` without a
      remote command, we keep that ssh process open and route subsequent inputs to it.

    This gives you an interactive SSH shell without requiring a full Windows PTY.
    """

    def __init__(self, *, initial_cwd: Path, on_chunk: Optional[callable] = None) -> None:
        self.cwd = Path(initial_cwd).resolve()
        self._on_chunk = on_chunk
        self._proc: subprocess.Popen[bytes] | None = None  # current one-shot
        self._session_proc: subprocess.Popen[bytes] | None = None  # interactive session (ssh)
        self._session_desc: str | None = None
        self._stop_flag = False
        self._buf: list[str] = []
        self._buf_lock = QtCore.QMutex()

    def _emit(self, s: str) -> None:
        cb = self._on_chunk
        if cb is not None and s:
            cb(s)

    def _append_buf(self, s: str) -> None:
        self._buf_lock.lock()
        try:
            self._buf.append(s)
            if len(self._buf) > 20000:
                self._buf = self._buf[-12000:]
        finally:
            self._buf_lock.unlock()

    def _start_readers(self, p: subprocess.Popen[bytes]) -> None:
        def reader(stream) -> None:
            try:
                while True:
                    if self._stop_flag:
                        break
                    chunk = stream.read(1024)
                    if not chunk:
                        break
                    try:
                        s = chunk.decode("utf-8", errors="replace")
                    except Exception:
                        s = chunk.decode(errors="replace")
                    s = _strip_ansi(s)
                    if not s:
                        continue
                    self._append_buf(s)
                    self._emit(s)
            except Exception:
                pass

        if p.stdout is not None:
            threading.Thread(target=reader, args=(p.stdout,), daemon=True).start()
        if p.stderr is not None:
            threading.Thread(target=reader, args=(p.stderr,), daemon=True).start()

    def stop_and_reset(self) -> None:
        """Hard stop: kills any running process and clears interactive session."""
        self._stop_flag = True
        for p in (self._proc, self._session_proc):
            if p is None:
                continue
            try:
                p.kill()
            except Exception:
                pass
        self._proc = None
        self._session_proc = None
        self._session_desc = None
        self._stop_flag = False

    def interrupt(self) -> None:
        # Best-effort: kill current one-shot; for SSH session, kill it (hard).
        self.stop_and_reset()

    def _is_interactive_ssh(self, cmd: str) -> bool:
        c = (cmd or "").strip()
        if not c:
            return False
        if not re.match(r"(?is)^ssh\\b", c):
            return False
        # Heuristic: if it contains a quoted remote command, treat as non-interactive.
        if re.search(r"(?s)(?:\\\"|\\')", c):
            return False
        return True

    def _start_ssh_session(self, cmd: str) -> None:
        args = shlex.split(cmd, posix=False)
        if not args:
            raise RuntimeError("empty ssh command")
        if args[0].lower() != "ssh":
            raise RuntimeError("not an ssh command")
        # Force TTY allocation so the remote stays interactive.
        if "-t" not in args and "-tt" not in args:
            args.insert(1, "-tt")

        p = subprocess.Popen(  # noqa: S603
            args,
            cwd=str(self.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
        self._session_proc = p
        self._session_desc = cmd.strip()
        self._start_readers(p)

    def session_status(self) -> str:
        p = self._session_proc
        if p is None:
            return "local"
        if p.poll() is not None:
            self._session_proc = None
            self._session_desc = None
            return "local"
        return f"ssh:{self._session_desc or 'connected'}"

    def _wait_idle_capture(self, *, start_idx: int, idle_ms: int, max_wait_s: float) -> str:
        start = time.monotonic()
        end_deadline = start + float(max_wait_s)
        idle_s = max(0.05, float(idle_ms) / 1000.0)
        min_wait_s = max(0.15, idle_s)
        earliest_done = start + min_wait_s

        last_change = time.monotonic()
        last_len = start_idx

        while time.monotonic() < end_deadline and not self._stop_flag:
            self._buf_lock.lock()
            try:
                cur_len = len(self._buf)
            finally:
                self._buf_lock.unlock()
            if cur_len != last_len:
                last_len = cur_len
                last_change = time.monotonic()
            if time.monotonic() >= earliest_done and (time.monotonic() - last_change) >= idle_s:
                break
            time.sleep(0.03)

        self._buf_lock.lock()
        try:
            return "".join(self._buf[start_idx:])
        finally:
            self._buf_lock.unlock()

    def send_and_collect(self, *, block: str, idle_ms: int = 450, max_wait_s: float = 25.0) -> CommandResult:
        cmd = (block or "").strip()
        if not cmd:
            return CommandResult(command="", exit_code=0, stdout="", stderr="", elapsed_s=0.0)

        # If we're already inside an interactive ssh session, route input there.
        if self._session_proc is not None and self._session_proc.poll() is None:
            p = self._session_proc
            assert p.stdin is not None
            self._buf_lock.lock()
            try:
                start_idx = len(self._buf)
            finally:
                self._buf_lock.unlock()
            start = time.monotonic()
            p.stdin.write((cmd + "\n").encode("utf-8", errors="replace"))
            p.stdin.flush()
            captured = self._wait_idle_capture(start_idx=start_idx, idle_ms=idle_ms, max_wait_s=max_wait_s)
            return CommandResult(command=cmd, exit_code=0, stdout=captured, stderr="", elapsed_s=float(time.monotonic() - start), cwd_after=str(self.cwd))
        if self._session_proc is not None and self._session_proc.poll() is not None:
            self._session_proc = None
            self._session_desc = None

        # Start interactive ssh session if requested.
        if self._is_interactive_ssh(cmd):
            self._buf_lock.lock()
            try:
                start_idx = len(self._buf)
            finally:
                self._buf_lock.unlock()
            start = time.monotonic()
            self._start_ssh_session(cmd)
            captured = self._wait_idle_capture(start_idx=start_idx, idle_ms=idle_ms, max_wait_s=max_wait_s)
            return CommandResult(command=cmd, exit_code=0, stdout=captured, stderr="", elapsed_s=float(time.monotonic() - start), cwd_after=str(self.cwd))

        # One-shot PowerShell command.
        pwd_tag = "__TERMINAL_AGENT_PWD__"
        wrapped = f"{cmd}\n; Write-Output '{pwd_tag}'\n; (Get-Location).Path\n"
        start = time.monotonic()
        self._stop_flag = False
        p = subprocess.Popen(  # noqa: S603
            ["powershell.exe", "-NoProfile", "-Command", wrapped],
            cwd=str(self.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
        self._proc = p
        out_parts: list[str] = []
        err_parts: list[str] = []

        def collect(stream, sink) -> None:
            if stream is None:
                return
            while True:
                chunk = stream.read(1024)
                if not chunk:
                    break
                try:
                    s = chunk.decode("utf-8", errors="replace")
                except Exception:
                    s = chunk.decode(errors="replace")
                s = _strip_ansi(s)
                sink.append(s)
                self._emit(s)

        try:
            assert p.stdout is not None
            assert p.stderr is not None
            t_out = threading.Thread(target=collect, args=(p.stdout, out_parts), daemon=True)
            t_err = threading.Thread(target=collect, args=(p.stderr, err_parts), daemon=True)
            t_out.start()
            t_err.start()

            deadline = start + float(max_wait_s)
            while p.poll() is None and time.monotonic() < deadline and not self._stop_flag:
                time.sleep(0.03)
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        finally:
            self._proc = None

        rc = int(p.returncode or 0)
        stdout = "".join(out_parts)
        stderr = "".join(err_parts)
        cwd_after: str | None = None

        if pwd_tag in stdout:
            idx = stdout.rfind(pwd_tag)
            tail = stdout[idx:].splitlines()
            if len(tail) >= 2:
                cand = tail[1].strip()
                if cand:
                    cwd_after = cand
            # Remove tag+pwd from shown stdout.
            cleaned_lines: list[str] = []
            skip_next = False
            for ln in stdout.splitlines(True):
                if pwd_tag in ln:
                    skip_next = True
                    continue
                if skip_next:
                    skip_next = False
                    continue
                cleaned_lines.append(ln)
            stdout = "".join(cleaned_lines)

        if cwd_after:
            try:
                p2 = Path(cwd_after).resolve()
                if p2.exists() and p2.is_dir():
                    self.cwd = p2
            except Exception:
                pass

        return CommandResult(
            command=cmd,
            exit_code=rc,
            stdout=stdout,
            stderr=stderr,
            elapsed_s=float(time.monotonic() - start),
            cwd_after=cwd_after,
        )


class _ConptyTerminal:
    """ConPTY-backed interactive PowerShell session using pywinpty.

    This is the closest equivalent to a real PowerShell terminal on Windows:
    - interactive prompts
    - SSH behaves like a real terminal (password/sudo/TTY programs)
    """

    def __init__(self, *, initial_cwd: Path, on_chunk: Optional[callable] = None) -> None:
        self.cwd = Path(initial_cwd).resolve()
        self._on_chunk = on_chunk
        self._stop_flag = False
        self._pty = None
        self._buf: list[str] = []
        self._buf_lock = QtCore.QMutex()
        self._reader_thread: Optional[threading.Thread] = None
        self._in_ssh: bool = False
        self.start()

    def _emit(self, s: str) -> None:
        cb = self._on_chunk
        if cb is not None and s:
            cb(s)

    def _append_buf(self, s: str) -> None:
        self._buf_lock.lock()
        try:
            self._buf.append(s)
            if len(self._buf) > 30000:
                self._buf = self._buf[-18000:]
        finally:
            self._buf_lock.unlock()

    def _maybe_update_ssh_state_from_text(self, text: str) -> None:
        """Best-effort: infer if we're inside an interactive SSH shell.

        We cannot reliably detect the foreground program in a terminal, but for our
        use-case (PowerShell -> ssh -> Linux prompt) a heuristic works well:
        - SSH login banners / "Last login:" suggests we entered a remote session.
        - A Linux-style prompt "user@host:~ $" suggests we're in a remote shell.
        - Seeing our PowerShell sentinel prompt at the end suggests we're back.
        """

        if not text:
            return
        t = text.strip()
        if not t:
            return
        t_cf = t.casefold()
        is_banner = t_cf.startswith("last login:") or t_cf.startswith("linux ") or t_cf.startswith("debian gnu/linux")
        if is_banner or _LINUX_PROMPT_RE.search(t):
            self._in_ssh = True
            return
        # If we see our PowerShell prompt sentinel at the end, we are not in ssh.
        if _powershell_prompt_at_end(t):
            self._in_ssh = False

    def start(self) -> None:
        self._stop_flag = False
        try:
            from winpty import PtyProcess  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"pywinpty not available: {e}") from e

        # Spawn PowerShell in a real PTY (ConPTY backend=1).
        # Backend values in pywinpty:
        #   0 = winpty, 1 = conpty (preferred), others may exist.
        self._pty = PtyProcess.spawn(["powershell.exe", "-NoProfile", "-NoLogo"], cwd=str(self.cwd), backend=1)
        self._emit("[terminal] ConPTY interactive session started.\n")

        def reader() -> None:
            assert self._pty is not None
            while not self._stop_flag:
                try:
                    data = self._pty.read(1024)
                except Exception:
                    break
                if not data:
                    time.sleep(0.02)
                    continue
                if isinstance(data, bytes):
                    try:
                        s = data.decode("utf-8", errors="replace")
                    except Exception:
                        s = data.decode(errors="replace")
                else:
                    s = str(data)
                s = _strip_ansi(s)
                if not s:
                    continue
                self._maybe_update_ssh_state_from_text(s)
                # Hide internal completion markers from the terminal pane; they are only for
                # determining when a one-shot command has finished.
                s_ui = _DONE_RE.sub("", s)
                self._append_buf(s)
                if s_ui:
                    self._emit(s_ui)

        self._reader_thread = threading.Thread(target=reader, daemon=True)
        self._reader_thread.start()

        # Set a deterministic prompt so we can detect readiness in one-shot captures.
        self.send(
            (
                f"$global:__ta_prompt='{_PROMPT_SENTINEL}';"
                "function prompt { \"$($global:__ta_prompt) $PWD> \" }\n"
            )
        )
        self.send(f"Set-Location -LiteralPath {str(self.cwd)!r}\n")

    def stop_and_reset(self) -> None:
        self._stop_flag = True
        try:
            if self._pty is not None:
                self._pty.close()
        except Exception:
            pass
        self._pty = None
        self._stop_flag = False
        self.start()

    def interrupt(self) -> None:
        # Ctrl+C
        self.send("\x03")

    def send(self, text: str) -> None:
        if self._pty is None:
            self.start()
        assert self._pty is not None
        try:
            # ConPTY behaves best with CRLF.
            t = text
            if not (t.endswith("\n") or t.endswith("\r")):
                t = t + "\r\n"
            elif t.endswith("\n") and not t.endswith("\r\n"):
                t = t[:-1] + "\r\n"
            self._pty.write(t)
        except Exception:
            self.stop_and_reset()
            assert self._pty is not None
            self._pty.write(text if text.endswith("\r\n") else (text + "\r\n"))

    def _wait_for_prompt(self, *, start_idx: int, max_wait_s: float) -> str:
        deadline = time.monotonic() + float(max_wait_s)
        while time.monotonic() < deadline and not self._stop_flag:
            self._buf_lock.lock()
            try:
                captured = "".join(self._buf[start_idx:])
            finally:
                self._buf_lock.unlock()
            if _PROMPT_SENTINEL in captured:
                return captured
            time.sleep(0.03)
        return captured

    def _wait_idle(self, *, start_idx: int, idle_ms: int, max_wait_s: float) -> str:
        start = time.monotonic()
        deadline = start + float(max_wait_s)
        idle_s = max(0.05, float(idle_ms) / 1000.0)
        earliest_done = start + max(0.15, idle_s)
        last_len = start_idx
        last_change = start
        while time.monotonic() < deadline and not self._stop_flag:
            self._buf_lock.lock()
            try:
                cur_len = len(self._buf)
            finally:
                self._buf_lock.unlock()
            if cur_len != last_len:
                last_len = cur_len
                last_change = time.monotonic()
            if time.monotonic() >= earliest_done and (time.monotonic() - last_change) >= idle_s:
                break
            time.sleep(0.03)
        self._buf_lock.lock()
        try:
            return "".join(self._buf[start_idx:])
        finally:
            self._buf_lock.unlock()

    def _wait_for_substring(self, *, start_idx: int, needle: str, max_wait_s: float) -> str:
        deadline = time.monotonic() + float(max_wait_s)
        while time.monotonic() < deadline and not self._stop_flag:
            self._buf_lock.lock()
            try:
                captured = "".join(self._buf[start_idx:])
            finally:
                self._buf_lock.unlock()
            if needle and needle in captured:
                return captured
            time.sleep(0.03)
        return captured

    def _looks_like_interactive_ssh(self, cmd: str) -> bool:
        c = (cmd or "").strip()
        if not re.match(r"(?is)^ssh\\b", c):
            return False
        # If it contains quotes, assume non-interactive `ssh host "cmd"`.
        return not bool(re.search(r"(?s)(?:\\\"|\\')", c))

    def send_and_collect(self, *, block: str, idle_ms: int = 450, max_wait_s: float = 25.0) -> CommandResult:
        cmd = (block or "").strip()
        if not cmd:
            return CommandResult(command="", exit_code=0, stdout="", stderr="", elapsed_s=0.0)

        start = time.monotonic()
        self._buf_lock.lock()
        try:
            start_idx = len(self._buf)
        finally:
            self._buf_lock.unlock()

        # If we're currently inside an interactive SSH shell, do NOT append PowerShell
        # completion markers (they'd be typed into the remote shell). Use idle capture.
        if self._in_ssh:
            self.send(cmd)
            captured = self._wait_idle(start_idx=start_idx, idle_ms=idle_ms, max_wait_s=max_wait_s)
        # Starting interactive SSH: switch modes and use idle capture (no prompt sentinel).
        elif self._looks_like_interactive_ssh(cmd):
            self._in_ssh = True
            self.send(cmd)
            captured = self._wait_idle(start_idx=start_idx, idle_ms=idle_ms, max_wait_s=max_wait_s)
        else:
            done = f"__TA_DONE__{secrets.token_hex(4)}"
            self.send(cmd)
            # Ensure a deterministic completion marker.
            self.send(f"Write-Output '{done}'")
            captured = self._wait_for_substring(start_idx=start_idx, needle=done, max_wait_s=max_wait_s)
            if done in captured:
                captured = captured.split(done, 1)[0]

        # Only leave SSH mode if we end at our PowerShell prompt; the sentinel may appear
        # earlier in the transcript (e.g., before the ssh command starts).
        if _powershell_prompt_at_end(captured):
            self._in_ssh = False

        elapsed = time.monotonic() - start
        return CommandResult(command=cmd, exit_code=0, stdout=captured, stderr="", elapsed_s=float(elapsed), cwd_after=str(self.cwd))

    def session_status(self) -> str:
        # With ConPTY, there's a single interactive terminal; it may contain SSH or anything else.
        return "interactive(conpty)"


class _Worker(QtCore.QThread):
    def __init__(
        self,
        *,
        session: ChatSession,
        user_text: str,
        terminal: Any,
        max_rounds: int,
        hide_think: bool,
        parent: Optional[QtCore.QObject],
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.user_text = user_text
        self.terminal = terminal
        self.max_rounds = int(max_rounds)
        self.hide_think = hide_think
        self.signals = _Signals()

    def request_stop(self) -> None:
        try:
            self.requestInterruption()
        except Exception:
            pass
        try:
            self.terminal.interrupt()
        except Exception:
            pass

    def run(self) -> None:  # type: ignore[override]
        self.signals.busy.emit(True)
        try:
            pending_user = self.user_text
            for _r in range(max(1, self.max_rounds)):
                if self.isInterruptionRequested():
                    return

                full_raw = ""
                last_emit = 0.0
                last_tok_emit = 0.0
                prompt_tok_est = self.session.estimate_prompt_tokens(user_text=pending_user)
                max_ctx = int(getattr(self.session.cfg, "context_window_tokens", 0) or 0)

                for ev in self.session.send_stream(pending_user):
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
                            shown = strip_terminal_blocks(shown)
                            self.signals.chat_partial.emit(shown)
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
                    elif et == "error":
                        err = ev.get("error")
                        self.signals.error.emit(str(err))
                        return

                if self.isInterruptionRequested():
                    return

                final_shown = _strip_think(full_raw) if self.hide_think else full_raw
                final_shown = strip_terminal_blocks(final_shown)
                self.signals.chat_final.emit(final_shown.strip())
                self.signals.tokens.emit(self.session.usage_ratio_text())

                blocks = extract_terminal_blocks(full_raw)
                if not blocks:
                    return

                self.signals.tool_msg.emit(f"[terminal] executing {len(blocks)} block(s)…")
                results: list[CommandResult] = []

                for b in blocks:
                    if self.isInterruptionRequested():
                        return
                    self.signals.terminal_append.emit(f"\nPS {getattr(self.terminal, 'cwd', '')}> {b}\n")
                    res = self.terminal.send_and_collect(block=b, idle_ms=450, max_wait_s=25.0)
                    results.append(res)
                    if res.cwd_after:
                        self.signals.cwd_changed.emit(str(getattr(self.terminal, "cwd", res.cwd_after)))

                parts: list[str] = []
                for res in results:
                    so = (res.stdout or "").strip()
                    se = (res.stderr or "").strip()
                    if len(so) > 6000:
                        so = so[:6000] + "\n…(truncated)…"
                    if len(se) > 2000:
                        se = se[:2000] + "\n…(truncated)…"
                    parts.append(
                        "Command:\n"
                        + res.command
                        + "\n"
                        + f"Exit: {res.exit_code}  Elapsed: {res.elapsed_s:.2f}s\n"
                        + (f"CWD: {res.cwd_after}\n" if res.cwd_after else "")
                        + ("STDOUT:\n" + so + "\n" if so else "STDOUT: (empty)\n")
                        + ("STDERR:\n" + se + "\n" if se else "STDERR: (empty)\n")
                    )

                term_state = "unknown"
                try:
                    term_state = str(getattr(self.terminal, "session_status", lambda: "interactive")())
                except Exception:
                    term_state = "interactive"
                pending_user = (
                    f"Terminal state: {term_state}\n\n"
                    "Terminal command results:\n\n"
                    + "\n---\n".join(parts)
                    + "\n\n"
                    "If you need to run more commands, respond with more <Terminal>...</Terminal> blocks. "
                    "Otherwise, reply normally to the user with a summary of what happened.\n"
                )
        except Exception as e:  # noqa: BLE001
            self.signals.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.signals.busy.emit(False)


class TerminalAgentWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        super().__init__()

        self._session = self._make_session()
        self._terminal: Any = None
        self._term_signals = _TermSignals()
        self._worker: _Worker | None = None
        self._hide_think = False

        self._build_ui()
        self._apply_style()

        # Now that widgets exist, wire terminal output + start terminal backend.
        self._term_signals.chunk.connect(self._terminal_append)
        try:
            self._terminal = _ConptyTerminal(initial_cwd=_repo_root(), on_chunk=lambda s: self._term_signals.chunk.emit(s))
            self._using_conpty = True
        except Exception as e:
            self._terminal = _TerminalRunner(initial_cwd=_repo_root(), on_chunk=lambda s: self._term_signals.chunk.emit(s))
            self._using_conpty = False
            QtCore.QTimer.singleShot(0, lambda: self._append_chat(f"[terminal] ConPTY unavailable; using fallback runner: {e}", kind="tool"))

        self.resize(1680, 900)
        self.setWindowTitle("Terminal Agent")
        try:
            self._cwd_label.setText(str(getattr(self._terminal, "cwd", _repo_root())))
        except Exception:
            self._cwd_label.setText(str(_repo_root()))

        self._append_chat(
            "Tip: Ask for something like “@Main, set up a venv and run tests” and it can execute commands via <Terminal> blocks.",
            kind="tool",
        )

    def exec(self) -> int:
        self.show()
        return int(self._app.exec())

    def _apply_style(self) -> None:
        try:
            self._app.setStyleSheet(_load_qss())
        except Exception:
            pass
        font = QtGui.QFont()
        font.setFamilies(["SF Pro Display", "Segoe UI Variable", "Segoe UI", "Inter", "Arial"])
        font.setPointSize(10)
        self._app.setFont(font)

    def _make_session(self) -> ChatSession:
        app_cfg = load_config()
        api_key = app_cfg.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        ccfg = ChatConfig(
            model=(app_cfg.openai_model or DEFAULT_MODEL),
            enable_web_search=True,
            web_search_context_size="medium",
            enable_file_search=False,
            tool_base_dir=_repo_root(),
            allow_read_file=True,
            allow_write_files=True,
            allow_python_sandbox=True,
            python_sandbox_timeout_s=30.0,
            allow_model_set_system_prompt=True,
            allow_model_propose_tools=True,
            allow_model_create_tools=True,
            allow_model_create_analysis_tools=True,
            hide_think=False,
            allow_submodels=True,
        )
        s = ChatSession(api_key=api_key, config=ccfg)
        s.set_system_prompt(
            (
                "You are Terminal Agent, an assistant that can run PowerShell commands.\n"
                "When you want to execute commands, emit a <Terminal>...</Terminal> block.\n"
                "Anything inside <Terminal>...</Terminal> will be executed in the attached interactive PowerShell terminal (ConPTY).\n"
                "This behaves like a real terminal: SSH is interactive and stays open, and you can run subsequent commands naturally.\n"
                "Use `exit` to exit remote shells or close programs. The user can press Stop to reset the terminal.\n"
                "Prefer safe, deterministic commands. Avoid destructive operations unless explicitly requested.\n"
                "For SSH/SCP, prefer options like: -o StrictHostKeyChecking=accept-new\n"
                "After commands run, you will receive their stdout/stderr/exit codes and can continue.\n"
            )
        )
        return s

    # ---- UI ----

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

        # Left: chat
        chat_panel = QtWidgets.QFrame()
        chat_panel.setObjectName("panel")
        cl = QtWidgets.QVBoxLayout(chat_panel)
        cl.setContentsMargins(12, 12, 12, 12)
        cl.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Chat")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch(1)
        self._btn_stop = QtWidgets.QPushButton("Stop")
        self._btn_stop.setObjectName("btn_ghost")
        header.addWidget(self._btn_stop)
        cl.addLayout(header)

        self._chat_scroll = QtWidgets.QScrollArea()
        self._chat_scroll.setWidgetResizable(True)
        self._chat_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        cl.addWidget(self._chat_scroll, 1)

        chat = QtWidgets.QWidget()
        self._chat_layout = QtWidgets.QVBoxLayout(chat)
        self._chat_layout.setContentsMargins(0, 0, 0, 0)
        self._chat_layout.setSpacing(10)
        self._chat_layout.addStretch(1)
        self._chat_scroll.setWidget(chat)

        bottom = QtWidgets.QHBoxLayout()
        self._chat_input = QtWidgets.QTextEdit()
        self._chat_input.setPlaceholderText("Message… (Shift+Enter for newline)")
        self._chat_input.setFixedHeight(92)
        self._btn_send = QtWidgets.QPushButton("Send")
        self._btn_send.setObjectName("btn_primary")
        bottom.addWidget(self._chat_input, 1)
        bottom.addWidget(self._btn_send)
        cl.addLayout(bottom)

        self._tokens = QtWidgets.QLabel("")
        self._tokens.setObjectName("subtitle")
        self._tokens.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        cl.addWidget(self._tokens)

        splitter.addWidget(chat_panel)

        # Right: terminal
        term_panel = QtWidgets.QFrame()
        term_panel.setObjectName("panel")
        tl = QtWidgets.QVBoxLayout(term_panel)
        tl.setContentsMargins(12, 12, 12, 12)
        tl.setSpacing(10)

        thead = QtWidgets.QHBoxLayout()
        ttitle = QtWidgets.QLabel("Terminal")
        ttitle.setObjectName("title")
        thead.addWidget(ttitle)
        thead.addStretch(1)
        self._cwd_label = QtWidgets.QLabel("")
        self._cwd_label.setObjectName("cwd_label")
        thead.addWidget(self._cwd_label)
        tl.addLayout(thead)

        self._terminal_view = QtWidgets.QPlainTextEdit()
        self._terminal_view.setObjectName("terminal_view")
        self._terminal_view.setReadOnly(True)
        self._terminal_view.setWordWrapMode(QtGui.QTextOption.WrapMode.NoWrap)
        tl.addWidget(self._terminal_view, 1)

        tbot = QtWidgets.QHBoxLayout()
        self._terminal_input = QtWidgets.QLineEdit()
        self._terminal_input.setObjectName("terminal_input")
        self._terminal_input.setPlaceholderText("Type a command to run manually…")
        self._btn_run = QtWidgets.QPushButton("Run")
        self._btn_run.setObjectName("btn_ghost")
        tbot.addWidget(self._terminal_input, 1)
        tbot.addWidget(self._btn_run)
        tl.addLayout(tbot)

        splitter.addWidget(term_panel)
        splitter.setSizes([860, 820])

        # wiring
        self._btn_send.clicked.connect(self._on_send)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_run.clicked.connect(self._on_run_manual)
        self._terminal_input.returnPressed.connect(self._on_run_manual)
        self._chat_input.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if obj is self._chat_input and event.type() == QtCore.QEvent.Type.KeyPress:
            e = event  # type: ignore[assignment]
            if isinstance(e, QtGui.QKeyEvent) and e.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                if e.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    return False
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    # ---- chat helpers ----

    def _append_chat(self, text: str, *, kind: str) -> Bubble:
        bubble = Bubble(text=text, kind=kind)
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
        QtCore.QTimer.singleShot(0, self._scroll_chat_bottom)
        return bubble

    def _scroll_chat_bottom(self) -> None:
        bar = self._chat_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_busy(self, busy: bool) -> None:
        # Keep chat input enabled even while busy so the user can interrupt.
        # Sending while busy will stop the current worker and start a new turn.
        self._chat_input.setEnabled(True)
        self._btn_send.setEnabled(True)
        self._btn_stop.setEnabled(True)

        # Manual terminal interaction is still allowed while busy, but we disable the
        # Run button to avoid racing the model's auto-runs.
        self._btn_run.setEnabled(not busy)
        self._terminal_input.setEnabled(not busy)

    # ---- terminal helpers ----

    def _terminal_append(self, s: str) -> None:
        self._terminal_view.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self._terminal_view.insertPlainText(s)
        self._terminal_view.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    # ---- actions ----

    def _on_run_manual(self) -> None:
        cmd = self._terminal_input.text().strip()
        if not cmd:
            return
        self._terminal_input.clear()
        self._terminal_append(f"\nPS {self._terminal.cwd}> {cmd}\n")
        try:
            res = self._terminal.send_and_collect(block=cmd, idle_ms=450, max_wait_s=25.0)
            if res.cwd_after:
                self._cwd_label.setText(str(self._terminal.cwd))
        except Exception as e:
            self._terminal_append(f"\n[error] {type(e).__name__}: {e}\n")

    def _on_stop(self) -> None:
        w = self._worker
        if w is not None and w.isRunning():
            w.request_stop()
            self._append_chat("[stopped]", kind="tool")
        # Hard stop: kill any running command and close interactive SSH sessions.
        try:
            self._terminal.stop_and_reset()
            self._terminal_append("\n[terminal reset]\n")
        except Exception:
            pass

    def _on_send(self) -> None:
        text = self._chat_input.toPlainText().strip()
        if not text:
            return
        self._chat_input.clear()
        self._append_chat(text, kind="user")

        # One worker at a time.
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()

        assistant_bubble: Bubble | None = None

        w = _Worker(
            session=self._session,
            user_text=text,
            terminal=self._terminal,
            max_rounds=6,
            hide_think=self._hide_think,
            parent=self,
        )
        self._worker = w

        w.signals.busy.connect(self._set_busy)
        w.signals.tokens.connect(self._tokens.setText)
        w.signals.tool_msg.connect(lambda t: self._append_chat(t, kind="tool"))
        w.signals.terminal_append.connect(self._terminal_append)
        w.signals.cwd_changed.connect(self._cwd_label.setText)
        w.signals.error.connect(lambda e: self._append_chat(f"[error] {e}", kind="tool"))

        def on_partial(t: str) -> None:
            nonlocal assistant_bubble
            if assistant_bubble is None:
                assistant_bubble = self._append_chat("", kind="assistant")
            assistant_bubble.set_streaming_text(t)
            QtCore.QTimer.singleShot(0, self._scroll_chat_bottom)

        def on_final(t: str) -> None:
            nonlocal assistant_bubble
            if assistant_bubble is None:
                assistant_bubble = self._append_chat("", kind="assistant")
            assistant_bubble.set_text(t)
            QtCore.QTimer.singleShot(0, self._scroll_chat_bottom)
            # Terminal Agent may run multiple LLM rounds per user turn (when <Terminal> blocks
            # are executed and results are fed back). Each round should append a new assistant
            # message rather than overwriting the previous one.
            assistant_bubble = None

        w.signals.chat_partial.connect(on_partial)
        w.signals.chat_final.connect(on_final)
        w.finished.connect(lambda: setattr(self, "_worker", None))
        w.start()


def main() -> None:
    w = TerminalAgentWindow()
    raise SystemExit(w.exec())


if __name__ == "__main__":
    main()
