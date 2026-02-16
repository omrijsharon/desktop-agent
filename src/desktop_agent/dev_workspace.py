"""desktop_agent.dev_workspace

Per-chat development workspace with a Python venv.

Goals:
- Give the model a persistent project directory to build apps (HTML/JS/CSS/Python).
- Provide Python-only execution (venv python + pip) without arbitrary shell access.
- Enforce strong path safety: no absolute paths, no drive letters, no `..`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


class WorkspaceError(RuntimeError):
    pass


_BAD_PATH_RE = re.compile(r"(^[a-zA-Z]:)|(^/)|(^\\\\)|(\.\.)")


def _safe_rel_path(p: str) -> Path:
    s = str(p or "").replace("\\", "/").strip()
    if not s:
        raise WorkspaceError("path must be a non-empty relative path")
    if _BAD_PATH_RE.search(s):
        raise WorkspaceError("path must be workspace-relative (no abs paths, no drive letters, no '..')")
    # Also prevent tilde expansion-like patterns.
    if s.startswith("~"):
        raise WorkspaceError("path must be workspace-relative (no '~')")
    # Normalize and ensure it stays relative.
    pp = Path(s)
    if pp.is_absolute():
        raise WorkspaceError("path must be relative")
    return pp


def workspace_root(*, repo_root: Path, chat_id: str) -> Path:
    return (repo_root / "chat_history" / "workspaces" / str(chat_id)).resolve()


def ensure_workspace(*, repo_root: Path, chat_id: str) -> Path:
    root = workspace_root(repo_root=repo_root, chat_id=chat_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def venv_python_path(ws_root: Path) -> Path:
    # Windows venv layout.
    return (ws_root / ".venv" / "Scripts" / "python.exe").resolve()


def venv_exists(ws_root: Path) -> bool:
    return venv_python_path(ws_root).exists()


def write_text(*, ws_root: Path, rel_path: str, content: str) -> Path:
    rp = _safe_rel_path(rel_path)
    p = (ws_root / rp).resolve()
    if not str(p).startswith(str(ws_root)):
        raise WorkspaceError("refusing to write outside workspace")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(content or ""), encoding="utf-8")
    return p


def append_text(*, ws_root: Path, rel_path: str, text: str) -> Path:
    rp = _safe_rel_path(rel_path)
    p = (ws_root / rp).resolve()
    if not str(p).startswith(str(ws_root)):
        raise WorkspaceError("refusing to append outside workspace")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(str(text or ""))
    return p


def read_text(*, ws_root: Path, rel_path: str, start_line: int = 1, max_lines: int = 200, max_chars: int = 20000) -> str:
    rp = _safe_rel_path(rel_path)
    p = (ws_root / rp).resolve()
    if not str(p).startswith(str(ws_root)):
        raise WorkspaceError("refusing to read outside workspace")
    if not p.exists() or not p.is_file():
        raise WorkspaceError(f"file not found: {rel_path}")
    start = max(1, int(start_line))
    n = max(1, int(max_lines))
    limit = max(1, int(max_chars))
    out: list[str] = []
    chars = 0
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if i < start:
                continue
            if len(out) >= n:
                break
            if chars >= limit:
                break
            take = line
            if chars + len(take) > limit:
                take = take[: max(0, limit - chars)]
            out.append(take)
            chars += len(take)
    return "".join(out)


def list_dir(*, ws_root: Path, rel_dir: str = ".", max_entries: int = 200) -> list[str]:
    rp = _safe_rel_path(rel_dir) if rel_dir not in {".", ""} else Path(".")
    p = (ws_root / rp).resolve()
    if not str(p).startswith(str(ws_root)):
        raise WorkspaceError("refusing to list outside workspace")
    if not p.exists() or not p.is_dir():
        raise WorkspaceError(f"dir not found: {rel_dir}")
    out: list[str] = []
    for child in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        if len(out) >= int(max_entries):
            break
        name = child.name + ("/" if child.is_dir() else "")
        out.append(str((rp / name).as_posix()))
    return out


def create_venv(*, ws_root: Path) -> Path:
    ws_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    # Create/overwrite: venv module overwrites directories in-place inconsistently;
    # keep it simple: if exists, no-op.
    if venv_exists(ws_root):
        return venv_python_path(ws_root)
    subprocess.run([py, "-m", "venv", ".venv"], cwd=str(ws_root), check=True)
    return venv_python_path(ws_root)


def run_venv_python(
    *,
    ws_root: Path,
    args: list[str],
    timeout_s: float = 60.0,
    env_extra: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    py = venv_python_path(ws_root)
    if not py.exists():
        raise WorkspaceError("venv is not initialized; call workspace_create_venv first")
    env = os.environ.copy()
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})
    return subprocess.run(
        [str(py), *args],
        cwd=str(ws_root),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=float(timeout_s),
        check=False,
        env=env,
    )


def pip_install(
    *,
    ws_root: Path,
    packages: Iterable[str],
    timeout_s: float = 15 * 60.0,
    log_path: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    pkgs = [str(p).strip() for p in packages if str(p).strip()]
    if not pkgs:
        raise WorkspaceError("packages must be a non-empty list of strings")

    # Ensure venv exists.
    if not venv_exists(ws_root):
        create_venv(ws_root=ws_root)

    # Write a simple log file so the user can follow progress from the terminal/editor.
    if log_path is None:
        log_path = (ws_root / "pip_install.log").resolve()
    try:
        log_path.write_text("", encoding="utf-8")
    except Exception:
        pass

    start = time.time()
    cp = run_venv_python(
        ws_root=ws_root,
        args=["-m", "pip", "install", "--no-input", *pkgs],
        timeout_s=float(timeout_s),
    )
    dur = time.time() - start
    try:
        log_path.write_text(cp.stdout or "", encoding="utf-8")
    except Exception:
        pass
    # Attach a bit of metadata as best-effort.
    try:
        meta = ws_root / "pip_install.meta.txt"
        meta.write_text(f"exit_code={cp.returncode}\nduration_s={dur:.3f}\n", encoding="utf-8")
    except Exception:
        pass
    return cp


@dataclass
class HttpServerState:
    port: int
    pid: int
    root: str


def http_server_start(*, ws_root: Path, port: int = 8000, root: str = ".", use_venv: bool = True) -> HttpServerState:
    prt = int(port)
    if prt <= 0 or prt > 65535:
        raise WorkspaceError("port must be 1..65535")
    rr = str(_safe_rel_path(root)) if root not in {".", ""} else "."
    root_dir = (ws_root / Path(rr)).resolve()
    if not str(root_dir).startswith(str(ws_root)):
        raise WorkspaceError("refusing to serve outside workspace")
    if not root_dir.exists() or not root_dir.is_dir():
        raise WorkspaceError(f"dir not found: {root}")

    if use_venv and not venv_exists(ws_root):
        create_venv(ws_root=ws_root)

    py = venv_python_path(ws_root) if use_venv else Path(sys.executable)
    if use_venv and not py.exists():
        raise WorkspaceError("venv python not found")

    # Store PID so stop can work across UI restarts (best-effort).
    proc = subprocess.Popen(
        [str(py), "-m", "http.server", str(prt)],
        cwd=str(root_dir),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)),
    )
    st = HttpServerState(port=prt, pid=int(proc.pid), root=str(rr))
    try:
        (ws_root / "http_server.json").write_text(
            f'{{"port":{st.port},"pid":{st.pid},"root":{json.dumps(st.root)} }}', encoding="utf-8"
        )
    except Exception:
        pass
    return st


def http_server_stop(*, ws_root: Path) -> bool:
    p = (ws_root / "http_server.json").resolve()
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        pid = int(d.get("pid") or 0)
    except Exception:
        pid = 0
    ok = False
    if pid > 0:
        try:
            # Windows-friendly best-effort.
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok = True
        except Exception:
            ok = False
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass
    return ok
