from __future__ import annotations

import os
import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CmdResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    cmd: list[str]


def _run(cmd: list[str], *, timeout_s: float | None = None, cwd: str | None = None) -> CmdResult:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
        )
        return CmdResult(
            ok=(p.returncode == 0),
            returncode=int(p.returncode),
            stdout=p.stdout or "",
            stderr=p.stderr or "",
            cmd=list(cmd),
        )
    except subprocess.TimeoutExpired as e:
        return CmdResult(ok=False, returncode=-1, stdout=str(e.stdout or ""), stderr=str(e.stderr or ""), cmd=list(cmd))


def _wifi_profile_xml(*, ssid: str, password: str) -> str:
    # WPA2-Personal profile. Good enough for typical AP/hotspot.
    esc_ssid = ssid.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    esc_pw = password.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{esc_ssid}</name>
  <SSIDConfig>
    <SSID>
      <name>{esc_ssid}</name>
    </SSID>
  </SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>manual</connectionMode>
  <MSM>
    <security>
      <authEncryption>
        <authentication>WPA2PSK</authentication>
        <encryption>AES</encryption>
        <useOneX>false</useOneX>
      </authEncryption>
      <sharedKey>
        <keyType>passPhrase</keyType>
        <protected>false</protected>
        <keyMaterial>{esc_pw}</keyMaterial>
      </sharedKey>
    </security>
  </MSM>
</WLANProfile>
"""


def connect_wifi_windows(*, ssid: str, password: str, timeout_s: float = 40.0) -> CmdResult:
    """Best-effort Wi-Fi connect on Windows using netsh.

    Adds/overwrites a profile for the SSID, then connects.
    """

    def _wait_connected() -> CmdResult:
        start = time.time()
        while time.time() - start < timeout_s:
            st = _run(["netsh", "wlan", "show", "interfaces"], timeout_s=10)
            txt = (st.stdout or "").lower()
            if st.ok and ssid.lower() in txt and "state" in txt and "connected" in txt:
                return st
            time.sleep(1.0)
        return CmdResult(ok=False, returncode=-1, stdout="", stderr=f"Timed out connecting to {ssid}", cmd=["netsh", "wlan"])

    tmp = Path(os.environ.get("TEMP") or ".").resolve() / f"wifi_{int(time.time())}_{ssid}.xml"
    tmp.write_text(_wifi_profile_xml(ssid=ssid, password=password), encoding="utf-8")
    try:
        # First try connecting using any existing profile (common if a profile already exists via policy).
        conn0 = _run(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}"], timeout_s=15)
        st0 = _wait_connected()
        if st0.ok:
            return st0

        add = _run(["netsh", "wlan", "add", "profile", f"filename={str(tmp)}", "user=current"], timeout_s=15)
        if not add.ok:
            msg = f"{add.stdout}\n{add.stderr}".lower()
            # If the profile already exists (e.g. in group policy), we can't overwrite it.
            # Proceed to connect using the existing profile.
            if "already exists" not in msg:
                return add
        conn = _run(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}"], timeout_s=15)
        if not conn.ok:
            return conn
        return _wait_connected()
    finally:
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


def list_wifi_ssids_windows() -> list[str]:
    """Best-effort SSID scan via `netsh wlan show networks mode=bssid`."""

    r = _run(["netsh", "wlan", "show", "networks", "mode=bssid"], timeout_s=20)
    if not r.ok:
        return []
    out: list[str] = []
    for line in (r.stdout or "").splitlines():
        s = line.strip()
        if not s.lower().startswith("ssid "):
            continue
        # Format: "SSID 1 : MyNetwork"
        if ":" not in s:
            continue
        name = s.split(":", 1)[1].strip()
        if name and name not in out:
            out.append(name)
    return out


def wait_for_tcp(host: str, port: int, *, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=2.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def has_internet(timeout_s: float = 4.0) -> bool:
    # Simple connectivity check (Cloudflare DNS over TCP).
    return wait_for_tcp("1.1.1.1", 53, timeout_s=timeout_s)


def ensure_empty_dir(path: str | Path) -> None:
    p = Path(path)
    if p.exists():
        for child in p.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except Exception:
                    pass
    else:
        p.mkdir(parents=True, exist_ok=True)


def archive_dir(src: str | Path, archive_root: str | Path) -> Optional[Path]:
    s = Path(src)
    if not s.exists() or not s.is_dir():
        return None
    if not any(s.iterdir()):
        return None
    ar = Path(archive_root)
    ar.mkdir(parents=True, exist_ok=True)
    dst = ar / time.strftime("%Y-%m-%d_%H-%M-%S")
    shutil.copytree(s, dst, dirs_exist_ok=True)
    return dst


def scp_pull_dir(*, user: str, host: str, remote_dir: str, local_parent: str, timeout_s: float = 90.0) -> CmdResult:
    # Copies remote_dir into local_parent (scp -r behaves like rsync-ish folder copy).
    cmd = [
        "scp",
        "-r",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
        f"{user}@{host}:{remote_dir}",
        str(local_parent),
    ]
    return _run(cmd, timeout_s=timeout_s)


def ssh_run(*, user: str, host: str, remote_cmd: str, x11: bool = False, timeout_s: float | None = None) -> CmdResult:
    cmd = ["ssh"]
    if x11:
        cmd.append("-X")
    # Avoid interactive "Are you sure you want to continue connecting" prompts.
    # `accept-new` adds new hosts to known_hosts but refuses changed keys.
    cmd += ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
    cmd.append(f"{user}@{host}")
    cmd.append(remote_cmd)
    return _run(cmd, timeout_s=timeout_s)


def ssh_popen(*, user: str, host: str, remote_cmd: str, x11: bool = False, cwd: str | None = None) -> subprocess.Popen[str]:
    cmd = ["ssh"]
    if x11:
        cmd.append("-X")
    cmd += ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
    cmd.append(f"{user}@{host}")
    cmd.append(remote_cmd)
    return subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def remote_dir_listing(
    *,
    user: str,
    host: str,
    remote_dir: str,
    timeout_s: float = 10.0,
) -> tuple[bool, list[str], str]:
    """Return (ok, entries, raw) for a remote directory listing.

    Uses `sh -lc` so `~` expands. Lists up to 50 entries.
    """

    rd = str(remote_dir or "").strip()
    if not rd:
        return False, [], ""

    # `ls -A` to include dotfiles, one per line. If dir is missing, returns empty.
    quoted = shlex.quote(rd)
    cmd = f"sh -lc 'ls -A1 -- {quoted} 2>/dev/null | head -n 50'"
    res = ssh_run(user=user, host=host, remote_cmd=cmd, x11=False, timeout_s=float(timeout_s))
    raw = (res.stdout or "").strip()
    if not res.ok:
        # Treat non-zero as "no listing" (missing dir/permission).
        return False, [], raw or (res.stderr or "").strip()
    entries = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return True, entries, raw
