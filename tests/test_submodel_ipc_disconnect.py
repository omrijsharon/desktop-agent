from __future__ import annotations

import os
import subprocess
import sys
import time
from multiprocessing.connection import Listener


def test_submodel_worker_exits_on_parent_disconnect() -> None:
    auth = os.urandom(16)
    listener = Listener(("127.0.0.1", 0), authkey=auth)
    host, port = listener.address

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "test")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(repo_root, "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "desktop_agent.submodel_worker",
        "--connect-host",
        str(host),
        "--connect-port",
        str(port),
        "--authkey",
        auth.hex(),
        "--model",
        "gpt-5.2-2025-12-11",
        "--system-prompt",
        "test",
        "--base-dir",
        repo_root,
    ]

    proc = subprocess.Popen(cmd, env=env, cwd=repo_root)
    try:
        conn = listener.accept()
        hello = conn.recv()
        assert isinstance(hello, dict) and hello.get("type") == "hello"

        # Simulate parent crash/close: drop connection without sending stop.
        conn.close()
        listener.close()

        for _ in range(60):
            if proc.poll() is not None:
                break
            time.sleep(0.05)

        assert proc.poll() == 0
    finally:
        try:
            listener.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()

