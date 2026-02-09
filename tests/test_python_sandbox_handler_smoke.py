from __future__ import annotations

import json
from pathlib import Path


def test_python_sandbox_handler_runs_simple_code(tmp_path: Path) -> None:
    from desktop_agent.tools import make_python_sandbox_handler  # noqa: WPS433

    h = make_python_sandbox_handler(base_dir=tmp_path, timeout_s=5.0)
    out = h({"code": "print('hi')\n"})
    data = json.loads(out)
    assert data["ok"] is True
    assert "hi" in (data.get("stdout") or "")

