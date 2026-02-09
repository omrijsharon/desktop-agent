from __future__ import annotations

import json
from pathlib import Path


def test_python_sandbox_can_copy_directory(tmp_path: Path) -> None:
    from desktop_agent.tools import make_python_sandbox_handler  # noqa: WPS433

    # Arrange a fake repo tree under tmp_path
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "a.jsonl").write_text('{"x": 1}\n', encoding="utf-8")

    h = make_python_sandbox_handler(base_dir=tmp_path, timeout_s=5.0)
    out = h(
        {
            "code": "print(open('logs/a.jsonl','r',encoding='utf-8').read().strip())\n",
            "copy_from_repo": ["logs"],
        }
    )
    data = json.loads(out)
    assert data["ok"] is True
    assert "a.jsonl" in "".join(data.get("copied_from_repo") or [])
    assert '{"x": 1}' in (data.get("stdout") or "")

