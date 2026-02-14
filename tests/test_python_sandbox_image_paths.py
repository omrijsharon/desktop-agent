from __future__ import annotations

import json
from pathlib import Path


def test_python_sandbox_returns_image_paths(tmp_path) -> None:
    from desktop_agent.tools import make_python_sandbox_handler

    h = make_python_sandbox_handler(base_dir=tmp_path, timeout_s=20.0)
    code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([0, 1, 2], [0, 1, 0])
plt.title("test")
plt.savefig("plot.png", dpi=120)
print("ok")
"""
    out = json.loads(h({"code": code}))
    assert out["ok"] is True
    assert out["exit_code"] == 0
    imgs = out.get("image_paths")
    assert isinstance(imgs, list)
    assert any(isinstance(p, str) and p.endswith("/plot.png") for p in imgs)
    # File actually exists under base_dir
    p = next(p for p in imgs if isinstance(p, str) and p.endswith("/plot.png"))
    assert (tmp_path / Path(p)).exists()

