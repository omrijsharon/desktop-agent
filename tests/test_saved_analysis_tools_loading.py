from __future__ import annotations

import json
from pathlib import Path


def test_register_saved_analysis_tools_registers_latest(tmp_path) -> None:
    from desktop_agent.tools import ToolRegistry, list_saved_analysis_tool_names, register_saved_analysis_tools

    scripts_root = tmp_path / "analysis_tools"
    tool_dir = scripts_root / "ekf_metrics"
    v1 = tool_dir / "20260101T000000Z"
    v2 = tool_dir / "20260201T000000Z"
    v1.mkdir(parents=True)
    v2.mkdir(parents=True)

    (v1 / "tool.json").write_text(json.dumps({"tool_name": "ekf_metrics", "description": "v1", "created_at": "20260101T000000Z"}))
    (v1 / "ekf_metrics.py").write_text("print('v1')\n", encoding="utf-8")
    (v2 / "tool.json").write_text(json.dumps({"tool_name": "ekf_metrics", "description": "v2", "created_at": "20260201T000000Z"}))
    (v2 / "ekf_metrics.py").write_text("print('v2')\n", encoding="utf-8")

    assert list_saved_analysis_tool_names(scripts_root=scripts_root) == ["ekf_metrics"]

    seen: dict[str, str] = {}

    def fake_py(payload: dict) -> str:
        seen["code"] = payload.get("code", "")
        return json.dumps({"ok": True})

    reg = ToolRegistry(tools={}, handlers={})
    loaded = register_saved_analysis_tools(registry=reg, python_sandbox_handler=fake_py, scripts_root=scripts_root)
    assert loaded == ["ekf_metrics"]
    assert "ekf_metrics" in reg.tools
    assert "ekf_metrics" in reg.handlers

    out = reg.handlers["ekf_metrics"]({"args": {"x": 1}})
    assert json.loads(out)["ok"] is True
    # Should use newest version (v2).
    assert "print('v2')" in seen["code"]
    assert "print('v1')" not in seen["code"]

