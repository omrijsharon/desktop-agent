from __future__ import annotations

import json
from pathlib import Path


def test_create_and_register_analysis_tool_registers_callable(tmp_path: Path) -> None:
    from desktop_agent.tools import (  # noqa: WPS433
        ToolRegistry,
        make_create_and_register_analysis_tool_handler,
        make_python_sandbox_handler,
    )

    registry = ToolRegistry(tools={}, handlers={})
    py = make_python_sandbox_handler(base_dir=tmp_path, timeout_s=5.0)
    create = make_create_and_register_analysis_tool_handler(registry=registry, python_sandbox_handler=py, scripts_root=tmp_path / "scripts")

    rep = json.loads(
        create(
            {
                "tool_name": "analyze_one",
                "description": "Print args.x",
                "python_code": "print(args.get('x'))\n",
            }
        )
    )
    assert rep["ok"] is True
    assert "analyze_one" in registry.handlers

    out = registry.handlers["analyze_one"]({"args": {"x": 123}})
    data = json.loads(out)
    assert data["ok"] is True
    assert "123" in (data.get("stdout") or "")

