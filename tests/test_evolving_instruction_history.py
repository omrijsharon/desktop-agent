from __future__ import annotations

from pathlib import Path

from desktop_agent.evolving.config import EvolvingConfig
from desktop_agent.evolving.engine import _append_instruction_history, _ensure_agent_files, build_paths


def test_instruction_history_appends(tmp_path: Path) -> None:
    cfg = EvolvingConfig()
    paths = build_paths(root=tmp_path, cfg=cfg)

    mem, ins, hist = _ensure_agent_files(paths, agent_id="a001")
    assert mem.exists()
    assert ins.exists()
    assert hist.exists()

    before = hist.read_text(encoding="utf-8")
    _append_instruction_history(
        hist_path=hist,
        title="test-edit",
        instructions_text="Add-on instructions:\n- New rule\n",
        max_chars=cfg.limits.max_instruction_history_chars,
    )
    after = hist.read_text(encoding="utf-8")
    assert len(after) > len(before)
    assert "test-edit" in after

