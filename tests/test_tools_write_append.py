from __future__ import annotations

import json
from pathlib import Path

import pytest

from desktop_agent.tools import (
    ToolError,
    make_append_file_handler,
    make_write_file_handler,
)


def test_write_file_allowlisted(tmp_path: Path) -> None:
    h = make_write_file_handler(base_dir=tmp_path, allowed_globs=["memory.md"])
    out = json.loads(h({"path": "memory.md", "content": "hello"}))
    assert out["ok"] is True
    assert (tmp_path / "memory.md").read_text(encoding="utf-8") == "hello"


def test_write_file_blocks_non_allowlisted(tmp_path: Path) -> None:
    h = make_write_file_handler(base_dir=tmp_path, allowed_globs=["memory.md"])
    with pytest.raises(ToolError):
        h({"path": "secrets.txt", "content": "nope"})


def test_append_file_creates_and_appends(tmp_path: Path) -> None:
    h = make_append_file_handler(base_dir=tmp_path, allowed_globs=["memory.md"])
    json.loads(h({"path": "memory.md", "content": "a"}))
    json.loads(h({"path": "memory.md", "content": "b"}))
    assert (tmp_path / "memory.md").read_text(encoding="utf-8") == "a\nb\n"


def test_append_file_blocks_absolute(tmp_path: Path) -> None:
    h = make_append_file_handler(base_dir=tmp_path, allowed_globs=["memory.md"])
    with pytest.raises(ToolError):
        h({"path": str(Path(tmp_path).resolve()), "content": "x"})

