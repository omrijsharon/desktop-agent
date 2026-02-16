from pathlib import Path

import pytest


def test_dev_workspace_rejects_absolute_and_parent_paths(tmp_path: Path) -> None:
    from desktop_agent.dev_workspace import WorkspaceError, ensure_workspace, write_text

    ws = ensure_workspace(repo_root=tmp_path, chat_id="chat1")

    with pytest.raises(WorkspaceError):
        write_text(ws_root=ws, rel_path="..\\x.txt", content="no")
    with pytest.raises(WorkspaceError):
        write_text(ws_root=ws, rel_path="../x.txt", content="no")
    with pytest.raises(WorkspaceError):
        write_text(ws_root=ws, rel_path="C:\\x.txt", content="no")
    with pytest.raises(WorkspaceError):
        write_text(ws_root=ws, rel_path="/x.txt", content="no")


def test_dev_workspace_write_read_roundtrip(tmp_path: Path) -> None:
    from desktop_agent.dev_workspace import ensure_workspace, read_text, write_text

    ws = ensure_workspace(repo_root=tmp_path, chat_id="chat1")
    write_text(ws_root=ws, rel_path="index.html", content="hello\nworld\n")
    out = read_text(ws_root=ws, rel_path="index.html", start_line=2, max_lines=1, max_chars=100)
    assert out.strip() == "world"

