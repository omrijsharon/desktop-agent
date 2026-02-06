from __future__ import annotations

import os


def test_chat_ui_import_smoke() -> None:
    # Avoid needing a real display server in CI/headless contexts.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from desktop_agent.chat_ui import ChatWindow  # noqa: WPS433

    w = ChatWindow()
    assert w is not None
    try:
        w.close()
    except Exception:
        pass
