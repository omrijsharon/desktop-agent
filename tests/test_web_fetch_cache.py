from __future__ import annotations

import json
import time
from pathlib import Path


def test_web_fetch_uses_cache_when_fresh(tmp_path: Path) -> None:
    from desktop_agent.tools import make_web_fetch_handler  # noqa: WPS433

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = "https://example.com/page"
    # Pre-seed cache with a fresh entry.
    cache_path = cache_dir / "ignored.json"
    # The handler computes its own key; write a matching file by calling once with no network
    # is not possible. Instead, mirror the key computation by importing internal helper.
    from desktop_agent import tools as tools_mod  # noqa: WPS433

    key = tools_mod._safe_cache_key(url)  # type: ignore[attr-defined]
    cache_path = cache_dir / f"{key}.json"
    cache_path.write_text(
        json.dumps(
            {
                "ts": time.time(),
                "result": {
                    "ok": True,
                    "url": url,
                    "final_url": url,
                    "status_code": 200,
                    "content_type": "text/plain",
                    "text": "hello",
                    "cached": False,
                    "truncated": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    h = make_web_fetch_handler(cache_dir=cache_dir, default_cache_ttl_s=3600.0)
    out = json.loads(h({"url": url}))
    assert out["ok"] is True
    assert out["text"] == "hello"
    assert out["cached"] is True

