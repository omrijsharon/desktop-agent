from __future__ import annotations

from pathlib import Path


def test_model_caps_store_filters_and_persists(tmp_path: Path) -> None:
    from desktop_agent.model_caps import ModelCapsStore  # noqa: WPS433

    p = tmp_path / "caps.json"
    store = ModelCapsStore(path=p)
    store.mark_unsupported(model="m1", param="top_p")
    store.mark_unsupported(model="m1", param="temperature")

    # New instance loads persisted caps.
    store2 = ModelCapsStore(path=p)
    kw = {"top_p": 0.9, "temperature": 0.7, "max_output_tokens": 10}
    out = store2.filter_create_kwargs(model="m1", create_kwargs=kw)
    assert "top_p" not in out
    assert "temperature" not in out
    assert out["max_output_tokens"] == 10
