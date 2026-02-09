from __future__ import annotations

import pytest

from desktop_agent.evolving.schemas import validate_outcome


def test_validate_outcome_allows_null_interaction() -> None:
    out = validate_outcome(
        {
            "ok": True,
            "reason": "ok",
            "public_message": "x",
            "private_message": "",
            "actor_energy_delta": 1.0,
            "target_energy_delta": 0.0,
            "interaction": None,
        }
    )
    assert out["ok"] is True


def test_validate_outcome_rejects_non_object_interaction() -> None:
    with pytest.raises(ValueError):
        validate_outcome({"interaction": "bad"})

