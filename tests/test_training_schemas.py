from __future__ import annotations

import pytest

from desktop_agent.training.schemas import validate_actor_output, validate_riddler_output, validate_solver_output


def test_validate_riddler_output_requires_fields() -> None:
    with pytest.raises(ValueError):
        validate_riddler_output({})


def test_validate_actor_output_requires_exact_profile_count() -> None:
    payload = {
        "solver_instruction_profiles": ["a", "b"],
        "actor_summary": "s",
        "next_instruction_hypotheses": ["h1"],
    }
    with pytest.raises(ValueError):
        validate_actor_output(payload, expected_solver_count=3)


def test_validate_solver_output_accepts_minimal() -> None:
    payload = {
        "final_answer": "answer",
        "derivation": "because ...",
        "confidence": 0.5,
        "checks": ["check"],
        "help_request": {"needed": False, "what_i_tried": "", "where_i_am_stuck": ""},
    }
    out = validate_solver_output(payload)
    assert out["final_answer"] == "answer"
    assert out["help_request"]["needed"] is False
