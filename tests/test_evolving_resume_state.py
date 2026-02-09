from __future__ import annotations

import base64
import pickle
import random
from pathlib import Path

from desktop_agent.evolving.config import EvolvingConfig
from desktop_agent.evolving.engine import build_paths
from desktop_agent.evolving.state import init_population, load_state, save_json, state_to_jsonable


def test_state_roundtrip_includes_ephemeral_fields(tmp_path: Path) -> None:
    cfg = EvolvingConfig()
    paths = build_paths(root=tmp_path, cfg=cfg)
    rng = random.Random(0)
    st = init_population(n=2, start_age_days=0, start_energy=100.0, rng=rng)
    st.inbox["a001"] = ["hello"]
    st.last_private["a001"] = "note"
    st.conversation_partner["a001"] = "a002"
    st.current_step_actions = [{"actor_id": "a001", "action": {"op": "rest"}, "allowed_ops": None}]
    st.current_step_apply_index = 1
    st.phase = "day"
    st.day_remaining_s = 42.0
    st.rng_state_b64 = base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii")

    save_json(paths.state_json, state_to_jsonable(st))
    st2 = load_state(paths.state_json)
    assert st2 is not None
    assert st2.inbox["a001"] == ["hello"]
    assert st2.last_private["a001"] == "note"
    assert st2.conversation_partner["a001"] == "a002"
    assert st2.current_step_actions and st2.current_step_apply_index == 1
    assert st2.phase == "day"
    assert st2.day_remaining_s == 42.0
