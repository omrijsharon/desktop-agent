from __future__ import annotations

import random
from pathlib import Path

from desktop_agent.evolving.config import EvolvingConfig, RelationshipConfig
from desktop_agent.evolving.engine import build_paths
from desktop_agent.evolving.state import init_population


def test_relationship_matrix_backup_written(tmp_path: Path) -> None:
    cfg = EvolvingConfig(relationships=RelationshipConfig(matrix_path="rel.json", backup_path="rel.bak.json"))
    st = init_population(n=3, start_age_days=0, start_energy=100.0, rng=random.Random(0))
    paths = build_paths(root=tmp_path, cfg=cfg)

    # First write: no backup yet.
    from desktop_agent.evolving.engine import _save_relationship

    _save_relationship(paths, st)
    assert paths.relationship_json.exists()
    assert not paths.relationship_backup_json.exists() or paths.relationship_backup_json.read_text(encoding="utf-8") == ""

    # Mutate and write again: backup should exist.
    st.relationship["a001"]["a002"] = 5.0
    _save_relationship(paths, st)
    assert paths.relationship_backup_json.exists()
    assert "a001" in paths.relationship_backup_json.read_text(encoding="utf-8")

