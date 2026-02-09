from __future__ import annotations

from pathlib import Path

import pytest

from desktop_agent.evolving.config import EvolvingConfig
from desktop_agent.evolving.state import init_population
from desktop_agent.evolving.tensorboard_logger import TensorboardLogger, tensorboard_available


@pytest.mark.skipif(not tensorboard_available(), reason="tensorboard not installed")
def test_tensorboard_logger_writes_events(tmp_path: Path) -> None:
    import random

    cfg = EvolvingConfig()
    st = init_population(n=3, start_age_days=0, start_energy=100.0, rng=random.Random(0))

    logdir = tmp_path / "tb"
    tb = TensorboardLogger(logdir=logdir)
    try:
        tb.log_day(cfg=cfg, st=st, day_stats={"coop": 1, "comp": 0, "neutral": 0, "op_counts": {"rest": 3}}, day_index=1, world_root=tmp_path)
    finally:
        tb.close()

    # TensorBoard event files start with "events.out.tfevents".
    files = list(logdir.glob("events.out.tfevents.*"))
    assert files, "expected at least one tensorboard event file"

