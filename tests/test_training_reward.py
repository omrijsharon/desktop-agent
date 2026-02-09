from __future__ import annotations

import math

import pytest

from desktop_agent.training.reward import gaussian_reward


def test_gaussian_reward_peaks_at_sweet_spot() -> None:
    r = gaussian_reward(solve_rate_pct=50, sweet_spot_pct=50, std_pct=10)
    assert r == pytest.approx(1.0)


def test_gaussian_reward_symmetric() -> None:
    left = gaussian_reward(solve_rate_pct=40, sweet_spot_pct=50, std_pct=10)
    right = gaussian_reward(solve_rate_pct=60, sweet_spot_pct=50, std_pct=10)
    assert left == pytest.approx(right)


def test_gaussian_reward_declines_away_from_mean() -> None:
    mid = gaussian_reward(solve_rate_pct=50, sweet_spot_pct=50, std_pct=10)
    far = gaussian_reward(solve_rate_pct=0, sweet_spot_pct=50, std_pct=10)
    assert mid > far
    assert 0.0 <= far <= 1.0


def test_gaussian_reward_rejects_non_positive_std() -> None:
    with pytest.raises(ValueError):
        gaussian_reward(solve_rate_pct=50, sweet_spot_pct=50, std_pct=0)
    with pytest.raises(ValueError):
        gaussian_reward(solve_rate_pct=50, sweet_spot_pct=50, std_pct=-1)


def test_gaussian_reward_rejects_nan() -> None:
    with pytest.raises(ValueError):
        gaussian_reward(solve_rate_pct=float("nan"), sweet_spot_pct=50, std_pct=10)
    with pytest.raises(ValueError):
        gaussian_reward(solve_rate_pct=50, sweet_spot_pct=float("nan"), std_pct=10)
    with pytest.raises(ValueError):
        gaussian_reward(solve_rate_pct=50, sweet_spot_pct=50, std_pct=float("nan"))

