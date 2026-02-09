from __future__ import annotations

import math


def gaussian_reward(*, solve_rate_pct: float, sweet_spot_pct: float, std_pct: float) -> float:
    """Gaussian reward in [0, 1] with peak at sweet_spot_pct.

    reward = exp(-0.5 * ((p - mu) / sigma)^2)
    """

    if not isinstance(solve_rate_pct, (int, float)) or math.isnan(float(solve_rate_pct)):
        raise ValueError("solve_rate_pct must be a number")
    if not isinstance(sweet_spot_pct, (int, float)) or math.isnan(float(sweet_spot_pct)):
        raise ValueError("sweet_spot_pct must be a number")
    if not isinstance(std_pct, (int, float)) or math.isnan(float(std_pct)) or float(std_pct) <= 0.0:
        raise ValueError("std_pct must be a positive number")

    p = float(solve_rate_pct)
    mu = float(sweet_spot_pct)
    sigma = float(std_pct)

    z = (p - mu) / sigma
    return float(math.exp(-0.5 * (z * z)))

