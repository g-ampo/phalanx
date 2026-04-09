"""Confidence interval computation using the t-distribution.

Provides both
scalar CI (for final averages) and array CI (for time-series with
shaded fill_between bands).
"""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
from scipy.stats import t as t_dist


def compute_ci(
    values: Union[Sequence[float], np.ndarray],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute a confidence interval using the t-distribution.

    Args:
        values: Sample values from independent seeds.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of ``(mean, ci_lower, ci_upper)``.
    """
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    n = len(values)
    if n < 2:
        return mean, mean, mean
    std = float(np.std(values, ddof=1))
    t_crit = float(t_dist.ppf((1 + confidence) / 2, df=n - 1))
    margin = t_crit * std / np.sqrt(n)
    return mean, mean - margin, mean + margin


def compute_ci_array(
    arrays: list[np.ndarray],
    confidence: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-time-step confidence intervals over multiple seed traces.

    Truncates all arrays to the shortest length, then computes pointwise
    mean and CI bounds.  Useful for plotting convergence with shaded bands.

    Args:
        arrays: List of 1-D arrays (one per seed), all of the same metric.
        confidence: Confidence level.

    Returns:
        Tuple of ``(mean_array, lower_array, upper_array)``.
    """
    min_len = min(len(a) for a in arrays)
    stacked = np.array([a[:min_len] for a in arrays])
    n = len(arrays)
    mean = np.mean(stacked, axis=0)
    if n < 2:
        return mean, mean.copy(), mean.copy()
    std = np.std(stacked, axis=0, ddof=1)
    t_crit = float(t_dist.ppf((1 + confidence) / 2, df=max(n - 1, 1)))
    margin = t_crit * std / np.sqrt(n)
    return mean, mean - margin, mean + margin
