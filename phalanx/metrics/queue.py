"""Queue metric: backlog tracking and stability analysis.

Tracks per-flow and total queue backlog, maximum queue length,
and queue stability indicators.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from phalanx.core import Metric


class QueueMetric(Metric):
    """Queue backlog tracker with stability diagnostics.

    Records total and per-flow backlogs.  The summarize method
    reports mean, max, and a stability indicator (time-averaged
    backlog bounded).

    Args:
        F: Number of flows / queues.
    """

    name = "QueueMetric"

    def __init__(self, F: int = 4) -> None:
        self.F = F
        self.total_backlogs: List[float] = []
        self.max_queues: List[float] = []
        self.per_flow_backlogs: List[np.ndarray] = []

    def record(
        self,
        observations: np.ndarray,
        allocation: np.ndarray,
        perturbation: np.ndarray,
        state: Dict[str, Any],
    ) -> None:
        """Record queue backlog for one slot."""
        Q = state.get("Q", np.zeros(self.F))
        self.total_backlogs.append(float(np.sum(Q)))
        self.max_queues.append(float(np.max(Q)))
        self.per_flow_backlogs.append(Q.copy())

    def summarize(self, burn_in: int = 1000) -> Dict[str, float]:
        """Return backlog summary after burn-in.

        Keys: ``"mean"`` (total backlog), ``"mean_total_backlog"``,
        ``"max_backlog"``, ``"stable"`` (1.0 if time-avg backlog < 100).
        """
        totals = np.array(self.total_backlogs)
        maxes = np.array(self.max_queues)
        if len(totals) <= burn_in:
            t_sub, m_sub = totals, maxes
        else:
            t_sub = totals[burn_in:]
            m_sub = maxes[burn_in:]

        mean_total = float(np.mean(t_sub)) if len(t_sub) > 0 else 0.0
        return {
            "mean": mean_total,
            "mean_total_backlog": mean_total,
            "max_backlog": float(np.max(m_sub)) if len(m_sub) > 0 else 0.0,
            "stable": 1.0 if mean_total < 100.0 else 0.0,
        }

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        """Return time-series arrays."""
        totals = np.array(self.total_backlogs)
        return {
            "total_backlog": totals,
            "time_avg_backlog": np.cumsum(totals) / (np.arange(len(totals)) + 1),
            "max_queue": np.array(self.max_queues),
        }

    def reset(self) -> None:
        self.total_backlogs = []
        self.max_queues = []
        self.per_flow_backlogs = []
