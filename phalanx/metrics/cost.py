"""Cost metric collector."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from phalanx.core import Metric, compute_cost


class CostMetric(Metric):
    """Tracks per-slot cost with time-averaged summaries.

    The cost function is: c(x, o, j) = -throughput + usage_cost.
    """

    name = "CostMetric"

    def __init__(self):
        self.costs: List[float] = []
        self.queue_backlogs: List[float] = []

    def record(
        self,
        observations: np.ndarray,
        allocation: np.ndarray,
        perturbation: np.ndarray,
        state: Dict[str, Any],
    ) -> None:
        cost = state.get("cost", compute_cost(
            observations, allocation, perturbation, state.get("F", 4)))
        self.costs.append(cost)
        Q = state.get("Q", np.zeros(1))
        self.queue_backlogs.append(float(np.sum(Q)))

    def summarize(self, burn_in: int = 1000) -> Dict[str, float]:
        costs = np.array(self.costs)
        if len(costs) <= burn_in:
            subset = costs
        else:
            subset = costs[burn_in:]
        return {
            "mean": float(np.mean(subset)) if len(subset) > 0 else 0.0,
            "max": float(np.max(subset)) if len(subset) > 0 else 0.0,
            "std": float(np.std(subset)) if len(subset) > 0 else 0.0,
        }

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        costs = np.array(self.costs)
        return {
            "cost": costs,
            "time_avg_cost": np.cumsum(costs) / (np.arange(len(costs)) + 1),
            "queue_backlog": np.array(self.queue_backlogs),
        }

    def reset(self) -> None:
        self.costs = []
        self.queue_backlogs = []
