"""Age-of-Information metric collector."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from phalanx.core import Metric


class AoIMetric(Metric):
    """Tracks per-source Age of Information.

    AoI for source n:
    - Increments by 1 each slot.
    - Resets to 1 on successful delivery (source was selected and
      transmission succeeded).

    The metric expects state["selected_source"] (int) and
    state["tx_success"] (bool) to be set by the simulation loop.
    """

    name = "AoIMetric"

    def __init__(self, N_sources: int = 4):
        self.N = N_sources
        self.ages: np.ndarray = np.ones(N_sources)
        self.age_history: List[np.ndarray] = []

    def record(
        self,
        observations: np.ndarray,
        allocation: np.ndarray,
        perturbation: np.ndarray,
        state: Dict[str, Any],
    ) -> None:
        selected = state.get("selected_source", -1)
        success = state.get("tx_success", False)

        # Increment all ages
        self.ages += 1

        # Reset on successful delivery
        if 0 <= selected < self.N and success:
            self.ages[selected] = 1

        self.age_history.append(self.ages.copy())

    def summarize(self, burn_in: int = 1000) -> Dict[str, float]:
        if not self.age_history:
            return {"mean": 0.0, "max": 0.0, "per_source_mean": 0.0}

        history = np.array(self.age_history)
        if len(history) <= burn_in:
            subset = history
        else:
            subset = history[burn_in:]

        per_source_mean = np.mean(subset, axis=0)
        return {
            "mean": float(np.mean(per_source_mean)),
            "max": float(np.max(per_source_mean)),
            "per_source_mean": float(np.mean(per_source_mean)),
        }

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        history = np.array(self.age_history)
        return {
            "ages": history,
            "mean_age": np.mean(history, axis=1),
        }

    def reset(self) -> None:
        self.ages = np.ones(self.N)
        self.age_history = []
