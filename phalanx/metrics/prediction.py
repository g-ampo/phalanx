"""Prediction error metric collector.

Tracks the accuracy of observation predictions by comparing predicted
observations (from a :class:`~phalanx.predictors.base.Predictor`) against
actual channel observations at each time slot.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from phalanx.core import Metric


class PredictionErrorMetric(Metric):
    """Tracks prediction error (NMSE, RMSE) over the simulation.

    At each slot, if ``state["obs_pred"]`` is present, the metric
    records the squared error between the predicted and actual
    observations.  Slots without predictions are silently skipped.
    """

    name = "PredictionErrorMetric"

    def __init__(self):
        self.squared_errors: List[float] = []
        self.obs_sq_norms: List[float] = []

    def record(
        self,
        observations: np.ndarray,
        allocation: np.ndarray,
        perturbation: np.ndarray,
        state: Dict[str, Any],
    ) -> None:
        obs_pred = state.get("obs_pred")
        if obs_pred is None:
            return
        obs_flat = observations.ravel()
        pred_flat = np.asarray(obs_pred).ravel()
        self.squared_errors.append(float(np.sum((obs_flat - pred_flat) ** 2)))
        self.obs_sq_norms.append(float(np.sum(obs_flat ** 2)))

    def summarize(self, burn_in: int = 1000) -> Dict[str, float]:
        if len(self.squared_errors) == 0:
            return {"nmse": 0.0, "rmse": 0.0, "mean": 0.0}
        se = np.array(self.squared_errors)
        norms = np.array(self.obs_sq_norms)
        if len(se) > burn_in:
            se = se[burn_in:]
            norms = norms[burn_in:]
        mean_se = float(np.mean(se))
        mean_norm = float(np.mean(norms))
        nmse = mean_se / mean_norm if mean_norm > 0 else float("inf")
        rmse = float(np.sqrt(mean_se))
        return {"nmse": nmse, "rmse": rmse, "mean": nmse}

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        if len(self.squared_errors) == 0:
            return {}
        se = np.array(self.squared_errors)
        norms = np.array(self.obs_sq_norms)
        safe_norms = np.where(norms > 0, norms, 1.0)
        return {
            "prediction_mse": se,
            "prediction_nmse": se / safe_norms,
        }

    def reset(self) -> None:
        self.squared_errors = []
        self.obs_sq_norms = []
