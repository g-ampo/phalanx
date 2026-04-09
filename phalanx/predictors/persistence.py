"""Persistence (naive) predictor: predict the future as the present.

This is the simplest possible predictor baseline -- it returns the most
recent observation as the forecast for all future horizons.  Useful as a
lower bound on prediction quality.
"""
from __future__ import annotations

import numpy as np

from phalanx.predictors.base import Predictor


class PersistencePredictor(Predictor):
    """Predict future observation as the most recent observation.

    The latent space is the observation space itself (identity encoding).

    Args:
        d_obs_total: Flattened observation dimension (``M * d_obs``).
    """

    def __init__(self, d_obs_total: int):
        self.d_obs_total = d_obs_total

    def encode(self, obs_window: np.ndarray) -> np.ndarray:
        """Return the last observation in the window as the latent state.

        Args:
            obs_window: Shape ``(W, d_obs_total)``.

        Returns:
            Last observation, shape ``(d_obs_total,)``.
        """
        obs_window = np.asarray(obs_window)
        if obs_window.ndim == 1:
            return obs_window.copy()
        return obs_window[-1].copy()

    def predict(self, z: np.ndarray) -> np.ndarray:
        """Identity prediction: return the latent state unchanged.

        Args:
            z: Shape ``(d_obs_total,)``.

        Returns:
            Same array, shape ``(d_obs_total,)``.
        """
        return np.asarray(z).copy()

    def confidence(self, z: np.ndarray) -> float:
        """Persistence predictor has constant (maximum) confidence."""
        return 1.0

    def reset(self) -> None:
        pass
