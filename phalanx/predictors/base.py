"""Abstract base class for observation predictors.

A Predictor maps a window of past observations to a latent representation
and produces a forecast of future observations.  The interface is
framework-agnostic (numpy only) so that downstream packages can wrap
PyTorch, JAX, or analytical models without imposing dependencies on
Phalanx itself.
"""
from __future__ import annotations

import abc

import numpy as np


class Predictor(abc.ABC):
    """Abstract observation predictor.

    Subclasses implement ``encode`` and ``predict`` to provide
    latent-space prediction for any channel type.  The optional
    ``confidence`` hook exposes predictor uncertainty so that
    downstream scheduling policies can gate or weight predictions.
    """

    @abc.abstractmethod
    def encode(self, obs_window: np.ndarray) -> np.ndarray:
        """Encode an observation window into a latent representation.

        Args:
            obs_window: Past observations, shape ``(W, d_obs_total)``
                where *W* is the window length and *d_obs_total* the
                flattened observation dimension (``M * d_obs``).

        Returns:
            Latent vector of shape ``(d_z,)``.
        """

    @abc.abstractmethod
    def predict(self, z: np.ndarray) -> np.ndarray:
        """Predict future observations from a latent state.

        Args:
            z: Latent vector of shape ``(d_z,)``.

        Returns:
            Predicted observation of shape ``(d_obs_total,)``.
        """

    def confidence(self, z: np.ndarray) -> float:
        """Return a scalar confidence score for the current prediction.

        Higher values indicate more reliable predictions.  The default
        implementation returns 1.0 (maximum confidence).  Override this
        to expose predictor uncertainty -- e.g. from a VAE's latent
        variance -- so that schedulers can adaptively gate predictions.

        Args:
            z: Latent vector of shape ``(d_z,)``.

        Returns:
            Non-negative confidence scalar.
        """
        return 1.0

    def reset(self) -> None:
        """Reset internal state between simulation runs."""
