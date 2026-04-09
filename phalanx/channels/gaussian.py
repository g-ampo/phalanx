"""Correlated Gaussian channels via VAR(1) process.

Correlated Gaussian VAR(1) channel model with full
save/restore and ABC compliance.

Model:
    Hidden state:  s(t+1) = A s(t) + w(t),  w ~ N(0, 0.1^2 I)
    Observation:   o_m(t) = eta * C_m s(t) + v_m(t),  v ~ N(0, sigma^2 I)

Cross-channel correlation is introduced via the observation matrices
C_m sharing a common component.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from phalanx.config import ChannelConfig
from phalanx.core import Channel


class GaussianVARChannel(Channel):
    """Correlated Gaussian channels via VAR(1) process.

    Args:
        cfg: Channel configuration dataclass.
        rng: NumPy random generator for reproducibility.
    """

    def __init__(self, cfg: ChannelConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.M = cfg.M
        self.d_obs = cfg.d_obs
        self.d_hidden = cfg.M * 2

        # Diagonal AR(1) transition: temporal correlation = var_coefficient^k
        self.A = cfg.var_coefficient * np.eye(self.d_hidden)

        # Observation matrices with cross-channel correlation
        self.C: list[np.ndarray] = []
        for m in range(cfg.M):
            C_m = rng.standard_normal((cfg.d_obs, self.d_hidden))
            if m > 0:
                C_m += cfg.cross_correlation * self.C[0]
            self.C.append(C_m / np.linalg.norm(C_m, ord=2))

        self.state = rng.standard_normal(self.d_hidden) * 0.1

    def step(self) -> np.ndarray:
        """Advance channel by one slot.

        Returns:
            Observation array of shape ``(M, d_obs)``.
        """
        noise_w = self.rng.standard_normal(self.d_hidden) * 0.1
        self.state = self.A @ self.state + noise_w

        obs = np.zeros((self.M, self.d_obs))
        for m in range(self.M):
            noise_v = self.rng.standard_normal(self.d_obs) * self.cfg.noise_std
            obs[m] = self.cfg.quality_scale * (self.C[m] @ self.state) + noise_v
        return obs

    def get_hidden_state(self) -> np.ndarray:
        """Return a copy of the hidden state vector."""
        return self.state.copy()

    def save_state(self) -> Dict[str, Any]:
        """Snapshot channel state and RNG for later restore."""
        return {
            "state": self.state.copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def restore_state(self, saved: Dict[str, Any]) -> None:
        """Restore a previously saved snapshot."""
        self.state = saved["state"].copy()
        self.rng.bit_generator.state = saved["rng_state"]

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """Reset to initial conditions."""
        if rng is not None:
            self.rng = rng
        self.state = self.rng.standard_normal(self.d_hidden) * 0.1
