"""Markov-modulated channels with discrete hidden states.

Markov-modulated channel with discrete hidden states.

Model:
    Each channel has K hidden states.  A shared latent state induces
    cross-channel correlation: with probability ``cross_correlation``
    a channel copies the shared state, otherwise it transitions
    independently.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from phalanx.config import ChannelConfig
from phalanx.core import Channel


class MarkovModulatedChannel(Channel):
    """Markov-modulated channels with discrete hidden states.

    Args:
        cfg: Channel configuration dataclass.
        rng: NumPy random generator for reproducibility.
    """

    def __init__(self, cfg: ChannelConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.M = cfg.M
        self.K = cfg.n_states
        self.d_obs = cfg.d_obs

        # Per-channel transition matrices (doubly stochastic perturbation of I)
        self.P: list[np.ndarray] = []
        for _m in range(cfg.M):
            P_m = np.eye(self.K) * (1 - cfg.transition_prob * (self.K - 1))
            P_m += cfg.transition_prob * (1 - np.eye(self.K))
            self.P.append(P_m)

        # Per-channel, per-state mean observations (higher states = better)
        self.means = rng.standard_normal((cfg.M, self.K, cfg.d_obs))
        for k in range(self.K):
            self.means[:, k, :] *= k + 1

        # Hidden states
        self.states = rng.integers(0, self.K, size=cfg.M)
        self.shared_state = int(rng.integers(0, self.K))

    def step(self) -> np.ndarray:
        """Advance channel by one slot.

        Returns:
            Observation array of shape ``(M, d_obs)``.
        """
        old_shared = self.shared_state
        self.shared_state = int(
            self.rng.choice(self.K, p=self.P[0][old_shared])
        )
        obs = np.zeros((self.M, self.d_obs))
        for m in range(self.M):
            if self.rng.random() < self.cfg.cross_correlation:
                self.states[m] = self.shared_state
            else:
                self.states[m] = int(
                    self.rng.choice(self.K, p=self.P[m][self.states[m]])
                )
            obs[m] = (
                self.means[m, self.states[m]]
                + self.rng.standard_normal(self.d_obs) * self.cfg.noise_std
            )
        return obs

    def get_hidden_state(self) -> np.ndarray:
        """Return discrete hidden states as float array."""
        return self.states.copy().astype(float)

    def save_state(self) -> Dict[str, Any]:
        return {
            "states": self.states.copy(),
            "shared_state": int(self.shared_state),
            "rng_state": self.rng.bit_generator.state,
        }

    def restore_state(self, saved: Dict[str, Any]) -> None:
        self.states = saved["states"].copy()
        self.shared_state = saved["shared_state"]
        self.rng.bit_generator.state = saved["rng_state"]

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        if rng is not None:
            self.rng = rng
        self.states = self.rng.integers(0, self.K, size=self.M)
        self.shared_state = int(self.rng.integers(0, self.K))
