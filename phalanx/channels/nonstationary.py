"""Non-stationary channel with slowly drifting statistics."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from phalanx.core import Channel
from phalanx.config import ChannelConfig
from phalanx.channels.gaussian import GaussianVARChannel


class NonstationaryChannel(Channel):
    """Non-stationary channel built on top of a Gaussian VAR(1) channel.

    Adds a sinusoidal drift with rate ``drift_rate`` along random
    per-channel directions.  This models slow time-varying mean
    channel quality.
    """

    def __init__(self, cfg: ChannelConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.M = cfg.M
        self.d_obs = cfg.d_obs
        self.t = 0
        self.base = GaussianVARChannel(cfg, rng)
        self.drift_dirs = rng.standard_normal((cfg.M, cfg.d_obs))
        for m in range(cfg.M):
            norm = np.linalg.norm(self.drift_dirs[m]) + 1e-8
            self.drift_dirs[m] /= norm

    def step(self) -> np.ndarray:
        obs = self.base.step()
        drift = np.sin(2 * np.pi * self.cfg.drift_rate * self.t)
        for m in range(self.M):
            obs[m] += drift * self.drift_dirs[m]
        self.t += 1
        return obs

    def save_state(self) -> Dict[str, Any]:
        saved = self.base.save_state()
        saved["ns_t"] = self.t
        return saved

    def restore_state(self, saved: Dict[str, Any]) -> None:
        self.base.restore_state(saved)
        self.t = saved["ns_t"]

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        self.base.reset(rng)
        self.t = 0
