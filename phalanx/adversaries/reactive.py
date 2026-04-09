"""Reactive (listen-then-jam) adversary.

Listen-then-jam adaptive adversary.

The adversary observes the scheduler's realised action and jams that
channel.  Adaptive-offline listen-then-jam adversary.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from phalanx.config import AdversaryConfig
from phalanx.core import Adversary


class ReactiveAdversary(Adversary):
    """Listen-then-jam adversary.

    Observes the scheduler's allocation and concentrates perturbation
    on the channel that received the most allocation.  Active with
    probability ``J_total`` each slot (J_total acts as an activity
    probability in [0, 1]).

    Optionally adds observation noise to the perceived allocation.

    Args:
        cfg: Adversary configuration, or keyword arguments.
        J_total: Activity probability / budget.
        observation_noise: Std of Gaussian noise on observed allocation.
    """

    name = "ReactiveAdversary"

    def __init__(
        self,
        cfg: AdversaryConfig | None = None,
        *,
        J_total: float = 0.3,
        observation_noise: float = 0.0,
    ):
        if cfg is not None:
            self.J_total = cfg.J_total
            self.noise = cfg.observation_noise
        else:
            self.J_total = J_total
            self.noise = observation_noise

    def attack(
        self,
        allocation: np.ndarray,
        observations: np.ndarray,
        state: Dict[str, Any],
    ) -> np.ndarray:
        """Jam the channel with the highest allocation.

        Returns:
            Perturbation vector with budget concentrated on the
            most-allocated channel.
        """
        rng: np.random.Generator = state.get("rng", np.random.default_rng())
        M = len(allocation)
        j = np.zeros(M)

        # Activity check
        if rng.random() >= self.J_total:
            return j

        # Observe allocation (possibly with noise)
        perceived = allocation.copy()
        if self.noise > 0:
            perceived += rng.normal(0, self.noise, size=M)
            perceived = np.maximum(perceived, 0.0)

        # Target highest-allocation channel
        target = int(np.argmax(perceived))
        j[target] = self.J_total
        return j

    def reset(self) -> None:
        pass
