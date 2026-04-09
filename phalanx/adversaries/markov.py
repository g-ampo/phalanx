"""Markov-modulated adversary with state-dependent jamming."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from phalanx.config import AdversaryConfig
from phalanx.core import Adversary


class MarkovAdversary(Adversary):
    """Markov-modulated adversary with hidden state transitions.

    The adversary has a finite number of hidden states.  Each state
    determines which channel is targeted.  State transitions follow a
    doubly stochastic matrix with switching probability p_switch.

    Args:
        cfg: Adversary configuration, or keyword arguments.
        J_total: Perturbation budget when active.
        n_states: Number of hidden jammer states.
        p_switch: Probability of switching state each slot.
    """

    name = "MarkovAdversary"

    def __init__(
        self,
        cfg: AdversaryConfig | None = None,
        *,
        J_total: float = 0.5,
        n_states: int = 3,
        p_switch: float = 0.1,
    ):
        if cfg is not None:
            self.J_total = cfg.J_total
            self.n_states = cfg.n_jammer_states
            self.p_switch = cfg.p_switch
        else:
            self.J_total = J_total
            self.n_states = n_states
            self.p_switch = p_switch
        self._state = 0

    def attack(
        self,
        allocation: np.ndarray,
        observations: np.ndarray,
        state: Dict[str, Any],
    ) -> np.ndarray:
        rng: np.random.Generator = state.get("rng", np.random.default_rng())
        M = len(allocation)

        # State transition
        if rng.random() < self.p_switch:
            self._state = int(rng.integers(0, max(self.n_states, 1)))

        j = np.zeros(M)
        if self.J_total > 0 and M > 0:
            target = self._state % M
            j[target] = self.J_total
        return j

    def reset(self) -> None:
        self._state = 0
