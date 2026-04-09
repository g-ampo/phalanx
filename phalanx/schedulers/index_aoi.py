"""AoI-specific scheduling policies.

AoIIndexPolicy:  per-source index for AoI minimisation.
EpsilonEqualization:  mixed-strategy scheduling under adversarial jamming.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from phalanx.core import Scheduler


class AoIIndexPolicy(Scheduler):
    """Per-source AoI index policy.

    Computes a priority index per source based on current age, success
    probability, energy cost, and a tradeoff parameter V.  Selects the
    source with the highest index each slot.

    Args:
        V: DPP tradeoff parameter.
        N_sources: Number of AoI sources.
        p_succ_base: Base success probability.
        e_tx: Energy cost per transmission.
    """

    name = "AoI Index"

    def __init__(
        self,
        V: float = 10.0,
        N_sources: int = 4,
        p_succ_base: float = 0.8,
        e_tx: float = 1.0,
    ):
        self.V = V
        self.N = N_sources
        self.p_succ = p_succ_base
        self.e_tx = e_tx

    def decide(
        self, observations: np.ndarray, state: Dict[str, Any]
    ) -> np.ndarray:
        M = observations.shape[0]
        N = min(self.N, M)

        ages = state.get("ages", np.ones(N))
        weights = state.get("weights", np.ones(N))
        energy_queues = state.get("energy_queues", np.zeros(N))

        indices = np.zeros(N)
        for n in range(N):
            d = ages[n]
            aoi_benefit = (
                weights[n] * self.p_succ * d * (d / 2.0 + 1.0 + self.V)
            )
            energy_cost = energy_queues[n] * self.e_tx
            indices[n] = aoi_benefit - energy_cost

        x = np.zeros(M)
        best = int(np.argmax(indices))
        x[best] = 1.0
        return x

    def reset(self) -> None:
        pass


class EpsilonEqualization(Scheduler):
    """Mixed-strategy AoI scheduling under adversarial jamming.

    Blends age-proportional scheduling with a uniform component
    controlled by a mixing parameter epsilon, balancing exploitation
    of age state with robustness to adversarial observation.

    Args:
        pi_J: Jammer activity probability.
        p_succ: Common unjammed success probability.
        d: Jamming damage (success probability reduction under jamming).
    """

    name = "Eps-Equalization"

    def __init__(
        self,
        pi_J: float = 0.3,
        p_succ: float = 0.8,
        d: float = 0.5,
    ):
        self.pi_J = pi_J
        self.p_succ = p_succ
        self.d = d

    def _compute_epsilon(self, N: int) -> float:
        """Optimal eps* clamped to (0, 1)."""
        if self.pi_J <= 0 or self.d <= 0:
            return 1.0  # no jammer -> pure uniform
        eps = np.sqrt(N * self.p_succ**2 / (2.0 * self.pi_J * self.d))
        return float(np.clip(eps, 1e-6, 1.0))

    def decide(
        self, observations: np.ndarray, state: Dict[str, Any]
    ) -> np.ndarray:
        rng: np.random.Generator = state.get("rng", np.random.default_rng())
        M = observations.shape[0]
        ages = state.get("ages", np.ones(M))
        N = len(ages)
        eps = self._compute_epsilon(N)

        ages_safe = np.maximum(ages, 1.0)
        inv_ages = 1.0 / ages_safe
        total_inv = inv_ages.sum()
        if total_inv <= 0:
            x = np.zeros(M)
            x[int(rng.integers(0, M))] = 1.0
            return x

        sigma_eq = inv_ages / total_inv
        sigma_unif = np.full(N, 1.0 / N)
        sigma = (1.0 - eps) * sigma_eq + eps * sigma_unif

        # Normalise for numerical safety
        sigma = np.maximum(sigma, 0.0)
        sigma /= sigma.sum()

        # Sample and return one-hot
        chosen = int(rng.choice(N, p=sigma))
        x = np.zeros(M)
        x[min(chosen, M - 1)] = 1.0
        return x

    def reset(self) -> None:
        pass
