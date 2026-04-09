"""Stackelberg adversary (optimal best-response).

Stackelberg best-response adversary.

The adversary is the Stackelberg follower.  Given the scheduler's
allocation x, it solves for the perturbation j that maximises the
DPP objective (minimises throughput / maximises cost).

For fixed x, the adversary's problem is a convex maximisation over
the budget polytope, solved by greedy vertex enumeration.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from phalanx.config import AdversaryConfig
from phalanx.core import (
    QUALITY_SCALE,
    USAGE_WEIGHT,
    Adversary,
    channel_quality,
    compute_cost,
    compute_service_rate,
)


class StackelbergAdversary(Adversary):
    """Optimal Stackelberg-follower adversary.

    Maximises the combined DPP objective over the budget polytope,
    accounting for both current and (if available) predicted
    observations.

    Args:
        cfg: Adversary configuration, or keyword arguments.
        J_total: Total perturbation budget.
    """

    name = "StackelbergAdversary"

    def __init__(
        self,
        cfg: AdversaryConfig | None = None,
        *,
        J_total: float = 0.5,
    ):
        if cfg is not None:
            self.J_total = cfg.J_total
        else:
            self.J_total = J_total

    def attack(
        self,
        allocation: np.ndarray,
        observations: np.ndarray,
        state: Dict[str, Any],
    ) -> np.ndarray:
        """Compute Stackelberg-follower best response.

        Uses the combined DPP objective (current + prediction-weighted
        terms) when ``state["obs_pred"]`` and ``state["alpha"]`` are
        available.
        """
        M = len(allocation)
        if self.J_total <= 0:
            return np.zeros(M)

        quality = channel_quality(observations)
        obs_pred = state.get("obs_pred", None)
        quality_pred = (
            channel_quality(obs_pred) if obs_pred is not None else quality
        )
        Q = state.get("Q", np.zeros(1))
        Q_sum = float(np.sum(Q))
        F = state.get("F", 4)
        V = state.get("V", 5.0)
        alpha = state.get("alpha", 0.0)

        j = np.zeros(M)
        remaining = self.J_total

        for _ in range(M):
            if remaining <= 1e-10:
                break
            eff_q = np.maximum(quality - j, 0)
            eff_q_pred = np.maximum(quality_pred - j, 0)

            # Marginal throughput reduction per unit jamming
            marginals_current = (
                allocation
                * QUALITY_SCALE
                / (1.0 + QUALITY_SCALE * eff_q)
            )
            marginals_current[eff_q <= 1e-10] = 0.0

            # Combined marginal: cost (V * throughput) + drift (Q/F * throughput)
            combined = (Q_sum / max(F, 1) + V) * marginals_current

            # Prediction-augmented term
            if alpha > 0 and obs_pred is not None:
                marginals_pred = (
                    allocation
                    * QUALITY_SCALE
                    / (1.0 + QUALITY_SCALE * eff_q_pred)
                )
                marginals_pred[eff_q_pred <= 1e-10] = 0.0
                combined += (
                    alpha * (Q_sum / max(F, 1)) * marginals_pred
                )

            if np.max(combined) < 1e-10:
                break

            target = int(np.argmax(combined))
            max_alloc_current = max(quality[target] - j[target], 0)
            max_alloc_pred = (
                max(quality_pred[target] - j[target], 0)
                if obs_pred is not None
                else max_alloc_current
            )
            alloc = min(remaining, max(max_alloc_current, max_alloc_pred))
            j[target] += alloc
            remaining -= alloc

        return j

    def reset(self) -> None:
        pass
