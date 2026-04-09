"""Baseline scheduling policies for comparison.

RoundRobin: equal allocation across all channels.
Greedy: proportional to current channel quality.
Random: uniformly random allocation on the simplex.
Oracle: future-knowledge upper bound (performance ceiling).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from phalanx.core import (
    QUALITY_SCALE,
    USAGE_WEIGHT,
    Scheduler,
    channel_quality,
    compute_cost,
    compute_service_rate,
    project_simplex,
)


class RoundRobin(Scheduler):
    """Equal allocation across all channels.

    Simple, fair, ignores channel quality and queues.
    """

    name = "RoundRobin"

    def decide(
        self, observations: np.ndarray, state: Dict[str, Any]
    ) -> np.ndarray:
        M = observations.shape[0]
        return np.ones(M) / M

    def reset(self) -> None:
        pass


class Greedy(Scheduler):
    """Allocate proportional to current channel quality.

    Myopic -- ignores queues, adversary, and future dynamics.
    """

    name = "Greedy"

    def decide(
        self, observations: np.ndarray, state: Dict[str, Any]
    ) -> np.ndarray:
        quality = channel_quality(observations)
        total = np.sum(quality)
        if total < 1e-10:
            M = observations.shape[0]
            return np.ones(M) / M
        return quality / total

    def reset(self) -> None:
        pass


class Random(Scheduler):
    """Uniformly random allocation on the simplex.

    Draws from the symmetric Dirichlet(1) distribution each slot.
    """

    name = "Random"

    def decide(
        self, observations: np.ndarray, state: Dict[str, Any]
    ) -> np.ndarray:
        rng: np.random.Generator = state.get("rng", np.random.default_rng())
        M = observations.shape[0]
        x = rng.exponential(1.0, size=M)
        return x / np.sum(x)

    def reset(self) -> None:
        pass


class Oracle(Scheduler):
    """Oracle scheduler: cost-optimal DPP with large V (performance ceiling).

    Approximates the optimal stationary cost ``c*`` from the Neely
    framework by running the DPP objective with a very large tradeoff
    parameter ``V_oracle``.  As ``V → ∞``, the DPP allocation converges
    to the cost-minimising allocation, sacrificing queue stability for
    the best achievable per-slot cost.

    This is the correct performance ceiling for DPP validation: any
    causal DPP scheduler with finite V achieves cost ``c* + O(1/V)``,
    so the Oracle's cost should be the asymptotic lower bound.

    Args:
        J_total: Adversary budget.
        F: Number of demand streams.
        lambda_mean: Mean arrival rate per stream.
        V_oracle: Internal V for cost optimisation (default 50).
        n_outer: Alternating best-response iterations.
        lr: Projected-gradient step size.
    """

    name = "Oracle"

    def __init__(
        self,
        J_total: float = 0.0,
        F: int = 4,
        lambda_mean: float = 0.35,
        V_oracle: float = 50.0,
        n_outer: int = 20,
        lr: float = 0.05,
    ):
        self.J_total = J_total
        self.F = F
        self.lambda_mean = lambda_mean
        self.V = V_oracle
        self.n_outer = n_outer
        self.lr = lr

    def decide(
        self, observations: np.ndarray, state: Dict[str, Any]
    ) -> np.ndarray:
        M = observations.shape[0]
        F = state.get("F", self.F)
        Q = state.get("Q", np.zeros(F))
        Q_sum = float(np.sum(Q))
        lambda_arr = state.get("arrivals", np.full(F, self.lambda_mean))

        quality = channel_quality(observations)

        # Initialize: quality-proportional
        x = np.maximum(quality, 1e-6)
        x = x / np.sum(x)

        best_x = x.copy()
        best_val = np.inf

        for iteration in range(self.n_outer):
            # Follower best response
            j = self._follower_br(x, quality, F, Q_sum, M)

            # Leader gradient step
            eff_q = np.maximum(quality - j, 0)
            log_rate = np.log1p(QUALITY_SCALE * eff_q)

            grad = np.zeros(M)
            for m in range(M):
                grad[m] = (
                    -(Q_sum / max(F, 1)) * log_rate[m]
                    - self.V * log_rate[m]
                    + self.V * USAGE_WEIGHT * quality[m]
                )

            effective_lr = self.lr / (1.0 + 0.02 * iteration)
            x = x - effective_lr * grad
            x = project_simplex(x)

            # Evaluate
            j_eval = self._follower_br(x, quality, F, Q_sum, M)
            mu = compute_service_rate(observations, x, j_eval, F)
            drift = float(np.sum(Q * (lambda_arr[:F] - mu)))
            cost = compute_cost(observations, x, j_eval, F)
            val = drift + self.V * cost

            if val < best_val:
                best_val = val
                best_x = x.copy()

        return best_x

    def _follower_br(
        self,
        x: np.ndarray,
        quality: np.ndarray,
        F: int,
        Q_sum: float,
        M: int,
    ) -> np.ndarray:
        """Greedy adversary best response on the DPP objective."""
        if self.J_total <= 0:
            return np.zeros(M)

        j = np.zeros(M)
        remaining = self.J_total

        for _ in range(M):
            if remaining <= 1e-10:
                break
            eff_q = np.maximum(quality - j, 0)
            marginals = x * QUALITY_SCALE / (1.0 + QUALITY_SCALE * eff_q)
            marginals[eff_q <= 1e-10] = 0.0
            combined = (Q_sum / max(F, 1) + self.V) * marginals

            if np.max(combined) < 1e-10:
                break

            target = int(np.argmax(combined))
            max_alloc = max(quality[target] - j[target], 0)
            alloc = min(remaining, max_alloc)
            j[target] += alloc
            remaining -= alloc

        return j

    def reset(self) -> None:
        pass
