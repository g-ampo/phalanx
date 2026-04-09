"""Lyapunov drift-plus-penalty schedulers.


LyapunovDPP:  full Stackelberg-robust DPP with alternating best-response.
LyapunovOnly: DPP minimization without adversary modelling (j = 0).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from phalanx.config import SchedulerConfig
from phalanx.core import (
    QUALITY_SCALE,
    USAGE_WEIGHT,
    Scheduler,
    channel_quality,
    compute_cost,
    compute_service_rate,
    project_simplex,
)


class LyapunovDPP(Scheduler):
    """Lyapunov drift-plus-penalty scheduler with Stackelberg robustness.

    At each slot, solves:

        min_x  max_j  V * c(x, o, j) + Q^T (lambda - mu(x, o, j))

    via alternating best-response (leader projected gradient descent,
    follower greedy best response on the budget polytope).

    Args:
        cfg: Scheduler configuration dataclass.  If *None*, keyword
            arguments are used directly.
        V: DPP tradeoff parameter.
        F: Number of demand streams.
        alpha: Prediction weight in [0, 1].
        J_total: Adversary budget (0 disables adversary modelling).
        lambda_mean: Mean arrival rate per stream.
        n_outer: Alternating best-response iterations.
        lr: Leader projected-gradient step size.
    """

    name = "LyapunovDPP"

    def __init__(
        self,
        cfg: Optional[SchedulerConfig] = None,
        *,
        V: float = 5.0,
        F: int = 4,
        alpha: float = 0.3,
        J_total: float = 0.0,
        lambda_mean: float = 0.35,
        n_outer: int = 15,
        lr: float = 0.05,
    ):
        if cfg is not None:
            self.V = cfg.V
            self.F = cfg.F
            self.alpha = cfg.alpha
            self.lambda_mean = cfg.lambda_mean
            self.n_outer = n_outer
            self.lr = lr
        else:
            self.V = V
            self.F = F
            self.alpha = alpha
            self.lambda_mean = lambda_mean
            self.n_outer = n_outer
            self.lr = lr
        self.J_total = J_total

    def decide(
        self, observations: np.ndarray, state: Dict[str, Any]
    ) -> np.ndarray:
        """Compute Stackelberg-robust DPP allocation.

        Args:
            observations: Shape ``(M, d_obs)``.
            state: Must contain ``"Q"`` (queue backlogs) and ``"F"``.

        Returns:
            Allocation vector on the simplex, shape ``(M,)``.
        """
        M = observations.shape[0]
        F = state.get("F", self.F)
        Q = state.get("Q", np.zeros(F))
        Q_sum = float(np.sum(Q))
        lambda_arr = state.get(
            "arrivals", np.full(F, self.lambda_mean)
        )
        obs_pred = state.get("obs_pred", None)

        quality = channel_quality(observations)

        # Initialize: quality-proportional
        x = np.maximum(quality, 1e-6)
        x = x / np.sum(x)

        best_x = x.copy()
        best_val = np.inf
        best_j = np.zeros(M)

        for iteration in range(self.n_outer):
            # Follower best response
            j = self._follower_best_response(
                x, quality, observations, obs_pred, Q, F
            )

            # Leader gradient step
            grad = self._dpp_gradient(
                x, quality, j, Q_sum, F, observations, obs_pred
            )
            effective_lr = self.lr / (1.0 + 0.02 * iteration)
            x = x - effective_lr * grad
            x = project_simplex(x)

            # Evaluate and track best
            j_eval = self._follower_best_response(
                x, quality, observations, obs_pred, Q, F
            )
            val = self._evaluate_dpp(
                x, j_eval, Q, observations, obs_pred, lambda_arr, F
            )
            if val < best_val:
                best_val = val
                best_x = x.copy()
                best_j = j_eval.copy()

        return best_x

    def _follower_best_response(
        self,
        x: np.ndarray,
        quality: np.ndarray,
        obs: np.ndarray,
        obs_pred: Optional[np.ndarray],
        Q: np.ndarray,
        F: int,
    ) -> np.ndarray:
        """Greedy adversary best response on the combined DPP objective.

        Concentrate budget on channels with highest marginal impact on the
        joint current + prediction DPP terms.
        """
        M = len(x)
        if self.J_total <= 0:
            return np.zeros(M)

        quality_pred = (
            channel_quality(obs_pred) if obs_pred is not None else quality
        )
        Q_sum = float(np.sum(Q))
        j = np.zeros(M)
        remaining = self.J_total

        for _ in range(M):
            if remaining <= 1e-10:
                break
            eff_q = np.maximum(quality - j, 0)
            eff_q_pred = np.maximum(quality_pred - j, 0)

            marginals_current = x * QUALITY_SCALE / (1.0 + QUALITY_SCALE * eff_q)
            marginals_current[eff_q <= 1e-10] = 0.0

            combined = (Q_sum / max(F, 1) + self.V) * marginals_current

            if self.alpha > 0 and obs_pred is not None:
                marginals_pred = (
                    x * QUALITY_SCALE / (1.0 + QUALITY_SCALE * eff_q_pred)
                )
                marginals_pred[eff_q_pred <= 1e-10] = 0.0
                combined += self.alpha * (Q_sum / max(F, 1)) * marginals_pred

            if np.max(combined) < 1e-10:
                break

            target = int(np.argmax(combined))
            max_alloc = max(quality[target] - j[target], 0)
            if obs_pred is not None:
                max_alloc = max(
                    max_alloc, max(quality_pred[target] - j[target], 0)
                )
            alloc = min(remaining, max_alloc)
            j[target] += alloc
            remaining -= alloc

        return j

    def _dpp_gradient(
        self,
        x: np.ndarray,
        quality: np.ndarray,
        j: np.ndarray,
        Q_sum: float,
        F: int,
        obs: np.ndarray,
        obs_pred: Optional[np.ndarray],
    ) -> np.ndarray:
        """Analytical gradient of DPP w.r.t. x, holding j fixed."""
        eff_q = np.maximum(quality - j, 0)
        log_rate = np.log1p(QUALITY_SCALE * eff_q)
        M = len(x)

        grad = np.zeros(M)
        for m in range(M):
            grad[m] = (
                -(Q_sum / max(F, 1)) * log_rate[m]
                - self.V * log_rate[m]
                + self.V * USAGE_WEIGHT * quality[m]
            )

        if self.alpha > 0 and obs_pred is not None:
            quality_pred = channel_quality(obs_pred)
            eff_q_pred = np.maximum(quality_pred - j, 0)
            log_rate_pred = np.log1p(QUALITY_SCALE * eff_q_pred)
            for m in range(M):
                grad[m] -= self.alpha * (Q_sum / max(F, 1)) * log_rate_pred[m]

        return grad

    def _evaluate_dpp(
        self,
        x: np.ndarray,
        j: np.ndarray,
        Q: np.ndarray,
        obs: np.ndarray,
        obs_pred: Optional[np.ndarray],
        lambda_arr: np.ndarray,
        F: int,
    ) -> float:
        """Compute DPP value for tracking best solution."""
        mu = compute_service_rate(obs, x, j, F)
        drift = float(np.sum(Q * (lambda_arr[:F] - mu)))
        cost = compute_cost(obs, x, j, F)
        dpp = drift + self.V * cost

        if self.alpha > 0 and obs_pred is not None:
            mu_pred = compute_service_rate(obs_pred, x, j, F)
            dpp += self.alpha * float(np.sum(Q * (lambda_arr[:F] - mu_pred)))

        return dpp

    def reset(self) -> None:
        """Reset internal state."""


class LyapunovOnly(Scheduler):
    """Lyapunov DPP without adversary modelling (j = 0 always).

    A simpler, faster variant that ignores the Stackelberg game and
    optimizes assuming no perturbation.  Useful as a baseline to
    quantify the value of adversary awareness.
    """

    name = "LyapunovOnly"

    def __init__(
        self,
        cfg: Optional[SchedulerConfig] = None,
        *,
        V: float = 5.0,
        F: int = 4,
        lambda_mean: float = 0.35,
    ):
        if cfg is not None:
            self.V = cfg.V
            self.F = cfg.F
            self.lambda_mean = cfg.lambda_mean
        else:
            self.V = V
            self.F = F
            self.lambda_mean = lambda_mean

    def decide(
        self, observations: np.ndarray, state: Dict[str, Any]
    ) -> np.ndarray:
        """Compute DPP allocation assuming j = 0.

        Returns:
            Allocation vector on the simplex, shape ``(M,)``.
        """
        M = observations.shape[0]
        F = state.get("F", self.F)
        Q = state.get("Q", np.zeros(F))
        Q_sum = float(np.sum(Q))

        quality = channel_quality(observations)
        log_rate = np.log1p(QUALITY_SCALE * quality)

        # Gradient at j = 0
        grad = np.zeros(M)
        for m in range(M):
            grad[m] = (
                -(Q_sum / max(F, 1)) * log_rate[m]
                - self.V * log_rate[m]
                + self.V * USAGE_WEIGHT * quality[m]
            )

        # Single projected gradient step from quality-proportional init
        x = np.maximum(quality, 1e-6)
        x = x / np.sum(x)
        x = x - 0.1 * grad
        x = project_simplex(x)
        return x

    def reset(self) -> None:
        pass
