"""Abstract base classes for Phalanx simulation components.

Every channel, scheduler, adversary, and metric inherits from these ABCs.
The Simulation class orchestrates a single-seed run loop.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phalanx.predictors.base import Predictor


class Channel(abc.ABC):
    """Abstract stochastic channel.

    A channel produces observations at each time step and supports
    save/restore for oracle look-ahead and Stackelberg best-response
    computation.
    """

    @abc.abstractmethod
    def step(self) -> np.ndarray:
        """Advance the channel by one slot and return observations.

        Returns:
            Observation array of shape ``(M, d_obs)`` where *M* is the
            number of links and *d_obs* the per-link observation dimension.
        """

    @abc.abstractmethod
    def save_state(self) -> Dict[str, Any]:
        """Snapshot all mutable state (including RNG) for later restore."""

    @abc.abstractmethod
    def restore_state(self, saved: Dict[str, Any]) -> None:
        """Restore a previously saved snapshot."""

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """Reset channel to initial conditions, optionally with a new RNG."""


class Scheduler(abc.ABC):
    """Abstract scheduling policy.

    Given channel observations and internal state (queues, ages, etc.),
    the scheduler produces a resource-allocation vector.
    """

    name: str = "BaseScheduler"

    @abc.abstractmethod
    def decide(
        self,
        observations: np.ndarray,
        state: Dict[str, Any],
    ) -> np.ndarray:
        """Choose a resource allocation.

        Args:
            observations: Channel observations, shape ``(M, d_obs)``.
            state: Shared simulation state dict containing at minimum
                ``"Q"`` (queue backlogs), ``"t"`` (current slot), and
                any metric-specific entries the scheduler may need.

        Returns:
            Allocation vector of shape ``(M,)`` on the probability
            simplex (entries >= 0, sum = 1).
        """

    def reset(self) -> None:
        """Reset any internal state between runs."""


class Adversary(abc.ABC):
    """Abstract adversary (jammer / perturbation source).

    The adversary observes the scheduler's allocation and channel
    observations and produces a perturbation vector.
    """

    name: str = "BaseAdversary"

    @abc.abstractmethod
    def attack(
        self,
        allocation: np.ndarray,
        observations: np.ndarray,
        state: Dict[str, Any],
    ) -> np.ndarray:
        """Choose a perturbation.

        Args:
            allocation: Scheduler's allocation vector, shape ``(M,)``.
            observations: Channel observations, shape ``(M, d_obs)``.
            state: Shared simulation state dict.

        Returns:
            Perturbation vector of shape ``(M,)``, representing
            per-channel adversarial effort.
        """

    def reset(self) -> None:
        """Reset any internal state between runs."""


class Metric(abc.ABC):
    """Abstract metric collector.

    Records per-slot data and provides a scalar summary.
    """

    name: str = "BaseMetric"

    @abc.abstractmethod
    def record(
        self,
        observations: np.ndarray,
        allocation: np.ndarray,
        perturbation: np.ndarray,
        state: Dict[str, Any],
    ) -> None:
        """Record data for one time slot.

        Args:
            observations: Channel observations this slot.
            allocation: Scheduler allocation this slot.
            perturbation: Adversary perturbation this slot.
            state: Shared simulation state dict.
        """

    @abc.abstractmethod
    def summarize(self, burn_in: int = 1000) -> Dict[str, float]:
        """Return a summary dict (e.g. mean, max, tail percentile).

        Args:
            burn_in: Number of initial slots to discard.
        """

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        """Return time-series arrays for plotting (optional)."""
        return {}

    def reset(self) -> None:
        """Clear all recorded data."""


# ---------------------------------------------------------------------------
# Utility: channel-quality extraction
# ---------------------------------------------------------------------------

def channel_quality(obs: np.ndarray) -> np.ndarray:
    """Extract scalar quality per channel from observation array.

    If ``obs`` is 1-D, returns absolute values.  If 2-D with shape
    ``(M, d_obs)``, returns the L2 norm per row.
    """
    if obs.ndim == 1:
        return np.abs(obs)
    return np.linalg.norm(obs, axis=1)


def project_simplex(x: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex.

    Uses the efficient O(n log n) algorithm of Duchi et al. (2008).
    """
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.ones(n) / n
    rho = ind[cond][-1]
    theta = cssv[rho - 1] / rho
    return np.maximum(x - theta, 0)


# ---------------------------------------------------------------------------
# Service-rate and cost helpers
# ---------------------------------------------------------------------------

QUALITY_SCALE: float = 10.0
USAGE_WEIGHT: float = 0.1


def compute_service_rate(
    obs: np.ndarray,
    x: np.ndarray,
    j: np.ndarray,
    F: int,
) -> np.ndarray:
    """Per-flow service rate: mu_f = (1/F) sum_m x_m log(1 + eta max(q_m - j_m, 0)).

    Args:
        obs: Channel observations, shape ``(M,)`` or ``(M, d_obs)``.
        x: Allocation vector, shape ``(M,)``.
        j: Perturbation vector, shape ``(M,)``.
        F: Number of demand streams.

    Returns:
        Service-rate vector of shape ``(F,)``.
    """
    quality = channel_quality(obs)
    eff_q = np.maximum(quality - j, 0)
    total_rate = np.sum(x * np.log1p(QUALITY_SCALE * eff_q))
    return np.full(F, total_rate / F)


def compute_cost(
    obs: np.ndarray,
    x: np.ndarray,
    j: np.ndarray,
    F: int,
) -> float:
    """Per-slot cost: -throughput + usage_cost.

    c(x, o, j) = -sum_m x_m log(1 + eta max(q_m - j_m, 0)) + gamma sum_m x_m q_m
    """
    quality = channel_quality(obs)
    eff_q = np.maximum(quality - j, 0)
    throughput = np.sum(x * np.log1p(QUALITY_SCALE * eff_q))
    usage_cost = USAGE_WEIGHT * np.sum(x * quality)
    return -throughput + usage_cost


# ---------------------------------------------------------------------------
# Simulation orchestrator
# ---------------------------------------------------------------------------

class Simulation:
    """Single-seed simulation run loop.

    Orchestrates the interaction between channel, scheduler, adversary,
    and metric for *T* time slots.
    """

    def run(
        self,
        channel: Channel,
        scheduler: Scheduler,
        adversary: Adversary,
        metric: Metric,
        T: int = 10_000,
        seed: int = 42,
        F: int = 4,
        lambda_mean: float = 0.35,
        predictor: Optional["Predictor"] = None,
        obs_window_size: int = 20,
    ) -> Dict[str, Any]:
        """Execute a simulation run.

        Args:
            channel: Channel model instance.
            scheduler: Scheduling policy instance.
            adversary: Adversary model instance.
            metric: Metric collector instance.
            T: Number of time slots.
            seed: Random seed for arrivals and tie-breaking.
            F: Number of demand streams.
            lambda_mean: Mean arrival rate per stream.
            predictor: Optional observation predictor.  When provided,
                the loop maintains an observation buffer and injects
                ``state["obs_pred"]`` and ``state["pred_confidence"]``
                for the scheduler to consume.
            obs_window_size: Number of past observations to feed
                the predictor's ``encode()`` method.

        Returns:
            Dict with ``"final_avg"``, ``"trajectory"``, and the full
            ``"metric"`` object.
        """
        rng = np.random.default_rng(seed)
        M = None

        # Initialize queues
        Q: Optional[np.ndarray] = None

        # Observation buffer for predictor
        obs_buffer: List[np.ndarray] = [] if predictor is not None else []

        scheduler.reset()
        adversary.reset()
        metric.reset()
        if predictor is not None:
            predictor.reset()

        for t in range(T):
            obs = channel.step()
            if M is None:
                M = obs.shape[0]
                Q = np.zeros(F)

            # Arrivals
            arrivals = rng.poisson(lambda_mean, size=F).astype(float)

            # Build state dict
            state: Dict[str, Any] = {
                "Q": Q.copy(),
                "t": t,
                "F": F,
                "M": M,
                "arrivals": arrivals,
                "lambda_mean": lambda_mean,
                "rng": rng,
            }

            # Prediction pipeline: predict from past observations
            # (excluding current obs) so the forecast is non-trivial.
            if predictor is not None:
                if len(obs_buffer) >= obs_window_size:
                    window = np.array(obs_buffer[-obs_window_size:])
                    z = predictor.encode(window)
                    obs_pred_flat = predictor.predict(z)
                    state["obs_pred"] = obs_pred_flat.reshape(obs.shape)
                    state["pred_confidence"] = predictor.confidence(z)
                obs_buffer.append(obs.ravel())

            # Scheduler decides
            allocation = scheduler.decide(obs, state)

            # Adversary attacks
            perturbation = adversary.attack(allocation, obs, state)

            # Compute service and update queues
            mu = compute_service_rate(obs, allocation, perturbation, F)
            Q = np.maximum(Q + arrivals - mu, 0)
            state["Q"] = Q.copy()
            state["mu"] = mu
            state["cost"] = compute_cost(obs, allocation, perturbation, F)

            # Record metric
            metric.record(obs, allocation, perturbation, state)

        summary = metric.summarize()
        return {
            "final_avg": summary.get("mean", 0.0),
            "summary": summary,
            "trajectory": metric.get_trajectory(),
            "metric": metric,
        }
