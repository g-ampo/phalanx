"""Bandit-based scheduling policies: UCB and Thompson Sampling.

These treat each channel as an arm and use multi-armed bandit algorithms
to balance exploration and exploitation of channel quality.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from phalanx.core import Scheduler, channel_quality


class UCBScheduler(Scheduler):
    """UCB-inspired scheduler with full channel observation.

    Unlike classical UCB where only the pulled arm is observed, in the
    multi-link scheduling setting all channel qualities are observed
    every slot.  The UCB index for channel m at slot t is:

        UCB_m = hat{q}_m + c * sqrt(ln(t) / N_m)

    Since all channels are observed simultaneously, N_m = t for all m,
    reducing the exploration bonus to ``c * sqrt(ln(t) / t)`` -- a
    global decay that vanishes uniformly.  The allocation is a soft-max
    over UCB scores, converging to a quality-proportional policy as the
    exploration bonus diminishes.

    Args:
        c: Exploration constant (higher = more exploration early on).
    """

    name = "UCB"

    def __init__(self, c: float = 2.0):
        self.c = c
        self._counts: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._t: int = 0

    def decide(
        self, observations: np.ndarray, state: Dict[str, Any]
    ) -> np.ndarray:
        M = observations.shape[0]
        quality = channel_quality(observations)
        self._t += 1

        if self._counts is None:
            self._counts = np.zeros(M)
            self._values = np.zeros(M)

        # Update running mean for each channel
        for m in range(M):
            self._counts[m] += 1
            n = self._counts[m]
            self._values[m] += (quality[m] - self._values[m]) / n

        # UCB indices
        ucb = np.zeros(M)
        for m in range(M):
            if self._counts[m] < 1:
                ucb[m] = 1e6  # force exploration
            else:
                bonus = self.c * np.sqrt(
                    np.log(self._t) / self._counts[m]
                )
                ucb[m] = self._values[m] + bonus

        # Softmax allocation
        ucb_shifted = ucb - np.max(ucb)
        exp_ucb = np.exp(ucb_shifted)
        x = exp_ucb / np.sum(exp_ucb)
        return x

    def reset(self) -> None:
        self._counts = None
        self._values = None
        self._t = 0


class ThompsonSamplingScheduler(Scheduler):
    """Thompson Sampling scheduler with Gaussian reward model.

    Maintains per-channel running mean and variance estimates.
    Each slot, samples from the posterior and allocates to the
    channel with the highest sample.

    Uses a Normal-Gamma conjugate model simplified to Normal with
    known variance (estimated from data).
    """

    name = "ThompsonSampling"

    def __init__(self):
        self._counts: np.ndarray | None = None
        self._means: np.ndarray | None = None
        self._sq_sums: np.ndarray | None = None

    def decide(
        self, observations: np.ndarray, state: Dict[str, Any]
    ) -> np.ndarray:
        rng: np.random.Generator = state.get("rng", np.random.default_rng())
        M = observations.shape[0]
        quality = channel_quality(observations)

        if self._counts is None:
            self._counts = np.ones(M)  # prior pseudo-count
            self._means = np.zeros(M)
            self._sq_sums = np.ones(M)

        # Update sufficient statistics
        for m in range(M):
            self._counts[m] += 1
            n = self._counts[m]
            delta = quality[m] - self._means[m]
            self._means[m] += delta / n
            self._sq_sums[m] += delta * (quality[m] - self._means[m])

        # Sample from posterior: N(mu_m, sigma_m^2 / n_m)
        samples = np.zeros(M)
        for m in range(M):
            n = self._counts[m]
            var = max(self._sq_sums[m] / max(n - 1, 1), 1e-6)
            std = np.sqrt(var / n)
            samples[m] = rng.normal(self._means[m], std)

        # One-hot on the best sample (Thompson Sampling is inherently random)
        best = int(np.argmax(samples))
        x = np.zeros(M)
        x[best] = 1.0
        return x

    def reset(self) -> None:
        self._counts = None
        self._means = None
        self._sq_sums = None
