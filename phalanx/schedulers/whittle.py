"""Whittle index scheduler for restless bandits.

The Whittle index for source n in the AoI restless-bandit model
(Kadota et al., ToN 2019) is:

    W_n = w_n * Delta_n * p_s_n

where w_n is importance weight, Delta_n is current age, and p_s_n is
success probability.  Indexability is proven for AoI bandits.

In the multi-link scheduling context, the index is adapted to use
channel quality as a proxy for success probability.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from phalanx.core import Scheduler, channel_quality


class WhittleIndex(Scheduler):
    """Whittle index policy for AoI or restless-bandit scheduling.

    When ``state`` contains ``"ages"`` and ``"weights"``, uses them
    (AoI mode).  Otherwise falls back to channel quality (throughput mode).

    Args:
        N_sources: Number of sources (AoI mode).
        p_succ_base: Base success probability (AoI mode).
    """

    name = "Whittle"

    def __init__(self, N_sources: int = 4, p_succ_base: float = 0.8):
        self.N = N_sources
        self.p_succ = p_succ_base

    def decide(
        self, observations: np.ndarray, state: Dict[str, Any]
    ) -> np.ndarray:
        M = observations.shape[0]

        ages = state.get("ages", None)
        if ages is not None:
            # AoI restless-bandit mode
            weights = state.get("weights", np.ones(len(ages)))
            N = len(ages)
            whittle = np.zeros(N)
            for n in range(N):
                whittle[n] = weights[n] * ages[n] * self.p_succ

            x = np.zeros(M)
            best = int(np.argmax(whittle))
            x[min(best, M - 1)] = 1.0
            return x

        # Throughput mode: use channel quality as Whittle proxy
        quality = channel_quality(observations)
        x = np.zeros(M)
        x[int(np.argmax(quality))] = 1.0
        return x

    def reset(self) -> None:
        pass
