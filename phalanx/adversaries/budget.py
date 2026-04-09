"""Budget-constrained adversary.

Budget-constrained greedy adversary.

The adversary has a total perturbation budget J_total and greedily
concentrates it on the channel(s) where the marginal throughput
reduction is highest.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from phalanx.config import AdversaryConfig
from phalanx.core import QUALITY_SCALE, Adversary, channel_quality


class BudgetAdversary(Adversary):
    """Budget-constrained greedy adversary.

    Concentrates its budget on the channel with the highest marginal
    impact on the scheduler's throughput (greedy vertex search on the
    budget polytope).

    Args:
        cfg: Adversary configuration, or keyword arguments.
        J_total: Total perturbation budget.
    """

    name = "BudgetAdversary"

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
        """Greedy best response: concentrate budget on max-impact channel.

        Since throughput is concave in j, the cost is convex and the
        maximum over the budget polytope is attained at a vertex.
        """
        M = len(allocation)
        if self.J_total <= 0:
            return np.zeros(M)

        quality = channel_quality(observations)
        j = np.zeros(M)
        remaining = self.J_total

        for _ in range(M):
            if remaining <= 1e-10:
                break
            eff_q = np.maximum(quality - j, 0)
            marginals = (
                allocation * QUALITY_SCALE / (1.0 + QUALITY_SCALE * eff_q)
            )
            marginals[eff_q <= 1e-10] = 0.0

            if np.max(marginals) < 1e-10:
                break

            target = int(np.argmax(marginals))
            alloc = min(remaining, max(quality[target] - j[target], 0))
            j[target] += alloc
            remaining -= alloc

        return j

    def reset(self) -> None:
        pass
