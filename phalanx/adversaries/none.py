"""Benign (no-adversary) baseline."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from phalanx.core import Adversary


class NoAdversary(Adversary):
    """Benign channel with no adversarial perturbation.

    Always returns a zero perturbation vector.
    """

    name = "NoAdversary"

    def attack(
        self,
        allocation: np.ndarray,
        observations: np.ndarray,
        state: Dict[str, Any],
    ) -> np.ndarray:
        return np.zeros(len(allocation))

    def reset(self) -> None:
        pass
