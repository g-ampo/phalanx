"""Replay channel: feed pre-recorded observation traces through the
standard Channel interface.

Enables evaluation on real-world data (e.g. network traces, measurement
campaigns) without any generative model -- the scheduler, adversary, and
predictor stack operates on actual recorded observations.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from phalanx.core import Channel


class ReplayChannel(Channel):
    """Replays pre-recorded observation traces.

    Each simulation run consumes a contiguous segment of
    ``segment_length`` slots from the trace.  On :meth:`reset`, a new
    random segment is selected (if an RNG is provided), so multi-seed
    runs see different portions of the trace.

    Args:
        traces: Observation array of shape ``(T_total, M, d_obs)``.
        M: Number of channels (must match ``traces.shape[1]``).
        d_obs: Observation dimension (must match ``traces.shape[2]``).
        segment_length: Slots per simulation run.

    Raises:
        ValueError: If shapes are inconsistent or trace is too short.
    """

    def __init__(
        self,
        traces: np.ndarray,
        M: int,
        d_obs: int,
        segment_length: int = 10_000,
    ):
        traces = np.asarray(traces, dtype=np.float64)
        if traces.ndim != 3:
            raise ValueError(
                f"traces must be 3-D (T, M, d_obs), got shape {traces.shape}"
            )
        if traces.shape[1] != M or traces.shape[2] != d_obs:
            raise ValueError(
                f"traces shape {traces.shape} incompatible with "
                f"M={M}, d_obs={d_obs}"
            )
        if traces.shape[0] < segment_length:
            raise ValueError(
                f"traces length {traces.shape[0]} < "
                f"segment_length {segment_length}"
            )

        self.traces = traces
        self.M = M
        self.d_obs = d_obs
        self.segment_length = segment_length
        self._t = 0
        self._segment_start = 0

    def step(self) -> np.ndarray:
        """Return observation at current time and advance by one slot.

        Returns:
            Observation array of shape ``(M, d_obs)``.

        Raises:
            IndexError: If the simulation has exceeded ``segment_length``.
        """
        if self._t >= self.segment_length:
            raise IndexError(
                f"ReplayChannel exhausted: step {self._t} exceeds "
                f"segment_length {self.segment_length}"
            )
        idx = self._segment_start + self._t
        obs = self.traces[idx].copy()
        self._t += 1
        return obs

    def save_state(self) -> Dict[str, Any]:
        """Snapshot position for oracle look-ahead."""
        return {"t": self._t, "segment_start": self._segment_start}

    def restore_state(self, saved: Dict[str, Any]) -> None:
        """Restore a previously saved position."""
        self._t = saved["t"]
        self._segment_start = saved["segment_start"]

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """Reset to a new random segment of the trace.

        Args:
            rng: If provided, picks a uniformly random segment start.
                Otherwise resets to the beginning.
        """
        max_start = self.traces.shape[0] - self.segment_length
        if rng is not None and max_start > 0:
            self._segment_start = int(rng.integers(0, max_start))
        else:
            self._segment_start = 0
        self._t = 0
