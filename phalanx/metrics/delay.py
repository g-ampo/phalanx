"""Delay metric: per-packet delay and tail-latency tracking.

Models a simple FIFO queue per flow.  Packets arrive at each slot and
depart when served.  The delay of each packet is measured from arrival
to departure.

Provides mean, max, and tail percentile (P95, P99) summaries.
"""
from __future__ import annotations

import collections
from typing import Any, Deque, Dict, List

import numpy as np

from phalanx.core import Metric


class DelayMetric(Metric):
    """Per-packet delay tracker.

    Maintains a virtual FIFO per flow.  Each slot, new packets arrive
    (Poisson count from ``state["arrivals"]``) and ``state["mu"]``
    packets depart.  The delay of each departed packet is the number
    of slots it spent in the queue.

    Args:
        F: Number of flows.
    """

    name = "DelayMetric"

    def __init__(self, F: int = 4) -> None:
        self.F = F
        self._queues: List[Deque[int]] = [collections.deque() for _ in range(F)]
        self._t: int = 0
        self.delays: List[float] = []
        self.per_slot_mean_delay: List[float] = []

    def record(
        self,
        observations: np.ndarray,
        allocation: np.ndarray,
        perturbation: np.ndarray,
        state: Dict[str, Any],
    ) -> None:
        """Record one slot of delay evolution."""
        self._t += 1
        arrivals = state.get("arrivals", np.zeros(self.F))
        mu = state.get("mu", np.zeros(self.F))

        # Enqueue new packets (tagged with arrival time)
        for f in range(min(self.F, len(arrivals))):
            n_arr = int(arrivals[f])
            for _ in range(n_arr):
                self._queues[f].append(self._t)

        # Dequeue served packets
        slot_delays: List[float] = []
        for f in range(min(self.F, len(mu))):
            n_serve = int(np.floor(mu[f]))
            for _ in range(n_serve):
                if self._queues[f]:
                    arrival_time = self._queues[f].popleft()
                    delay = self._t - arrival_time + 1
                    self.delays.append(float(delay))
                    slot_delays.append(float(delay))

        if slot_delays:
            self.per_slot_mean_delay.append(float(np.mean(slot_delays)))
        else:
            self.per_slot_mean_delay.append(0.0)

    def summarize(self, burn_in: int = 1000) -> Dict[str, float]:
        """Return delay summary after burn-in.

        Keys: ``"mean"``, ``"max"``, ``"p95"``, ``"p99"``,
        ``"mean_slot_delay"``.
        """
        if not self.delays:
            return {
                "mean": 0.0,
                "max": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "mean_slot_delay": 0.0,
            }

        d = np.array(self.delays)
        slot_d = np.array(self.per_slot_mean_delay)
        if len(slot_d) > burn_in:
            slot_d = slot_d[burn_in:]

        return {
            "mean": float(np.mean(d)),
            "max": float(np.max(d)),
            "p95": float(np.percentile(d, 95)),
            "p99": float(np.percentile(d, 99)),
            "mean_slot_delay": float(np.mean(slot_d)) if len(slot_d) > 0 else 0.0,
        }

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        """Return per-slot mean delay trajectory."""
        return {
            "per_slot_mean_delay": np.array(self.per_slot_mean_delay),
        }

    def reset(self) -> None:
        self._queues = [collections.deque() for _ in range(self.F)]
        self._t = 0
        self.delays = []
        self.per_slot_mean_delay = []
