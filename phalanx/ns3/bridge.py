"""NS3Bridge: interface between Phalanx schedulers and ns-3 simulation.

This is a stub implementation that defines the API surface.
Full ns-3 integration requires the ns3-ai Python package and a
running ns-3 simulation process.

The bridge performs two mappings:
1. ns-3 observations -> Phalanx format (M, d_obs) arrays
2. Phalanx allocation -> ns-3 action (scheduling decision)

When a Predictor is registered, the bridge maintains an observation
buffer and injects ``obs_pred`` and ``pred_confidence`` into the
scheduler's state dict -- mirroring the prediction pipeline in
:class:`~phalanx.core.Simulation`.

Usage:
    bridge = NS3Bridge(env_id="phalanx-v0", M=8, d_obs=4, F=4)
    bridge.register_scheduler(my_scheduler)
    bridge.register_predictor(my_predictor, obs_window_size=20)
    bridge.register_metric(CostMetric())
    results = bridge.run(T=1000)
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from phalanx.core import (
    Adversary,
    Channel,
    Metric,
    Scheduler,
    compute_cost,
    compute_service_rate,
)

if TYPE_CHECKING:
    from phalanx.predictors.base import Predictor


class NS3Bridge:
    """Bridge between Phalanx components and an ns-3 simulation.

    The bridge connects to an ns-3 process via the ns3-ai gym interface,
    translates observations and actions, and optionally runs the full
    Phalanx prediction pipeline so that prediction-augmented schedulers
    work identically in both native Phalanx and ns-3 co-simulation.

    Args:
        env_id: ns3-ai gym environment identifier.
        M: Number of channels / links.
        d_obs: Observation dimension per channel.
        F: Number of demand streams.
        lambda_mean: Mean arrival rate per stream.
        ns3_path: Path to the ns-3 installation (or None for auto-detect).
    """

    def __init__(
        self,
        env_id: str = "phalanx-v0",
        M: int = 8,
        d_obs: int = 4,
        F: int = 4,
        lambda_mean: float = 0.35,
        ns3_path: Optional[str] = None,
    ):
        self.env_id = env_id
        self.M = M
        self.d_obs = d_obs
        self.F = F
        self.lambda_mean = lambda_mean
        self.ns3_path = ns3_path

        self._scheduler: Optional[Scheduler] = None
        self._predictor: Optional["Predictor"] = None
        self._metric: Optional[Metric] = None
        self._obs_window_size: int = 20

        self._connected = False
        self._env: Any = None

    def register_scheduler(self, scheduler: Scheduler) -> None:
        """Register a Phalanx scheduler to drive ns-3 actions.

        Args:
            scheduler: Any Phalanx Scheduler instance.
        """
        self._scheduler = scheduler

    def register_predictor(
        self,
        predictor: "Predictor",
        obs_window_size: int = 20,
    ) -> None:
        """Register a predictor for prediction-augmented scheduling.

        When registered, the bridge maintains a rolling observation
        buffer and injects ``state["obs_pred"]`` and
        ``state["pred_confidence"]`` each slot -- the same pipeline
        as :meth:`phalanx.core.Simulation.run`.

        Args:
            predictor: Any Phalanx Predictor instance.
            obs_window_size: Number of past observations for encoding.
        """
        self._predictor = predictor
        self._obs_window_size = obs_window_size

    def register_metric(self, metric: Metric) -> None:
        """Register a metric collector for the bridge loop.

        Args:
            metric: Any Phalanx Metric instance.
        """
        self._metric = metric

    def connect(self) -> bool:
        """Attempt to connect to the ns-3 simulation.

        Returns:
            True if connection succeeded, False if ns3-ai is not available.
        """
        try:
            import ns3ai_gym_env  # noqa: F401
            warnings.warn(
                "ns3-ai detected but full bridge is not yet implemented. "
                "Using stub mode.",
                stacklevel=2,
            )
            return False
        except ImportError:
            warnings.warn(
                "ns3-ai is not installed. NS3Bridge operates in stub mode. "
                "Install with: pip install ns3-ai",
                stacklevel=2,
            )
            return False

    @staticmethod
    def ns3_obs_to_phalanx(
        ns3_obs: Any,
        M: int,
        d_obs: int,
    ) -> np.ndarray:
        """Convert ns-3 observation to Phalanx format.

        Override this method for custom observation mappings.

        Args:
            ns3_obs: Raw observation from ns-3 gym environment.
            M: Number of channels / links.
            d_obs: Observation dimension per channel.

        Returns:
            Array of shape ``(M, d_obs)``.
        """
        obs = np.asarray(ns3_obs, dtype=float)
        if obs.ndim == 1:
            if len(obs) >= M * d_obs:
                return obs[: M * d_obs].reshape(M, d_obs)
            padded = np.zeros(M * d_obs)
            padded[: len(obs)] = obs
            return padded.reshape(M, d_obs)
        return obs[:M, :d_obs]

    @staticmethod
    def phalanx_action_to_ns3(allocation: np.ndarray) -> Any:
        """Convert Phalanx allocation to ns-3 action.

        Override this method for custom action mappings.

        Args:
            allocation: Simplex allocation vector from Phalanx scheduler.

        Returns:
            Action in ns-3 format (default: the allocation array itself).
        """
        return allocation

    def _build_state(
        self,
        t: int,
        Q: np.ndarray,
        arrivals: np.ndarray,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Build the per-slot state dict passed to scheduler and metric."""
        return {
            "Q": Q.copy(),
            "t": t,
            "F": self.F,
            "M": self.M,
            "arrivals": arrivals,
            "lambda_mean": self.lambda_mean,
            "rng": rng,
        }

    def _inject_prediction(
        self,
        state: Dict[str, Any],
        obs: np.ndarray,
        obs_buffer: List[np.ndarray],
    ) -> None:
        """Run the prediction pipeline and inject results into state.

        Prediction uses observations buffered *before* the current slot
        so the forecast is non-trivial (mirrors Simulation.run logic).
        The current observation is appended to the buffer *after*
        prediction.
        """
        if self._predictor is None:
            return
        if len(obs_buffer) >= self._obs_window_size:
            window = np.array(obs_buffer[-self._obs_window_size:])
            z = self._predictor.encode(window)
            obs_pred_flat = self._predictor.predict(z)
            state["obs_pred"] = obs_pred_flat.reshape(obs.shape)
            state["pred_confidence"] = self._predictor.confidence(z)
        obs_buffer.append(obs.ravel())

    def run(
        self,
        T: int = 1000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run the ns-3 bridge loop for T steps.

        In stub mode, runs a self-contained simulation using Phalanx's
        internal channel-quality model (no actual ns-3 process) so that
        the full pipeline -- including prediction -- can be validated.

        In full mode (when ns3-ai is available and connected), runs the
        actual ns-3 co-simulation loop with identical prediction and
        metric plumbing.

        Args:
            T: Number of simulation steps.
            seed: Random seed for arrivals.

        Returns:
            Dict with run metadata and collected metrics.
        """
        if self._scheduler is None:
            raise RuntimeError(
                "No scheduler registered. Call register_scheduler() first."
            )

        if not self._connected:
            self.connect()

        if not self._connected:
            return self._run_stub(T, seed)

        return self._run_full(T, seed)

    def _run_stub(self, T: int, seed: int) -> Dict[str, Any]:
        """Stub loop: validates pipeline without ns-3."""
        rng = np.random.default_rng(seed)
        Q = np.zeros(self.F)
        obs_buffer: List[np.ndarray] = []

        self._scheduler.reset()
        if self._predictor is not None:
            self._predictor.reset()
        if self._metric is not None:
            self._metric.reset()

        for t in range(T):
            # Synthetic observations (uniform random)
            obs = rng.standard_normal((self.M, self.d_obs))
            arrivals = rng.poisson(self.lambda_mean, size=self.F).astype(float)

            state = self._build_state(t, Q, arrivals, rng)
            self._inject_prediction(state, obs, obs_buffer)

            allocation = self._scheduler.decide(obs, state)

            # No real adversary in stub mode
            perturbation = np.zeros(self.M)

            mu = compute_service_rate(obs, allocation, perturbation, self.F)
            Q = np.maximum(Q + arrivals - mu, 0)
            state["Q"] = Q.copy()
            state["mu"] = mu
            state["cost"] = compute_cost(
                obs, allocation, perturbation, self.F
            )

            if self._metric is not None:
                self._metric.record(obs, allocation, perturbation, state)

        result: Dict[str, Any] = {
            "mode": "stub",
            "T": T,
            "scheduler": self._scheduler.name,
            "predictor": (
                type(self._predictor).__name__
                if self._predictor is not None
                else None
            ),
            "env_id": self.env_id,
        }
        if self._metric is not None:
            result["summary"] = self._metric.summarize()
        return result

    def _run_full(self, T: int, seed: int) -> Dict[str, Any]:
        """Full ns-3 co-simulation loop (requires ns3-ai).

        The structure mirrors _run_stub but observations come from ns-3
        and actions are sent back.  Prediction and metric plumbing are
        identical so schedulers behave the same in both modes.
        """
        rng = np.random.default_rng(seed)
        Q = np.zeros(self.F)
        obs_buffer: List[np.ndarray] = []

        self._scheduler.reset()
        if self._predictor is not None:
            self._predictor.reset()
        if self._metric is not None:
            self._metric.reset()

        # ns3-ai gym loop
        ns3_obs = self._env.reset()

        for t in range(T):
            obs = self.ns3_obs_to_phalanx(ns3_obs, self.M, self.d_obs)
            arrivals = rng.poisson(self.lambda_mean, size=self.F).astype(float)

            state = self._build_state(t, Q, arrivals, rng)
            self._inject_prediction(state, obs, obs_buffer)

            allocation = self._scheduler.decide(obs, state)
            action = self.phalanx_action_to_ns3(allocation)
            ns3_obs, reward, done, info = self._env.step(action)

            # Perturbation comes from the real environment
            perturbation = np.asarray(
                info.get("perturbation", np.zeros(self.M)), dtype=float
            )

            mu = compute_service_rate(obs, allocation, perturbation, self.F)
            Q = np.maximum(Q + arrivals - mu, 0)
            state["Q"] = Q.copy()
            state["mu"] = mu
            state["cost"] = compute_cost(
                obs, allocation, perturbation, self.F
            )

            if self._metric is not None:
                self._metric.record(obs, allocation, perturbation, state)

            if done:
                break

        result: Dict[str, Any] = {
            "mode": "full",
            "T": t + 1,
            "scheduler": self._scheduler.name,
            "predictor": (
                type(self._predictor).__name__
                if self._predictor is not None
                else None
            ),
            "env_id": self.env_id,
        }
        if self._metric is not None:
            result["summary"] = self._metric.summarize()
        return result

    def close(self) -> None:
        """Close the ns-3 bridge connection."""
        if self._env is not None:
            pass
        self._connected = False
