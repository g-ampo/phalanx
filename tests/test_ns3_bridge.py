"""Tests for the NS3Bridge prediction and metric pipeline."""
import numpy as np
import pytest

from phalanx.ns3.bridge import NS3Bridge
from phalanx.schedulers import RoundRobin, LyapunovDPP
from phalanx.predictors import PersistencePredictor
from phalanx.metrics import CostMetric, PredictionErrorMetric


M = 4
D_OBS = 4
F = 4
T_SHORT = 100


class TestNS3BridgeStub:
    """NS3Bridge stub mode should run the full pipeline without ns-3."""

    def test_stub_runs_without_predictor(self):
        bridge = NS3Bridge(M=M, d_obs=D_OBS, F=F)
        bridge.register_scheduler(RoundRobin())
        result = bridge.run(T=T_SHORT, seed=42)
        assert result["mode"] == "stub"
        assert result["T"] == T_SHORT
        assert result["predictor"] is None

    def test_stub_runs_with_predictor(self):
        bridge = NS3Bridge(M=M, d_obs=D_OBS, F=F)
        bridge.register_scheduler(RoundRobin())
        bridge.register_predictor(
            PersistencePredictor(d_obs_total=M * D_OBS),
            obs_window_size=10,
        )
        result = bridge.run(T=T_SHORT, seed=42)
        assert result["mode"] == "stub"
        assert result["predictor"] == "PersistencePredictor"

    def test_stub_collects_metrics(self):
        bridge = NS3Bridge(M=M, d_obs=D_OBS, F=F)
        bridge.register_scheduler(RoundRobin())
        bridge.register_metric(CostMetric())
        result = bridge.run(T=T_SHORT, seed=42)
        assert "summary" in result
        assert "mean" in result["summary"]
        assert np.isfinite(result["summary"]["mean"])

    def test_stub_prediction_error_metric(self):
        bridge = NS3Bridge(M=M, d_obs=D_OBS, F=F)
        bridge.register_scheduler(RoundRobin())
        bridge.register_predictor(
            PersistencePredictor(d_obs_total=M * D_OBS),
            obs_window_size=10,
        )
        bridge.register_metric(PredictionErrorMetric())
        result = bridge.run(T=T_SHORT, seed=42)
        assert "summary" in result
        assert result["summary"]["nmse"] > 0

    def test_stub_with_lyapunov_dpp(self):
        bridge = NS3Bridge(M=M, d_obs=D_OBS, F=F)
        bridge.register_scheduler(LyapunovDPP(V=5.0, F=F, J_total=0.0))
        bridge.register_predictor(
            PersistencePredictor(d_obs_total=M * D_OBS),
            obs_window_size=10,
        )
        bridge.register_metric(CostMetric())
        result = bridge.run(T=T_SHORT, seed=42)
        assert result["mode"] == "stub"
        assert np.isfinite(result["summary"]["mean"])

    def test_no_scheduler_raises(self):
        bridge = NS3Bridge(M=M, d_obs=D_OBS, F=F)
        with pytest.raises(RuntimeError, match="No scheduler registered"):
            bridge.run(T=10)

    def test_stub_deterministic(self):
        results = []
        for _ in range(2):
            bridge = NS3Bridge(M=M, d_obs=D_OBS, F=F)
            bridge.register_scheduler(RoundRobin())
            bridge.register_metric(CostMetric())
            bridge.register_predictor(
                PersistencePredictor(d_obs_total=M * D_OBS),
                obs_window_size=10,
            )
            results.append(bridge.run(T=T_SHORT, seed=42))
        assert results[0]["summary"]["mean"] == results[1]["summary"]["mean"]


class TestNS3BridgeObsConversion:
    """Observation format conversion should handle edge cases."""

    def test_flat_obs_exact_size(self):
        flat = np.arange(M * D_OBS, dtype=float)
        obs = NS3Bridge.ns3_obs_to_phalanx(flat, M, D_OBS)
        assert obs.shape == (M, D_OBS)
        np.testing.assert_array_equal(obs.ravel(), flat)

    def test_flat_obs_too_long(self):
        flat = np.arange(M * D_OBS + 10, dtype=float)
        obs = NS3Bridge.ns3_obs_to_phalanx(flat, M, D_OBS)
        assert obs.shape == (M, D_OBS)

    def test_flat_obs_too_short(self):
        flat = np.ones(3)
        obs = NS3Bridge.ns3_obs_to_phalanx(flat, M, D_OBS)
        assert obs.shape == (M, D_OBS)
        assert obs[0, 0] == 1.0  # first elements filled
        assert obs[-1, -1] == 0.0  # padding

    def test_2d_obs(self):
        arr = np.ones((M + 2, D_OBS + 1))
        obs = NS3Bridge.ns3_obs_to_phalanx(arr, M, D_OBS)
        assert obs.shape == (M, D_OBS)
