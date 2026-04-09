"""Tests for Phalanx predictor infrastructure."""
import numpy as np
import pytest

from phalanx.predictors.base import Predictor
from phalanx.predictors.persistence import PersistencePredictor
from phalanx.predictors.factory import create_predictor, register_predictor
from phalanx.channels import GaussianVARChannel, ReplayChannel
from phalanx.config import ChannelConfig
from phalanx.schedulers import RoundRobin
from phalanx.adversaries import NoAdversary
from phalanx.metrics import CostMetric, PredictionErrorMetric
from phalanx.core import Simulation


M = 4
D_OBS = 4
D_OBS_TOTAL = M * D_OBS
T_SHORT = 100
WINDOW = 10


# ── Predictor ABC ──

class TestPredictorABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Predictor()

    def test_persistence_is_subclass(self):
        assert issubclass(PersistencePredictor, Predictor)


# ── PersistencePredictor ──

class TestPersistencePredictor:
    def test_encode_returns_last_obs(self):
        pred = PersistencePredictor(d_obs_total=D_OBS_TOTAL)
        window = np.random.default_rng(0).standard_normal((WINDOW, D_OBS_TOTAL))
        z = pred.encode(window)
        np.testing.assert_array_equal(z, window[-1])

    def test_encode_1d_input(self):
        pred = PersistencePredictor(d_obs_total=D_OBS_TOTAL)
        obs = np.random.default_rng(0).standard_normal(D_OBS_TOTAL)
        z = pred.encode(obs)
        np.testing.assert_array_equal(z, obs)

    def test_predict_identity(self):
        pred = PersistencePredictor(d_obs_total=D_OBS_TOTAL)
        z = np.random.default_rng(0).standard_normal(D_OBS_TOTAL)
        o_hat = pred.predict(z)
        np.testing.assert_array_equal(o_hat, z)

    def test_predict_returns_copy(self):
        pred = PersistencePredictor(d_obs_total=D_OBS_TOTAL)
        z = np.ones(D_OBS_TOTAL)
        o_hat = pred.predict(z)
        o_hat[0] = 999.0
        assert z[0] == 1.0

    def test_confidence_default(self):
        pred = PersistencePredictor(d_obs_total=D_OBS_TOTAL)
        z = np.zeros(D_OBS_TOTAL)
        assert pred.confidence(z) == 1.0

    def test_reset_is_noop(self):
        pred = PersistencePredictor(d_obs_total=D_OBS_TOTAL)
        pred.reset()  # should not raise


# ── Factory ──

class TestPredictorFactory:
    def test_create_persistence(self):
        pred = create_predictor("persistence", d_obs_total=D_OBS_TOTAL)
        assert isinstance(pred, PersistencePredictor)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown predictor type"):
            create_predictor("nonexistent")

    def test_register_custom(self):
        class DummyPredictor(Predictor):
            def __init__(self, **kwargs):
                pass
            def encode(self, obs_window):
                return np.zeros(1)
            def predict(self, z):
                return np.zeros(1)

        register_predictor("dummy_test", DummyPredictor)
        pred = create_predictor("dummy_test")
        assert isinstance(pred, DummyPredictor)


# ── ReplayChannel ──

class TestReplayChannel:
    def _make_traces(self, T_total=200, M=M, d_obs=D_OBS, seed=42):
        return np.random.default_rng(seed).standard_normal((T_total, M, d_obs))

    def test_step_returns_correct_shape(self):
        traces = self._make_traces()
        ch = ReplayChannel(traces, M=M, d_obs=D_OBS, segment_length=100)
        obs = ch.step()
        assert obs.shape == (M, D_OBS)

    def test_step_returns_correct_data(self):
        traces = self._make_traces()
        ch = ReplayChannel(traces, M=M, d_obs=D_OBS, segment_length=100)
        obs0 = ch.step()
        np.testing.assert_array_equal(obs0, traces[0])
        obs1 = ch.step()
        np.testing.assert_array_equal(obs1, traces[1])

    def test_save_restore_roundtrip(self):
        traces = self._make_traces()
        ch = ReplayChannel(traces, M=M, d_obs=D_OBS, segment_length=100)
        ch.step()
        ch.step()
        saved = ch.save_state()
        obs_before = ch.step()
        ch.restore_state(saved)
        obs_after = ch.step()
        np.testing.assert_array_equal(obs_before, obs_after)

    def test_reset_with_rng(self):
        traces = self._make_traces(T_total=500)
        ch = ReplayChannel(traces, M=M, d_obs=D_OBS, segment_length=100)
        rng = np.random.default_rng(99)
        ch.reset(rng)
        obs = ch.step()
        # After reset with rng, segment_start should be > 0 (with high probability)
        # and the obs should match the trace at that offset
        assert obs.shape == (M, D_OBS)

    def test_reset_without_rng(self):
        traces = self._make_traces()
        ch = ReplayChannel(traces, M=M, d_obs=D_OBS, segment_length=100)
        ch.step()
        ch.step()
        ch.reset()
        obs = ch.step()
        np.testing.assert_array_equal(obs, traces[0])

    def test_invalid_traces_ndim(self):
        with pytest.raises(ValueError, match="3-D"):
            ReplayChannel(np.zeros((100, M * D_OBS)), M=M, d_obs=D_OBS)

    def test_traces_too_short(self):
        traces = self._make_traces(T_total=50)
        with pytest.raises(ValueError, match="segment_length"):
            ReplayChannel(traces, M=M, d_obs=D_OBS, segment_length=100)

    def test_shape_mismatch(self):
        traces = self._make_traces()
        with pytest.raises(ValueError, match="incompatible"):
            ReplayChannel(traces, M=M + 1, d_obs=D_OBS, segment_length=100)


# ── PredictionErrorMetric ──

class TestPredictionErrorMetric:
    def test_records_when_obs_pred_present(self):
        metric = PredictionErrorMetric()
        obs = np.ones((M, D_OBS))
        pred = np.ones((M, D_OBS)) * 1.1
        state = {"obs_pred": pred}
        metric.record(obs, np.zeros(M), np.zeros(M), state)
        assert len(metric.squared_errors) == 1

    def test_skips_when_obs_pred_absent(self):
        metric = PredictionErrorMetric()
        obs = np.ones((M, D_OBS))
        state = {}
        metric.record(obs, np.zeros(M), np.zeros(M), state)
        assert len(metric.squared_errors) == 0

    def test_nmse_perfect_prediction(self):
        metric = PredictionErrorMetric()
        obs = np.ones((M, D_OBS)) * 2.0
        for _ in range(50):
            metric.record(obs, np.zeros(M), np.zeros(M), {"obs_pred": obs})
        summary = metric.summarize(burn_in=0)
        assert summary["nmse"] == pytest.approx(0.0, abs=1e-12)

    def test_nmse_nonzero_error(self):
        metric = PredictionErrorMetric()
        obs = np.ones((M, D_OBS)) * 2.0
        pred = np.zeros((M, D_OBS))
        for _ in range(50):
            metric.record(obs, np.zeros(M), np.zeros(M), {"obs_pred": pred})
        summary = metric.summarize(burn_in=0)
        assert summary["nmse"] > 0

    def test_reset_clears_data(self):
        metric = PredictionErrorMetric()
        obs = np.ones((M, D_OBS))
        metric.record(obs, np.zeros(M), np.zeros(M), {"obs_pred": obs})
        metric.reset()
        assert len(metric.squared_errors) == 0


# ── Prediction-aware simulation loop ──

class TestPredictionInSimLoop:
    def test_sim_with_predictor_runs(self):
        cfg = ChannelConfig(M=M, d_obs=D_OBS)
        rng = np.random.default_rng(42)
        channel = GaussianVARChannel(cfg, rng)
        scheduler = RoundRobin()
        adversary = NoAdversary()
        metric = CostMetric()
        predictor = PersistencePredictor(d_obs_total=D_OBS_TOTAL)

        sim = Simulation()
        result = sim.run(
            channel, scheduler, adversary, metric,
            T=T_SHORT, seed=42,
            predictor=predictor, obs_window_size=WINDOW,
        )
        assert np.isfinite(result["final_avg"])

    def test_sim_without_predictor_still_works(self):
        cfg = ChannelConfig(M=M, d_obs=D_OBS)
        rng = np.random.default_rng(42)
        channel = GaussianVARChannel(cfg, rng)
        scheduler = RoundRobin()
        adversary = NoAdversary()
        metric = CostMetric()

        sim = Simulation()
        result = sim.run(channel, scheduler, adversary, metric, T=T_SHORT, seed=42)
        assert np.isfinite(result["final_avg"])

    def test_prediction_error_metric_receives_data(self):
        cfg = ChannelConfig(M=M, d_obs=D_OBS)
        rng = np.random.default_rng(42)
        channel = GaussianVARChannel(cfg, rng)
        scheduler = RoundRobin()
        adversary = NoAdversary()
        metric = PredictionErrorMetric()
        predictor = PersistencePredictor(d_obs_total=D_OBS_TOTAL)

        sim = Simulation()
        result = sim.run(
            channel, scheduler, adversary, metric,
            T=T_SHORT, seed=42,
            predictor=predictor, obs_window_size=WINDOW,
        )
        # Prediction starts at slot WINDOW (after buffer fills from 0..WINDOW-1)
        expected_records = T_SHORT - WINDOW
        assert len(metric.squared_errors) == expected_records
        summary = result["summary"]
        assert "nmse" in summary
        # Persistence predictor on VAR(1) channel: NMSE should be > 0
        # but reasonable (not infinite)
        assert 0 < summary["nmse"] < 10.0

    def test_predictor_determinism(self):
        """Same seed + predictor should produce identical results."""
        results = []
        for _ in range(2):
            cfg = ChannelConfig(M=M, d_obs=D_OBS)
            rng = np.random.default_rng(42)
            channel = GaussianVARChannel(cfg, rng)
            scheduler = RoundRobin()
            adversary = NoAdversary()
            metric = CostMetric()
            predictor = PersistencePredictor(d_obs_total=D_OBS_TOTAL)

            sim = Simulation()
            result = sim.run(
                channel, scheduler, adversary, metric,
                T=T_SHORT, seed=42,
                predictor=predictor, obs_window_size=WINDOW,
            )
            results.append(result)
        assert results[0]["final_avg"] == results[1]["final_avg"]
