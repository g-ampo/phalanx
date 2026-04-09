"""Tests for Phalanx metric collectors."""
import numpy as np
import pytest
from scipy.stats import t as t_dist

from phalanx.metrics import CostMetric, AoIMetric, compute_ci
from phalanx.metrics.ci import compute_ci_array


# ── compute_ci tests ──

class TestComputeCI:
    """CI computation must match scipy.stats.t."""

    def test_matches_scipy(self):
        """CI bounds should match the t-distribution formula exactly."""
        rng = np.random.default_rng(42)
        values = rng.normal(5.0, 2.0, size=30)

        mean, lo, hi = compute_ci(values, confidence=0.95)

        # Manual computation with scipy
        n = len(values)
        expected_mean = float(np.mean(values))
        expected_std = float(np.std(values, ddof=1))
        t_crit = float(t_dist.ppf(0.975, df=n - 1))
        margin = t_crit * expected_std / np.sqrt(n)

        assert abs(mean - expected_mean) < 1e-10
        assert abs(lo - (expected_mean - margin)) < 1e-10
        assert abs(hi - (expected_mean + margin)) < 1e-10

    def test_single_value(self):
        """With one value, CI should collapse to that value."""
        mean, lo, hi = compute_ci([3.14])
        assert mean == lo == hi == 3.14

    def test_two_values(self):
        """With two values, CI should be wider than the range."""
        mean, lo, hi = compute_ci([1.0, 3.0])
        assert mean == pytest.approx(2.0)
        assert lo < 1.0  # CI extends below the min
        assert hi > 3.0  # CI extends above the max

    def test_identical_values(self):
        """With identical values, CI should collapse to that value."""
        mean, lo, hi = compute_ci([5.0, 5.0, 5.0, 5.0])
        assert mean == pytest.approx(5.0)
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
    def test_higher_confidence_wider(self, confidence):
        """Higher confidence level should produce wider CI."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, size=20)

        _, lo_95, hi_95 = compute_ci(values, confidence=0.95)
        _, lo_c, hi_c = compute_ci(values, confidence=confidence)

        width_95 = hi_95 - lo_95
        width_c = hi_c - lo_c

        if confidence > 0.95:
            assert width_c >= width_95 - 1e-10
        elif confidence < 0.95:
            assert width_c <= width_95 + 1e-10

    def test_ci_array(self):
        """compute_ci_array should return per-time-step CI arrays."""
        rng = np.random.default_rng(42)
        arrays = [rng.normal(0, 1, size=100) for _ in range(10)]
        mean_arr, lo_arr, hi_arr = compute_ci_array(arrays)

        assert mean_arr.shape == (100,)
        assert lo_arr.shape == (100,)
        assert hi_arr.shape == (100,)
        assert np.all(lo_arr <= mean_arr + 1e-10)
        assert np.all(hi_arr >= mean_arr - 1e-10)


# ── AoIMetric tests ──

class TestAoIMetric:
    """AoI metric must track age correctly."""

    def _make_state(self, selected=-1, success=False):
        return {
            "selected_source": selected,
            "tx_success": success,
            "Q": np.zeros(1),
            "F": 4,
            "M": 4,
        }

    def test_age_increments_without_success(self):
        """Without any successful delivery, ages should monotonically increase."""
        metric = AoIMetric(N_sources=4)
        metric.reset()

        obs = np.zeros((4, 4))
        x = np.ones(4) / 4
        j = np.zeros(4)

        for t in range(10):
            metric.record(obs, x, j, self._make_state(selected=-1, success=False))

        history = np.array(metric.age_history)
        # Ages should be 2, 3, 4, ..., 11 (start at 1, increment each slot)
        for t in range(10):
            expected_age = t + 2  # initial=1, first increment to 2
            np.testing.assert_array_equal(
                history[t], np.full(4, expected_age)
            )

    def test_age_resets_on_success(self):
        """Successful delivery should reset the selected source's age to 1."""
        metric = AoIMetric(N_sources=4)
        metric.reset()

        obs = np.zeros((4, 4))
        x = np.ones(4) / 4
        j = np.zeros(4)

        # Slot 0-4: no delivery (ages reach 6)
        for _ in range(5):
            metric.record(obs, x, j, self._make_state(selected=-1, success=False))

        # Slot 5: source 0 succeeds
        metric.record(obs, x, j, self._make_state(selected=0, success=True))

        last_ages = metric.age_history[-1]
        # Source 0 was reset to 1, others continued incrementing
        assert last_ages[0] == 1
        assert last_ages[1] == 7  # started at 1, incremented 6 times
        assert last_ages[2] == 7
        assert last_ages[3] == 7

    def test_age_does_not_reset_on_failure(self):
        """If transmission fails, age should NOT reset."""
        metric = AoIMetric(N_sources=4)
        metric.reset()

        obs = np.zeros((4, 4))
        x = np.ones(4) / 4
        j = np.zeros(4)

        # 3 slots, no success
        for _ in range(3):
            metric.record(obs, x, j, self._make_state(selected=-1, success=False))

        # Source 1 selected but fails
        metric.record(obs, x, j, self._make_state(selected=1, success=False))

        last_ages = metric.age_history[-1]
        # Source 1 should NOT be reset (failed delivery)
        assert last_ages[1] == 5  # initial=1, incremented 4 times

    def test_summarize_returns_finite(self):
        metric = AoIMetric(N_sources=4)
        metric.reset()

        obs = np.zeros((4, 4))
        x = np.ones(4) / 4
        j = np.zeros(4)

        for t in range(50):
            success = (t % 10 == 0)
            metric.record(obs, x, j, self._make_state(
                selected=t % 4 if success else -1, success=success))

        summary = metric.summarize(burn_in=10)
        assert np.isfinite(summary["mean"])
        assert summary["mean"] > 0

    def test_reset_clears_history(self):
        metric = AoIMetric(N_sources=4)
        obs = np.zeros((4, 4))
        x = np.ones(4) / 4
        j = np.zeros(4)

        metric.record(obs, x, j, self._make_state())
        assert len(metric.age_history) == 1

        metric.reset()
        assert len(metric.age_history) == 0
        np.testing.assert_array_equal(metric.ages, np.ones(4))


# ── CostMetric tests ──

class TestCostMetric:
    """CostMetric must correctly track and summarize costs."""

    def test_records_costs(self):
        metric = CostMetric()
        metric.reset()

        obs = np.random.default_rng(42).standard_normal((4, 4))
        x = np.ones(4) / 4
        j = np.zeros(4)

        for t in range(10):
            state = {"cost": float(t), "Q": np.zeros(4), "F": 4}
            metric.record(obs, x, j, state)

        assert len(metric.costs) == 10
        assert metric.costs[5] == 5.0

    def test_summarize_with_burn_in(self):
        metric = CostMetric()
        metric.reset()

        obs = np.zeros((4, 4))
        x = np.ones(4) / 4
        j = np.zeros(4)

        # Record 20 slots: first 10 have cost=100, next 10 have cost=1
        for t in range(20):
            cost = 100.0 if t < 10 else 1.0
            state = {"cost": cost, "Q": np.zeros(4), "F": 4}
            metric.record(obs, x, j, state)

        summary = metric.summarize(burn_in=10)
        assert summary["mean"] == pytest.approx(1.0)
