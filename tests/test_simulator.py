"""Tests for the Phalanx simulation runner."""
import numpy as np
import pytest

from phalanx.channels import GaussianVARChannel
from phalanx.config import ChannelConfig
from phalanx.schedulers import LyapunovDPP, RoundRobin
from phalanx.adversaries import NoAdversary, StackelbergAdversary
from phalanx.metrics import CostMetric
from phalanx.core import Simulation
from phalanx.simulator import MultiSeedSimulator


M = 4
D_OBS = 4
F = 4
T_SHORT = 200


def _make_components(seed=42):
    """Create a standard set of simulation components."""
    cfg = ChannelConfig(M=M, d_obs=D_OBS)
    rng = np.random.default_rng(seed)
    channel = GaussianVARChannel(cfg, rng)
    scheduler = LyapunovDPP(V=5.0, F=F, J_total=0.5)
    adversary = StackelbergAdversary(J_total=0.5)
    metric = CostMetric()
    return channel, scheduler, adversary, metric


# ── Single-seed run ──

class TestSingleSeedRun:
    """A single-seed run must produce valid results."""

    def test_run_produces_metrics(self):
        channel, scheduler, adversary, metric = _make_components()
        sim = Simulation()
        result = sim.run(channel, scheduler, adversary, metric, T=T_SHORT, seed=42)

        assert "final_avg" in result
        assert "summary" in result
        assert "trajectory" in result
        assert isinstance(result["final_avg"], float)
        assert np.isfinite(result["final_avg"])

    def test_run_summary_keys(self):
        channel, scheduler, adversary, metric = _make_components()
        sim = Simulation()
        result = sim.run(channel, scheduler, adversary, metric, T=T_SHORT, seed=42)

        summary = result["summary"]
        assert "mean" in summary

    def test_trajectory_has_cost_array(self):
        channel, scheduler, adversary, metric = _make_components()
        sim = Simulation()
        result = sim.run(channel, scheduler, adversary, metric, T=T_SHORT, seed=42)

        traj = result["trajectory"]
        assert "cost" in traj
        assert len(traj["cost"]) == T_SHORT

    def test_run_with_no_adversary(self):
        cfg = ChannelConfig(M=M, d_obs=D_OBS)
        rng = np.random.default_rng(42)
        channel = GaussianVARChannel(cfg, rng)
        scheduler = RoundRobin()
        adversary = NoAdversary()
        metric = CostMetric()

        sim = Simulation()
        result = sim.run(channel, scheduler, adversary, metric, T=T_SHORT, seed=42)
        assert np.isfinite(result["final_avg"])


# ── Multi-seed aggregation ──

class TestMultiSeedAggregation:
    """Multi-seed runs must compute valid confidence intervals."""

    def test_multi_seed_produces_ci(self):
        channel, scheduler, adversary, metric = _make_components()
        runner = MultiSeedSimulator(n_seeds=3, n_workers=0)
        agg = runner.run(
            channel, scheduler, adversary, metric,
            T=T_SHORT, base_seed=42, F=F,
        )

        assert "ci" in agg
        mean, lo, hi = agg["ci"]
        assert np.isfinite(mean)
        assert lo <= mean <= hi

    def test_multi_seed_n_seeds_correct(self):
        channel, scheduler, adversary, metric = _make_components()
        n = 5
        runner = MultiSeedSimulator(n_seeds=n, n_workers=0)
        agg = runner.run(
            channel, scheduler, adversary, metric,
            T=T_SHORT, base_seed=42, F=F,
        )

        assert agg["n_seeds"] == n
        assert len(agg["per_seed"]) == n

    def test_ci_width_decreases_with_seeds(self):
        """CI width should generally decrease as n_seeds increases."""
        channel, scheduler, adversary, metric = _make_components()

        widths = []
        for n in [3, 10]:
            runner = MultiSeedSimulator(n_seeds=n, n_workers=0)
            agg = runner.run(
                channel, scheduler, adversary, metric,
                T=T_SHORT, base_seed=42, F=F,
            )
            _, lo, hi = agg["ci"]
            widths.append(hi - lo)

        # With more seeds, CI should be narrower (or at least not much wider)
        assert widths[1] <= widths[0] * 2.0, (
            f"CI width with 10 seeds ({widths[1]:.4f}) should not be "
            f"much wider than with 3 seeds ({widths[0]:.4f})"
        )


# ── Determinism ──

class TestDeterminism:
    """Runs with the same seed must produce identical results."""

    def test_deterministic_single_seed(self):
        sim = Simulation()

        results = []
        for _ in range(2):
            cfg = ChannelConfig(M=M, d_obs=D_OBS)
            rng = np.random.default_rng(42)
            channel = GaussianVARChannel(cfg, rng)
            scheduler = LyapunovDPP(V=5.0, F=F, J_total=0.5)
            adversary = StackelbergAdversary(J_total=0.5)
            metric = CostMetric()
            result = sim.run(channel, scheduler, adversary, metric,
                             T=T_SHORT, seed=42)
            results.append(result)

        assert results[0]["final_avg"] == results[1]["final_avg"]
        np.testing.assert_array_equal(
            results[0]["trajectory"]["cost"],
            results[1]["trajectory"]["cost"],
        )

    def test_different_seeds_produce_different_results(self):
        sim = Simulation()

        results = []
        for seed in [42, 99]:
            cfg = ChannelConfig(M=M, d_obs=D_OBS)
            rng = np.random.default_rng(seed)
            channel = GaussianVARChannel(cfg, rng)
            scheduler = LyapunovDPP(V=5.0, F=F, J_total=0.5)
            adversary = StackelbergAdversary(J_total=0.5)
            metric = CostMetric()
            result = sim.run(channel, scheduler, adversary, metric,
                             T=T_SHORT, seed=seed)
            results.append(result)

        assert results[0]["final_avg"] != results[1]["final_avg"]
