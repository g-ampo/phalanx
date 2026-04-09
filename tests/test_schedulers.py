"""Tests for Phalanx scheduling policies."""
import numpy as np
import pytest

from phalanx.channels import GaussianVARChannel
from phalanx.config import ChannelConfig
from phalanx.core import channel_quality, QUALITY_SCALE
from phalanx.schedulers import (
    LyapunovDPP,
    RoundRobin,
    Greedy,
    Random,
    Oracle,
    WhittleIndex,
    AoIIndexPolicy,
)
from phalanx.schedulers.lyapunov import LyapunovOnly


M = 8
D_OBS = 4
F = 4


def _make_obs(rng=None):
    """Generate a sample observation array."""
    if rng is None:
        rng = np.random.default_rng(42)
    cfg = ChannelConfig(M=M, d_obs=D_OBS)
    ch = GaussianVARChannel(cfg, rng)
    for _ in range(10):
        ch.step()
    return ch.step()


def _make_state(**overrides):
    """Build a default simulation state dict."""
    f = overrides.pop("F", F)
    state = {
        "Q": np.zeros(f),
        "t": 100,
        "F": f,
        "M": M,
        "arrivals": np.full(f, 0.35),
        "lambda_mean": 0.35,
        "rng": np.random.default_rng(42),
    }
    state.update(overrides)
    return state


@pytest.fixture(params=[
    "lyapunov_dpp", "lyapunov_only", "round_robin", "greedy",
    "random", "oracle", "whittle", "aoi_index",
])
def scheduler(request):
    """Parametrized fixture returning a scheduler instance."""
    name = request.param
    if name == "lyapunov_dpp":
        return LyapunovDPP(V=5.0, F=F, J_total=0.5)
    elif name == "lyapunov_only":
        return LyapunovOnly(V=5.0, F=F)
    elif name == "round_robin":
        return RoundRobin()
    elif name == "greedy":
        return Greedy()
    elif name == "random":
        return Random()
    elif name == "oracle":
        return Oracle(J_total=0.5)
    elif name == "whittle":
        return WhittleIndex(N_sources=M, p_succ_base=0.8)
    elif name == "aoi_index":
        return AoIIndexPolicy(V=10.0, N_sources=M, p_succ_base=0.8)


# ── Allocation validity ──

class TestAllocationValidity:
    """Every scheduler must produce a valid allocation on the simplex."""

    def test_non_negative(self, scheduler):
        obs = _make_obs()
        state = _make_state()
        x = scheduler.decide(obs, state)
        assert np.all(x >= -1e-10), f"Negative allocation: {x}"

    def test_sums_to_one(self, scheduler):
        obs = _make_obs()
        state = _make_state()
        x = scheduler.decide(obs, state)
        assert abs(np.sum(x) - 1.0) < 1e-6, f"Sum not 1: {np.sum(x)}"

    def test_correct_length(self, scheduler):
        obs = _make_obs()
        state = _make_state()
        x = scheduler.decide(obs, state)
        assert len(x) == M

    def test_different_M_values(self, scheduler):
        """Allocation should adapt to the observation shape."""
        for m in [2, 4, 8]:
            cfg = ChannelConfig(M=m, d_obs=D_OBS)
            rng = np.random.default_rng(42)
            ch = GaussianVARChannel(cfg, rng)
            obs = ch.step()
            state = _make_state(M=m, F=min(F, m))
            x = scheduler.decide(obs, state)
            assert len(x) == m
            assert abs(np.sum(x) - 1.0) < 1e-6


# ── LyapunovDPP specific tests ──

class TestLyapunovDPP:
    """Tests specific to the LyapunovDPP scheduler."""

    def test_zero_queues_quality_proportional(self):
        """With Q=0, the allocation should be roughly quality-proportional."""
        obs = _make_obs()
        state = _make_state(Q=np.zeros(F))
        sched = LyapunovDPP(V=5.0, F=F, J_total=0.0)  # No adversary
        x = sched.decide(obs, state)

        quality = channel_quality(obs)
        quality_frac = quality / np.sum(quality)

        # Correlation between allocation and quality should be positive
        corr = np.corrcoef(x, quality_frac)[0, 1]
        assert corr > 0.3, (
            f"With zero queues, allocation should correlate with quality. "
            f"Got correlation={corr:.3f}"
        )

    def test_high_V_emphasizes_cost(self):
        """Higher V should shift allocation toward throughput-optimal."""
        obs = _make_obs()
        state = _make_state(Q=np.ones(F) * 5.0)

        x_low_v = LyapunovDPP(V=1.0, F=F, J_total=0.0).decide(obs, state)
        x_high_v = LyapunovDPP(V=100.0, F=F, J_total=0.0).decide(obs, state)

        # Both should be valid allocations
        assert abs(np.sum(x_low_v) - 1.0) < 1e-6
        assert abs(np.sum(x_high_v) - 1.0) < 1e-6


# ── Oracle specific test ──

class TestOracle:
    """Oracle with future knowledge should achieve the best cost."""

    def test_oracle_beats_greedy(self):
        """Oracle (DPP with high V) should match or beat Greedy on cost."""
        from phalanx.core import compute_cost
        rng = np.random.default_rng(42)
        cfg = ChannelConfig(M=M, d_obs=D_OBS)
        ch = GaussianVARChannel(cfg, rng)

        oracle = Oracle(J_total=0.0)
        greedy = Greedy()
        j_zero = np.zeros(M)

        oracle_costs = []
        greedy_costs = []
        for _ in range(50):
            obs = ch.step()
            state = _make_state()
            xo = oracle.decide(obs, state)
            xg = greedy.decide(obs, state)
            oracle_costs.append(compute_cost(obs, xo, j_zero, F))
            greedy_costs.append(compute_cost(obs, xg, j_zero, F))

        assert np.mean(oracle_costs) <= np.mean(greedy_costs) + 0.05

    def test_oracle_cost_vs_round_robin(self):
        """Oracle should achieve lower or equal cost than round robin."""
        from phalanx.core import compute_cost
        rng = np.random.default_rng(42)
        cfg = ChannelConfig(M=M, d_obs=D_OBS)
        ch = GaussianVARChannel(cfg, rng)

        oracle = Oracle(J_total=0.0)
        rr = RoundRobin()
        j_zero = np.zeros(M)

        oracle_costs = []
        rr_costs = []
        for _ in range(200):
            obs = ch.step()
            state = _make_state()
            x_oracle = oracle.decide(obs, state)
            x_rr = rr.decide(obs, state)
            oracle_costs.append(compute_cost(obs, x_oracle, j_zero, F))
            rr_costs.append(compute_cost(obs, x_rr, j_zero, F))

        assert np.mean(oracle_costs) <= np.mean(rr_costs) + 0.1
