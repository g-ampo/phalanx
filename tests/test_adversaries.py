"""Tests for Phalanx adversary models."""
import numpy as np
import pytest

from phalanx.channels import GaussianVARChannel
from phalanx.config import ChannelConfig
from phalanx.core import channel_quality
from phalanx.adversaries import (
    NoAdversary,
    BudgetAdversary,
    StackelbergAdversary,
    ReactiveAdversary,
    MarkovAdversary,
)


M = 8
D_OBS = 4


def _make_obs():
    cfg = ChannelConfig(M=M, d_obs=D_OBS)
    rng = np.random.default_rng(42)
    ch = GaussianVARChannel(cfg, rng)
    for _ in range(10):
        ch.step()
    return ch.step()


def _make_allocation(rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    x = rng.exponential(1.0, size=M)
    return x / np.sum(x)


def _make_state(**overrides):
    state = {
        "Q": np.zeros(4),
        "t": 100,
        "F": 4,
        "M": M,
        "rng": np.random.default_rng(42),
    }
    state.update(overrides)
    return state


# ── NoAdversary ──

class TestNoAdversary:
    """NoAdversary must always return a zero perturbation vector."""

    def test_returns_zeros(self):
        adv = NoAdversary()
        obs = _make_obs()
        x = _make_allocation()
        state = _make_state()
        j = adv.attack(x, obs, state)
        np.testing.assert_array_equal(j, np.zeros(M))

    def test_returns_correct_shape(self):
        adv = NoAdversary()
        obs = _make_obs()
        x = _make_allocation()
        state = _make_state()
        j = adv.attack(x, obs, state)
        assert j.shape == (M,)

    def test_invariant_to_allocation(self):
        adv = NoAdversary()
        obs = _make_obs()
        state = _make_state()
        j1 = adv.attack(np.ones(M) / M, obs, state)
        j2 = adv.attack(np.eye(M)[0], obs, state)
        np.testing.assert_array_equal(j1, j2)


# ── BudgetAdversary ──

class TestBudgetAdversary:
    """BudgetAdversary must respect the budget constraint."""

    @pytest.mark.parametrize("J_total", [0.0, 0.5, 1.0, 5.0])
    def test_budget_constraint(self, J_total):
        adv = BudgetAdversary(J_total=J_total)
        obs = _make_obs()
        x = _make_allocation()
        state = _make_state()
        j = adv.attack(x, obs, state)

        assert np.all(j >= -1e-10), f"Negative perturbation: {j}"
        assert np.sum(j) <= J_total + 1e-6, (
            f"Budget violated: sum(j)={np.sum(j):.6f} > J_total={J_total}"
        )

    def test_zero_budget_returns_zeros(self):
        adv = BudgetAdversary(J_total=0.0)
        obs = _make_obs()
        x = _make_allocation()
        state = _make_state()
        j = adv.attack(x, obs, state)
        np.testing.assert_array_equal(j, np.zeros(M))

    def test_nonzero_with_positive_budget(self):
        adv = BudgetAdversary(J_total=1.0)
        obs = _make_obs()
        x = _make_allocation()
        state = _make_state()
        j = adv.attack(x, obs, state)
        assert np.sum(j) > 0, "Budget adversary should spend some budget"


# ── StackelbergAdversary ──

class TestStackelbergAdversary:
    """StackelbergAdversary must target the highest-impact channel."""

    def test_budget_constraint(self):
        J_total = 0.5
        adv = StackelbergAdversary(J_total=J_total)
        obs = _make_obs()
        x = _make_allocation()
        state = _make_state()
        j = adv.attack(x, obs, state)

        assert np.all(j >= -1e-10)
        assert np.sum(j) <= J_total + 1e-6

    def test_targets_high_allocation_channel(self):
        """With concentrated allocation, adversary should focus on that channel."""
        adv = StackelbergAdversary(J_total=0.5)
        obs = _make_obs()

        # Concentrate allocation on channel 0
        x = np.zeros(M)
        x[0] = 1.0
        state = _make_state()
        j = adv.attack(x, obs, state)

        # Channel 0 should receive the most perturbation
        if np.sum(j) > 0:
            assert j[0] >= np.max(j[1:]) - 1e-6, (
                f"Adversary should target highest-allocation channel. "
                f"j={j}"
            )

    def test_zero_budget(self):
        adv = StackelbergAdversary(J_total=0.0)
        obs = _make_obs()
        x = _make_allocation()
        state = _make_state()
        j = adv.attack(x, obs, state)
        np.testing.assert_array_equal(j, np.zeros(M))


# ── ReactiveAdversary ──

class TestReactiveAdversary:
    """ReactiveAdversary targets the most-allocated channel probabilistically."""

    def test_perturbation_non_negative(self):
        adv = ReactiveAdversary(J_total=0.5)
        obs = _make_obs()
        x = _make_allocation()
        state = _make_state()
        j = adv.attack(x, obs, state)
        assert np.all(j >= -1e-10)

    def test_activity_rate(self):
        """Over many slots, the fraction of active slots should be near J_total."""
        J_total = 0.3
        adv = ReactiveAdversary(J_total=J_total)
        obs = _make_obs()
        x = _make_allocation()

        active_count = 0
        n_trials = 2000
        for i in range(n_trials):
            state = _make_state(rng=np.random.default_rng(i))
            j = adv.attack(x, obs, state)
            if np.sum(j) > 0:
                active_count += 1

        activity_rate = active_count / n_trials
        assert abs(activity_rate - J_total) < 0.05, (
            f"Activity rate {activity_rate:.3f} too far from J_total={J_total}"
        )


# ── MarkovAdversary ──

class TestMarkovAdversary:
    """MarkovAdversary should cycle through different target channels."""

    def test_perturbation_non_negative(self):
        adv = MarkovAdversary(J_total=0.5, n_states=3, p_switch=0.1)
        obs = _make_obs()
        x = _make_allocation()
        state = _make_state()
        j = adv.attack(x, obs, state)
        assert np.all(j >= -1e-10)

    def test_different_channels_over_time(self):
        """Markov adversary should target multiple channels over many slots."""
        adv = MarkovAdversary(J_total=0.5, n_states=4, p_switch=0.5)
        obs = _make_obs()
        x = _make_allocation()

        targets = set()
        for i in range(200):
            state = _make_state(rng=np.random.default_rng(i))
            j = adv.attack(x, obs, state)
            if np.sum(j) > 0:
                targets.add(int(np.argmax(j)))

        assert len(targets) > 1, (
            f"Markov adversary should target multiple channels. "
            f"Only saw targets: {targets}"
        )
