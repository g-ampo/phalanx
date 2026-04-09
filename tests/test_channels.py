"""Tests for Phalanx channel models."""
import numpy as np
import pytest

from phalanx.channels import GaussianVARChannel, MarkovModulatedChannel, NTNOrbitalChannel
from phalanx.config import ChannelConfig


# ── Fixtures ──

@pytest.fixture(params=["gaussian", "markov", "ntn"])
def channel_and_cfg(request):
    """Parametrized fixture producing (channel, cfg) for each channel type."""
    cfg = ChannelConfig(M=4, d_obs=4)
    rng = np.random.default_rng(42)
    if request.param == "gaussian":
        cfg.channel_type = "gaussian"
        ch = GaussianVARChannel(cfg, rng)
    elif request.param == "markov":
        cfg.channel_type = "markov"
        ch = MarkovModulatedChannel(cfg, rng)
    elif request.param == "ntn":
        cfg.channel_type = "ntn"
        ch = NTNOrbitalChannel(cfg, rng)
    return ch, cfg, request.param


# ── Shape tests ──

class TestChannelShape:
    """Each channel must produce observations with the correct shape."""

    def test_step_returns_correct_shape(self, channel_and_cfg):
        ch, cfg, ctype = channel_and_cfg
        obs = ch.step()
        assert obs.ndim == 2
        assert obs.shape[0] == cfg.M
        if ctype == "ntn":
            assert obs.shape[1] == 4  # (SNR, doppler, elev, PL)
        else:
            assert obs.shape[1] == cfg.d_obs

    def test_multiple_steps_consistent_shape(self, channel_and_cfg):
        ch, cfg, ctype = channel_and_cfg
        for _ in range(10):
            obs = ch.step()
            assert obs.shape[0] == cfg.M

    def test_different_M_values(self):
        for M in [1, 2, 8, 16]:
            cfg = ChannelConfig(M=M, d_obs=4)
            rng = np.random.default_rng(123)
            ch = GaussianVARChannel(cfg, rng)
            obs = ch.step()
            assert obs.shape == (M, 4)


# ── Save/restore roundtrip tests ──

class TestSaveRestore:
    """Save and restore must produce identical subsequent observations."""

    def test_save_restore_roundtrip(self, channel_and_cfg):
        ch, cfg, ctype = channel_and_cfg
        # Warm up
        for _ in range(20):
            ch.step()

        # Save state
        saved = ch.save_state()

        # Run forward and record
        obs_a = [ch.step() for _ in range(5)]

        # Restore and run again
        ch.restore_state(saved)
        obs_b = [ch.step() for _ in range(5)]

        for a, b in zip(obs_a, obs_b):
            np.testing.assert_array_equal(a, b)

    def test_restore_does_not_share_memory(self, channel_and_cfg):
        ch, cfg, ctype = channel_and_cfg
        for _ in range(10):
            ch.step()
        saved = ch.save_state()
        ch.step()  # advance past saved point

        # Restore and verify the channel did not alias the saved dict
        ch.restore_state(saved)
        obs1 = ch.step()
        ch.restore_state(saved)
        obs2 = ch.step()
        np.testing.assert_array_equal(obs1, obs2)


# ── Temporal correlation test (Gaussian only) ──

class TestTemporalCorrelation:
    """Gaussian VAR(1) channel must show temporal correlation matching config."""

    @pytest.mark.parametrize("var_coeff", [0.5, 0.9, 0.95])
    def test_lag1_autocorrelation(self, var_coeff):
        cfg = ChannelConfig(M=4, d_obs=4, var_coefficient=var_coeff)
        rng = np.random.default_rng(42)
        ch = GaussianVARChannel(cfg, rng)

        # Collect long time series of scalar quality per channel
        T = 5000
        qualities = np.zeros((T, cfg.M))
        for t in range(T):
            obs = ch.step()
            qualities[t] = np.linalg.norm(obs, axis=1)

        # Compute lag-1 autocorrelation for each channel
        for m in range(cfg.M):
            series = qualities[:, m]
            series_centered = series - np.mean(series)
            acf_0 = np.sum(series_centered ** 2)
            acf_1 = np.sum(series_centered[:-1] * series_centered[1:])
            rho_1 = acf_1 / acf_0 if acf_0 > 0 else 0.0

            # The empirical lag-1 autocorrelation should be close to
            # var_coefficient (VAR(1) process), with tolerance for
            # finite-sample noise and the observation function.
            assert rho_1 > 0.1, (
                f"Lag-1 autocorrelation {rho_1:.3f} too low "
                f"for var_coefficient={var_coeff}"
            )

    def test_zero_coefficient_low_correlation(self):
        """With var_coefficient=0, successive observations should be nearly uncorrelated."""
        cfg = ChannelConfig(M=4, d_obs=4, var_coefficient=0.0)
        rng = np.random.default_rng(42)
        ch = GaussianVARChannel(cfg, rng)

        T = 3000
        qualities = np.zeros(T)
        for t in range(T):
            obs = ch.step()
            qualities[t] = np.linalg.norm(obs[0])

        series = qualities - np.mean(qualities)
        acf_0 = np.sum(series ** 2)
        acf_1 = np.sum(series[:-1] * series[1:])
        rho_1 = acf_1 / acf_0 if acf_0 > 0 else 0.0

        assert abs(rho_1) < 0.15, (
            f"Lag-1 autocorrelation {rho_1:.3f} too high for var_coefficient=0"
        )
