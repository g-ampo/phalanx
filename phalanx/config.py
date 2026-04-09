"""Dataclass configurations with YAML/JSON serialization.

Nested dataclasses with sane defaults,
property shortcuts, and round-trip serialization.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Channel configuration
# ---------------------------------------------------------------------------

@dataclass
class ChannelConfig:
    """Abstract stochastic channel configuration."""

    M: int = 8
    """Number of channels / links."""
    d_obs: int = 4
    """Observation dimension per channel."""
    channel_type: str = "gaussian"
    """One of ``"gaussian"``, ``"markov"``, ``"nonstationary"``, ``"ntn"``."""

    # -- Gaussian VAR(1) --
    var_coefficient: float = 0.9
    """AR(1) temporal-correlation coefficient."""
    noise_std: float = 0.15
    """Observation noise standard deviation."""
    cross_correlation: float = 0.3
    """Cross-channel correlation strength."""
    quality_scale: float = 10.0
    """Scale factor for channel quality signal."""

    # -- Markov-modulated --
    n_states: int = 3
    """Number of hidden states per channel."""
    transition_prob: float = 0.05
    """State transition probability."""

    # -- Non-stationary --
    drift_rate: float = 0.001
    """Sinusoidal drift rate for non-stationary channels."""

    # -- NTN / LEO --
    altitude_km: float = 550.0
    """LEO satellite altitude in km."""
    inclination_deg: float = 53.0
    """Orbital inclination in degrees."""
    carrier_freq_ghz: float = 2.0
    """Carrier frequency in GHz (S-band default)."""
    elevation_min_deg: float = 10.0
    """Minimum elevation angle for visibility."""


# ---------------------------------------------------------------------------
# Scheduler configuration
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    """Scheduler / optimization configuration."""

    scheduler_type: str = "lyapunov_dpp"
    """One of ``"lyapunov_dpp"``, ``"lyapunov_only"``, ``"ucb"``,
    ``"thompson"``, ``"round_robin"``, ``"greedy"``, ``"random"``,
    ``"oracle"``, ``"whittle"``, ``"aoi_index"``, ``"eps_equalization"``."""

    V: float = 5.0
    """Lyapunov drift-plus-penalty tradeoff parameter."""
    F: int = 4
    """Number of demand streams / flows."""
    alpha: float = 0.3
    """Prediction weight in [0, 1] for prediction-augmented DPP."""
    lambda_mean: float = 0.35
    """Mean arrival rate per stream."""

    # -- Bandit --
    ucb_c: float = 2.0
    """UCB exploration constant."""

    # -- AoI --
    N_sources: int = 4
    """Number of AoI sources."""
    p_succ_base: float = 0.8
    """Base transmission success probability."""
    e_tx: float = 1.0
    """Energy cost per transmission."""
    harvest_rate: float = 0.3
    """Energy harvest rate per slot."""

    # -- Epsilon equalization --
    pi_J: float = 0.3
    """Jammer activity probability."""
    jam_damage: float = 0.5
    """Jamming damage (success probability reduction under jamming)."""


# ---------------------------------------------------------------------------
# Adversary configuration
# ---------------------------------------------------------------------------

@dataclass
class AdversaryConfig:
    """Adversary / jammer configuration."""

    adversary_type: str = "none"
    """One of ``"none"``, ``"budget"``, ``"stackelberg"``, ``"reactive"``,
    ``"markov"``."""

    J_total: float = 0.5
    """Total adversary budget (power / probability)."""
    adaptive_delay: int = 1
    """Observation delay for adaptive adversary."""
    n_outer: int = 15
    """Stackelberg alternating best-response iterations."""
    lr: float = 0.05
    """Stackelberg leader projected-gradient learning rate."""

    # -- Markov adversary --
    n_jammer_states: int = 3
    """Number of Markov jammer states."""
    p_switch: float = 0.1
    """Markov state transition probability."""

    # -- Reactive --
    observation_noise: float = 0.0
    """Noise on adversary's observation of scheduler action."""


# ---------------------------------------------------------------------------
# Predictor configuration
# ---------------------------------------------------------------------------

@dataclass
class PredictorConfig:
    """Predictor configuration.

    Defines the predictor type, prediction horizon, and observation
    window size.  Specific predictor implementations (e.g. VAE, Kalman)
    extend this via their own config classes registered downstream.
    """

    predictor_type: str = "none"
    """One of ``"none"``, ``"persistence"``.  Extended by downstream packages."""
    tau: int = 5
    """Prediction horizon (time steps ahead)."""
    obs_window_size: int = 20
    """Number of past observations fed to the predictor's ``encode()``."""


# ---------------------------------------------------------------------------
# Top-level simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Master simulation configuration."""

    T: int = 10_000
    """Number of time slots."""
    seed: int = 42
    """Base random seed."""
    n_seeds: int = 30
    """Number of independent seeds for confidence intervals."""
    burn_in: int = 1000
    """Burn-in slots discarded before metric aggregation."""
    n_workers: int = 4
    """Number of parallel workers for multi-seed runs."""

    channel: ChannelConfig = field(default_factory=ChannelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    adversary: AdversaryConfig = field(default_factory=AdversaryConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    @property
    def M(self) -> int:
        return self.channel.M

    @property
    def F(self) -> int:
        return self.scheduler.F

    @property
    def d_obs_total(self) -> int:
        return self.channel.M * self.channel.d_obs

    # -- Serialization --

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (recursively)."""
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Write configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Union[str, Path]) -> None:
        """Write configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimConfig":
        """Construct from a plain dict (does not mutate the input)."""
        d = dict(d)  # shallow copy to avoid mutating caller's dict
        ch = ChannelConfig(**d.pop("channel", {}))
        sc = SchedulerConfig(**d.pop("scheduler", {}))
        ad = AdversaryConfig(**d.pop("adversary", {}))
        pr = PredictorConfig(**d.pop("predictor", {}))
        return cls(channel=ch, scheduler=sc, adversary=ad, predictor=pr, **d)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SimConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "SimConfig":
        """Load configuration from a JSON file."""
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)
