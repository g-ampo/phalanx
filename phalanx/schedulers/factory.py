"""Scheduler factory: dispatch by name string."""
from __future__ import annotations

from typing import Any, Dict

from phalanx.config import SchedulerConfig
from phalanx.core import Scheduler
from phalanx.schedulers.lyapunov import LyapunovDPP, LyapunovOnly
from phalanx.schedulers.bandits import UCBScheduler, ThompsonSamplingScheduler
from phalanx.schedulers.baselines import RoundRobin, Greedy, Random, Oracle
from phalanx.schedulers.whittle import WhittleIndex
from phalanx.schedulers.index_aoi import AoIIndexPolicy, EpsilonEqualization


_SCHEDULER_REGISTRY: dict[str, type] = {
    "lyapunov_dpp": LyapunovDPP,
    "lyapunov_only": LyapunovOnly,
    "ucb": UCBScheduler,
    "thompson": ThompsonSamplingScheduler,
    "round_robin": RoundRobin,
    "greedy": Greedy,
    "random": Random,
    "oracle": Oracle,
    "whittle": WhittleIndex,
    "aoi_index": AoIIndexPolicy,
    "eps_equalization": EpsilonEqualization,
}


def create_scheduler(
    name: str,
    config: SchedulerConfig | None = None,
    **kwargs: Any,
) -> Scheduler:
    """Instantiate a scheduler by name.

    Args:
        name: Registry key (e.g. ``"lyapunov_dpp"``).
        config: Optional scheduler config dataclass.
        **kwargs: Forwarded to the constructor.

    Returns:
        A :class:`Scheduler` instance.

    Raises:
        ValueError: If *name* is not recognised.
    """
    key = name.lower()
    if key not in _SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler: {name!r}. "
            f"Choose from {list(_SCHEDULER_REGISTRY)}"
        )
    cls = _SCHEDULER_REGISTRY[key]

    # Some schedulers accept a config object as first arg
    if config is not None:
        try:
            return cls(config, **kwargs)
        except TypeError:
            pass

    if kwargs:
        return cls(**kwargs)
    return cls()


def register_scheduler(name: str, cls: type) -> None:
    """Register a custom scheduler class."""
    _SCHEDULER_REGISTRY[name.lower()] = cls
