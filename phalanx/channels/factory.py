"""Channel factory: dispatch by config type string."""
from __future__ import annotations

import numpy as np

from phalanx.config import ChannelConfig
from phalanx.core import Channel
from phalanx.channels.gaussian import GaussianVARChannel
from phalanx.channels.markov import MarkovModulatedChannel
from phalanx.channels.nonstationary import NonstationaryChannel
from phalanx.channels.ntn import NTNOrbitalChannel


_CHANNEL_REGISTRY: dict[str, type] = {
    "gaussian": GaussianVARChannel,
    "markov": MarkovModulatedChannel,
    "nonstationary": NonstationaryChannel,
    "ntn": NTNOrbitalChannel,
}
# Note: ReplayChannel is not in the registry because it requires a traces
# array that cannot be expressed in ChannelConfig.  Instantiate it directly.


def create_channel(
    config: ChannelConfig,
    rng: np.random.Generator | None = None,
    seed: int = 42,
) -> Channel:
    """Instantiate a channel model from a configuration object.

    Args:
        config: Channel configuration dataclass.
        rng: Optional pre-seeded random generator.
        seed: Seed used if *rng* is None.

    Returns:
        An instance of the requested :class:`Channel` subclass.

    Raises:
        ValueError: If ``config.channel_type`` is not recognised.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    ctype = config.channel_type.lower()
    if ctype not in _CHANNEL_REGISTRY:
        raise ValueError(
            f"Unknown channel type: {config.channel_type!r}. "
            f"Choose from {list(_CHANNEL_REGISTRY)}"
        )
    return _CHANNEL_REGISTRY[ctype](config, rng)


def register_channel(name: str, cls: type) -> None:
    """Register a custom channel class for factory dispatch.

    Args:
        name: Key string (e.g. ``"my_channel"``).
        cls: A subclass of :class:`Channel`.
    """
    _CHANNEL_REGISTRY[name.lower()] = cls
