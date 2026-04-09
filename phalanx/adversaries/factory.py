"""Adversary factory: dispatch by name string or config."""
from __future__ import annotations

from typing import Any

from phalanx.config import AdversaryConfig
from phalanx.core import Adversary
from phalanx.adversaries.none import NoAdversary
from phalanx.adversaries.budget import BudgetAdversary
from phalanx.adversaries.stackelberg import StackelbergAdversary
from phalanx.adversaries.reactive import ReactiveAdversary
from phalanx.adversaries.markov import MarkovAdversary


_ADVERSARY_REGISTRY: dict[str, type] = {
    "none": NoAdversary,
    "budget": BudgetAdversary,
    "stackelberg": StackelbergAdversary,
    "reactive": ReactiveAdversary,
    "markov": MarkovAdversary,
}


def create_adversary(
    name: str = "none",
    config: AdversaryConfig | None = None,
    **kwargs: Any,
) -> Adversary:
    """Instantiate an adversary by name or from a config.

    Can be called as:
    - ``create_adversary("budget", J_total=0.3)``
    - ``create_adversary(config=AdversaryConfig(adversary_type="budget"))``

    Args:
        name: Registry key (e.g. ``"stackelberg"``).
        config: Optional adversary config dataclass.  If provided and
            *name* is ``"none"``, the type is taken from ``config``.
        **kwargs: Forwarded to the constructor.

    Returns:
        An :class:`Adversary` instance.

    Raises:
        ValueError: If the name/type is not recognised.
    """
    # Resolve name from config if not explicitly given
    if config is not None and name == "none":
        name = config.adversary_type

    key = name.lower()
    if key not in _ADVERSARY_REGISTRY:
        raise ValueError(
            f"Unknown adversary: {name!r}. "
            f"Choose from {list(_ADVERSARY_REGISTRY)}"
        )
    cls = _ADVERSARY_REGISTRY[key]

    # Try config first, then kwargs
    if config is not None:
        try:
            return cls(config, **kwargs)
        except TypeError:
            pass

    if kwargs:
        return cls(**kwargs)
    return cls()


def register_adversary(name: str, cls: type) -> None:
    """Register a custom adversary class."""
    _ADVERSARY_REGISTRY[name.lower()] = cls
