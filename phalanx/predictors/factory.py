"""Predictor factory: dispatch by config type string."""
from __future__ import annotations

from phalanx.predictors.base import Predictor
from phalanx.predictors.persistence import PersistencePredictor


_PREDICTOR_REGISTRY: dict[str, type] = {
    "persistence": PersistencePredictor,
}


def create_predictor(predictor_type: str, **kwargs) -> "Predictor | None":
    """Instantiate a predictor from a type string.

    Args:
        predictor_type: Registry key (e.g. ``"persistence"``).
            Use ``"none"`` to indicate no predictor.
        **kwargs: Forwarded to the predictor constructor.

    Returns:
        An instance of the requested :class:`Predictor` subclass,
        or ``None`` if ``predictor_type`` is ``"none"``.

    Raises:
        ValueError: If ``predictor_type`` is not recognised.
    """
    ptype = predictor_type.lower()
    if ptype == "none":
        return None
    if ptype not in _PREDICTOR_REGISTRY:
        raise ValueError(
            f"Unknown predictor type: {predictor_type!r}. "
            f"Choose from {list(_PREDICTOR_REGISTRY)}"
        )
    return _PREDICTOR_REGISTRY[ptype](**kwargs)


def register_predictor(name: str, cls: type) -> None:
    """Register a custom predictor class for factory dispatch.

    This is the extension point for downstream packages.  For example,
    a downstream package can register a VAE predictor::

        from phalanx.predictors import register_predictor
        register_predictor("vae", MyVAEPredictor)

    Args:
        name: Key string (e.g. ``"vae"``).
        cls: A subclass of :class:`Predictor`.
    """
    _PREDICTOR_REGISTRY[name.lower()] = cls
