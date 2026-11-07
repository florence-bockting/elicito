"""
Turn the flat settings of Elicit into the internal settings
"""

import sys
import warnings
from collections.abc import Callable
from numbers import Integral
from typing import Any

import tensorflow as tf

from elicito.optimizers.cmaes import CMAES
from elicito.types import Parameter, Trainer

AUTO = "auto"
ADAM = "adam"
ADAM_LEARNING_RATE = 0.1


def model(
    model: Any, parameters: list[Parameter] | None
) -> tuple[Callable[..., Any], list[Parameter]]:
    """
    Translate a PyMC model, and read its priors if parameters is None
    """
    # a user with a PyMC model has imported pymc
    pm = sys.modules.get("pymc")
    if pm is not None and isinstance(model, pm.Model):
        from elicito.adapter import pymc as adapter_pymc

        if parameters is None:
            parameters = adapter_pymc.parameters(model)
        return adapter_pymc.model(model)(), parameters
    if parameters is None:
        msg = "A callable model needs parameters=[el.parameter(...), ...]."
        raise ValueError(msg)
    return model, parameters


def method(network: Any) -> str:
    """Infer the method: a network means a deep prior"""
    return "parametric_prior" if network is None else "deep_prior"


def seeds(seeds: Any) -> tuple[int, ...]:
    """Check the seeds: a non-empty sequence of integers"""
    if (
        isinstance(seeds, Integral)
        or len(seeds) == 0
        or not all(isinstance(s, Integral) for s in seeds)
    ):
        msg = (
            "seeds must be a non-empty tuple of integers, for example "
            f"seeds=(0, 1). Got {seeds=}."
        )
        raise ValueError(msg)
    return tuple(int(s) for s in seeds)


def optimizer(
    optimizer: Any, cmaes: dict[str, Any] | None, network: Any
) -> dict[str, Any]:
    """Build the optimizer dictionary: the class and its settings"""
    if isinstance(optimizer, str) and optimizer == AUTO:
        optimizer = ADAM if network is not None else CMAES
    if cmaes is not None and optimizer != CMAES:
        warnings.warn(
            f"cmaes={cmaes} is ignored, because the optimizer is not '{CMAES}'.",
            UserWarning,
            stacklevel=4,
        )
    if isinstance(optimizer, tf.keras.optimizers.Optimizer):
        return dict(optimizer=type(optimizer), **optimizer.get_config())
    if optimizer == CMAES:
        return dict(optimizer=CMAES, **(cmaes or {}))
    if optimizer == ADAM:
        return dict(
            optimizer=tf.keras.optimizers.Adam, learning_rate=ADAM_LEARNING_RATE
        )
    msg = (
        f"optimizer must be '{AUTO}', '{CMAES}', '{ADAM}' or a tf.keras "
        f"optimizer. Got {optimizer=}."
    )
    raise ValueError(msg)


def trainer(  # noqa: PLR0913
    network: Any,
    seeds: tuple[int, ...],
    epochs: int,
    B: int,
    num_samples: int,
    kappa: float,
    verbose: bool,
) -> Trainer:
    """Check the training settings and build the trainer dictionary"""
    if epochs <= 0:
        msg = f"The number of epochs has to be greater 0. Got {epochs=}."
        raise ValueError(msg)
    if kappa < 0:
        msg = f"The weight kappa cannot be negative. Got {kappa=}."
        raise ValueError(msg)
    return dict(
        method=method(network),
        seed=seeds[0],
        B=int(B),
        num_samples=int(num_samples),
        epochs=int(epochs),
        progress=int(verbose),
        kappa=float(kappa),
    )
