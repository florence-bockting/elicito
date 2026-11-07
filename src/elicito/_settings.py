"""
Turn the flat settings of Elicit into the internal settings
"""

import warnings
from typing import Any

import tensorflow as tf

from elicito import specs
from elicito.optimizers.cmaes import CMAES
from elicito.types import Trainer

AUTO = "auto"
ADAM = "adam"
ADAM_LEARNING_RATE = 0.1


def method(network: Any) -> str:
    """Infer the method: a network means a deep prior"""
    return "parametric_prior" if network is None else "deep_prior"


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
    """Build the trainer dictionary"""
    return specs.trainer(
        method=method(network),  # type: ignore [arg-type]
        seed=seeds[0],
        epochs=epochs,
        B=B,
        num_samples=num_samples,
        progress=int(verbose),  # type: ignore [arg-type]
        kappa=kappa,
    )
