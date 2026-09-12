"""
Seed helpers and constraints shared by the prior-learning methods
"""

import contextlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import tensorflow as tf

from elicito.types import (
    Parameter,
)


@contextlib.contextmanager
def numpy_seed(seed: int) -> Iterator[None]:
    """
    Seed the numpy global generator for the duration of the block

    A network can draw from the numpy global generator while it is built.
    `elicito.parameters.networks.Permutation` does, and `tf.random.set_seed` does not
    control that generator. Two networks built from the same seed then
    permute differently, and a fitted network cannot be rebuilt from its
    stored weights.

    The generator of the caller is restored on exit.

    Parameters
    ----------
    seed
        Seed of the current workflow run.

    Yields
    ------
    :
        None. The block runs with the seeded generator.
    """
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(seed)  # noqa: NPY002
    try:
        yield
    finally:
        np.random.set_state(state)  # noqa: NPY002


def seed_pair(seed: int) -> Any:
    """
    Turn an integer seed into a stateless seed pair

    A stateless pair makes every draw a function of the seed alone. The draws
    then repeat without ``tf.random.set_seed``, which a compiled forward pass
    cannot use: it clears the kernel caches of the whole graph.

    ``tfp.random.split_seed`` accepts an integer, but it turns one into a pair
    with a stateful ``tf.random.uniform``. That op branches on a value, which
    a graph cannot do, and it would make the draws differ between calls.

    Parameters
    ----------
    seed
        Seed of the current workflow run.

    Returns
    -------
    pair :
        Seed as a pair of integers.

    """
    return tf.constant([0, seed], dtype=tf.int32)


def _constraints(parameters: list[Parameter]) -> dict[str, Any]:
    """Map each hyperparameter name to its constraint function."""
    constraints: dict[str, Any] = {}
    for param in parameters:
        hyperparams = param["hyperparams"]
        if hyperparams is None:
            continue
        for hyp in hyperparams:
            constraints[hyperparams[hyp]["name"]] = hyperparams[hyp]["constraint"]
    return constraints
