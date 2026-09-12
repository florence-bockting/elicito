"""
Search box shared by the initialization and the CMA-ES search
"""

from typing import Any

import numpy as np

from elicito.types import Parameter

# The search uses a quarter of the training draws. A noisier objective is
# acceptable, because the result is only a start value.
SEARCH_FRACTION = 4
MIN_SEARCH_SAMPLES = 100

# Value reported for a set of hyperparameters that cannot be used. Nelder-Mead
# needs a finite number, and a usable point always scores far below this one.
PENALTY = 1e12


def hyper_names(parameters: list[Parameter]) -> list[str]:
    """
    List the hyperparameter names in the order the initializer uses

    Parameters
    ----------
    parameters
        List including dictionary with all information about the
        (hyper-)parameters.

    Returns
    -------
    names :
        Hyperparameter names, in the order of ``parameters``.

    """
    names: list[str] = []
    for param in parameters:
        hyperparams = param["hyperparams"]
        if hyperparams is None:
            continue
        for hyp in hyperparams:
            names.append(hyperparams[hyp]["name"])
    return names


def variable_names(variables: Any) -> list[str]:
    """
    Read the hyperparameter name of each trainable variable

    The prior model names a variable ``"<constraint>.<hyperparameter>"``.
    The order is the order in which the optimizer reads the variables.

    Parameters
    ----------
    variables
        Trainable variables of the prior model.

    Returns
    -------
    names :
        One hyperparameter name per variable.

    """
    return [str(var.name)[:-2].split(".")[1] for var in variables]


def box_vector(box: dict[str, Any], names: list[str], key: str) -> list[float]:
    """Read one value per hyperparameter out of one entry of the box."""
    entry = box[key]
    if np.isscalar(entry):
        return [float(entry)] * len(names)  # type: ignore [arg-type]
    order = box["hyper"] if box["hyper"] is not None else names
    lookup = dict(zip(order, entry))
    return [float(lookup[name]) for name in names]


def start_vector(box: dict[str, Any], names: list[str]) -> list[float]:
    """Read one start value per hyperparameter out of the box."""
    return box_vector(box, names, "mean")
