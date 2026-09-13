"""
Adapter for PyMC models
"""

from collections import Counter
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito.adapter._pytensor import translate, vectorize
from elicito.exceptions import MissingOptionalDependencyError
from elicito.specs import hyper, parameter
from elicito.types import Parameter

tfd = tfp.distributions

# PyTensor random variable -> tfd family, and one entry per input of the
# random variable: the tfd argument name, or a value the input must equal
FAMILIES: dict[str, tuple[Any, tuple[str | float, ...]]] = {
    "NormalRV": (tfd.Normal, ("loc", "scale")),
    "HalfNormalRV": (tfd.HalfNormal, (0.0, "scale")),
}
# lower bound of a hyperparameter, by tfd argument name
LOWER = {"scale": 0.0}


def _unwrap(inp: Any) -> Any:
    """Remove the axes that PyMC adds to each input of a prior with a shape"""
    while inp.owner is not None and type(inp.owner.op).__name__ == "DimShuffle":
        inp = inp.owner.inputs[0]
    return inp


def _size(rv: Any) -> int:
    """Return the number of scalars in a prior; its shape must be fixed"""
    if None in rv.type.shape:
        msg = f"The prior '{rv.name}' needs a fixed shape."
        raise ValueError(msg)
    return int(np.prod(rv.type.shape))


def _is_prior(rv: Any) -> bool:
    """Tell if a free random variable is a prior: a `pm.Data` node is an input"""
    from pytensor.compile.sharedvalue import SharedVariable

    params = rv.owner.op.dist_params(rv.owner)
    return any(isinstance(_unwrap(inp), SharedVariable) for inp in params)


def _specs(rv: Any) -> tuple[Any, dict[str, tuple[str, float]]]:
    """
    Check the inputs of a prior

    Return the tfd family, and for each tfd argument the name of its
    `pm.Data` node and its lower bound.
    """
    from pytensor.compile.sharedvalue import SharedVariable
    from pytensor.graph.basic import Constant

    node = rv.owner
    op_name = type(node.op).__name__
    if op_name not in FAMILIES:
        msg = f"The adapter does not support {op_name}, the prior of '{rv.name}'."
        raise NotImplementedError(msg)

    family, entries = FAMILIES[op_name]
    specs = {}
    for entry, inp in zip(
        entries, map(_unwrap, node.op.dist_params(node)), strict=True
    ):
        is_data = isinstance(inp, SharedVariable) and bool(inp.name)
        if isinstance(entry, str) and is_data and inp.type.ndim == 0:
            specs[entry] = (inp.name, LOWER.get(entry, float("-inf")))
        elif isinstance(entry, float) and isinstance(inp, Constant):
            if np.any(inp.data != entry):
                msg = f"The prior of '{rv.name}' needs {entry}, not {inp.data}."
                raise ValueError(msg)
        else:
            msg = (
                f"An input of the prior of '{rv.name}' must be a scalar "
                "pm.Data node."
            )
            raise TypeError(msg)
    return family, specs


def parameters(model: Any) -> list[Parameter]:
    """
    Read the priors of a PyMC model

    A free random variable with a `pm.Data` input is a prior. Every input
    of a prior must be a named scalar `pm.Data` node, or the fixed value of
    its family. The name of the node becomes the name of the hyperparameter.
    A prior with k elements becomes k parameters, `name[0]` to
    `name[k-1]`, that share the hyperparameters. The generative model
    samples every other free random variable.

    Parameters
    ----------
    model
        PyMC model

    Returns
    -------
    :
        one parameter per prior, in the order of `model.free_RVs`

    Raises
    ------
    MissingOptionalDependencyError
        pymc is not installed

    NotImplementedError
        the adapter does not support the family of a prior

    ValueError
        a constant input of a prior is not the fixed value of the family,
        or the shape of a prior is not fixed

    TypeError
        an input of a prior is not a scalar `pm.Data` node, and it is not
        a constant
    """
    try:
        import pytensor  # noqa: F401
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "adapter.pymc", requirement="pymc"
        ) from exc

    priors = [rv for rv in model.free_RVs if _is_prior(rv)]
    checked = [_specs(rv) for rv in priors]
    # a pm.Data node that feeds two priors, or a prior with k > 1 elements,
    # is one shared hyperparameter
    uses: Counter[str] = Counter()
    for rv, (_, specs) in zip(priors, checked, strict=True):
        for hp_name, _ in specs.values():
            uses[hp_name] += _size(rv)

    params = []
    for rv, (family, specs) in zip(priors, checked, strict=True):
        if rv.type.ndim == 0:
            names = [rv.name]
        else:
            names = [f"{rv.name}[{i}]" for i in range(_size(rv))]
        for name in names:
            hyperparams = {
                entry: hyper(hp_name, lower=lower, shared=uses[hp_name] > 1)
                for entry, (hp_name, lower) in specs.items()
            }
            params.append(parameter(name=name, family=family, hyperparams=hyperparams))
    return params


def model(model: Any) -> type:
    """
    Translate a PyMC model into a generative model for `el.model`

    The generative model takes the samples of the priors that `parameters`
    reads. It samples every other free random variable itself.

    Parameters
    ----------
    model
        PyMC model

    Returns
    -------
    :
        generative model class; it returns every observed random variable
        and every `pm.Deterministic`, by name

    Raises
    ------
    MissingOptionalDependencyError
        pymc is not installed

    NotImplementedError
        the adapter does not support an op of the model
    """
    try:
        import pytensor.tensor as pt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "adapter.pymc", requirement="pymc"
        ) from exc

    names = [v.name for v in model.observed_RVs + model.deterministics]
    # a batched input of shape (B, S, *shape) replaces each prior
    priors = [rv for rv in model.free_RVs if _is_prior(rv)]
    batched = [
        pt.tensor(rv.name, shape=(None, None, *rv.type.shape), dtype=rv.dtype)
        for rv in priors
    ]
    replace = dict(zip(priors, batched, strict=True))
    outputs = vectorize([model[name] for name in names], replace)

    run = translate(outputs, batched)
    # the columns of prior_samples that hold each prior, in the order of
    # the parameters
    ends = np.cumsum([_size(rv) for rv in priors]).tolist()
    starts = [0, *ends][:-1]
    shapes = [tf.constant(rv.type.shape, dtype=tf.int32) for rv in priors]

    class PyMCModel:
        """Generative model translated from a PyMC model"""

        def __call__(self, prior_samples: Any, seed: Any) -> dict[str, Any]:
            batch = tf.shape(prior_samples)[:2]
            args = [
                tf.reshape(prior_samples[:, :, a:b], tf.concat([batch, shape], 0))
                for a, b, shape in zip(starts, ends, shapes, strict=True)
            ]
            return dict(zip(names, run(args, seed), strict=True))

    return PyMCModel


def set_hyperparameters(model: Any, eliobj: Any, replication: int = 0) -> None:
    """
    Write the learned hyperparameters into the PyMC model

    Each learned value replaces the value of the `pm.Data` node with the
    same name. The PyMC model then fits with the elicited priors.

    Parameters
    ----------
    model
        PyMC model that `parameters` read

    eliobj
        fitted Elicit object

    replication
        index of the replication

    Raises
    ------
    MissingOptionalDependencyError
        pymc is not installed
    """
    try:
        import pymc as pm
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "adapter.pymc", requirement="pymc"
        ) from exc

    pm.set_data(eliobj.hyperparameters(replication), model=model)
