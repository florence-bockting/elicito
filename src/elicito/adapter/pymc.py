"""
Adapter for PyMC models
"""

from collections import Counter
from typing import Any

import tensorflow_probability as tfp  # type: ignore

from elicito.adapter._pytensor import translate
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


def parameters(model: Any) -> list[Parameter]:
    """
    Read the priors of a PyMC model

    Every hyperparameter must be a named `pm.Data` node. Its name becomes
    the name of the hyperparameter.

    Parameters
    ----------
    model
        PyMC model

    Returns
    -------
    :
        one parameter per free random variable, in the order of
        `model.free_RVs`

    Raises
    ------
    MissingOptionalDependencyError
        pymc is not installed

    NotImplementedError
        the adapter does not support the family of a prior

    ValueError
        a constant input of a prior is not the fixed value of the family

    TypeError
        an input of a prior is not a `pm.Data` node, and it is not a
        constant
    """
    try:
        from pytensor.compile.sharedvalue import SharedVariable
        from pytensor.graph.basic import Constant
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "adapter.pymc", requirement="pymc"
        ) from exc

    # a pm.Data node that feeds two priors is one shared hyperparameter
    uses = Counter(
        inp.name
        for rv in model.free_RVs
        for inp in rv.owner.op.dist_params(rv.owner)
        if isinstance(inp, SharedVariable)
    )

    params = []
    for rv in model.free_RVs:
        node = rv.owner
        op_name = type(node.op).__name__
        if op_name not in FAMILIES:
            msg = f"The adapter does not support {op_name}, the prior of '{rv.name}'."
            raise NotImplementedError(msg)

        family, entries = FAMILIES[op_name]
        hyperparams = {}
        for entry, inp in zip(entries, node.op.dist_params(node), strict=True):
            if isinstance(entry, str) and isinstance(inp, SharedVariable) and inp.name:
                lower = LOWER.get(entry, float("-inf"))
                hyperparams[entry] = hyper(
                    inp.name, lower=lower, shared=uses[inp.name] > 1
                )
            elif isinstance(entry, float) and isinstance(inp, Constant):
                if inp.data != entry:
                    msg = f"The prior of '{rv.name}' needs {entry}, not {inp.data}."
                    raise ValueError(msg)
            else:
                msg = f"An input of the prior of '{rv.name}' must be a pm.Data node."
                raise TypeError(msg)

        params.append(parameter(name=rv.name, family=family, hyperparams=hyperparams))
    return params


def model(model: Any) -> type:
    """
    Translate a PyMC model into a generative model for `el.model`

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
        from pytensor.graph.replace import vectorize_graph
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "adapter.pymc", requirement="pymc"
        ) from exc

    names = [v.name for v in model.observed_RVs + model.deterministics]
    # a batched input of shape (B, S) replaces each free random variable
    batched = [pt.matrix(rv.name) for rv in model.free_RVs]
    replace = dict(zip(model.free_RVs, batched, strict=True))
    outputs = vectorize_graph([model[name] for name in names], replace=replace)

    run = translate(outputs, batched)

    class PyMCModel:
        """Generative model translated from a PyMC model"""

        def __call__(self, prior_samples: Any, seed: Any) -> dict[str, Any]:
            args = [prior_samples[:, :, i] for i in range(len(batched))]
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
