"""
Adapter for PyMC models
"""

import functools
from collections.abc import Callable
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

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
                hyperparams[entry] = hyper(inp.name, lower=lower)
            elif isinstance(entry, float) and isinstance(inp, Constant):
                if inp.data != entry:
                    msg = f"The prior of '{rv.name}' needs {entry}, not {inp.data}."
                    raise ValueError(msg)
            else:
                msg = f"An input of the prior of '{rv.name}' must be a pm.Data node."
                raise TypeError(msg)

        params.append(parameter(name=rv.name, family=family, hyperparams=hyperparams))
    return params


def _op_key(op: Any) -> str:
    """Name of the op, or of the scalar or core op that it wraps"""
    inner = getattr(op, "scalar_op", None) or getattr(op, "core_op", None)
    return type(inner or op).__name__


def _inputs(node: Any) -> list[Any]:
    """Return the inputs that carry values; skip the rng and size of an RV"""
    if hasattr(node.op, "dist_params"):
        return list(node.op.dist_params(node))
    return list(node.inputs)


def _constant(var: Any) -> Any:
    """Value of a constant or of a pm.Data node"""
    data = var.get_value() if hasattr(var, "get_value") else var.data
    return tf.constant(data, dtype=tf.float32)


def _dimshuffle(op: Any, x: Any) -> Any:
    """Drop, permute and add axes, as a PyTensor DimShuffle does"""
    if op.drop:
        x = tf.squeeze(x, axis=list(op.drop))
    x = tf.transpose(x, perm=list(op.shuffle))
    for axis in op.augment:
        x = tf.expand_dims(x, axis)
    return x


# op name -> TensorFlow rule; a rule takes the op, its input values and a seed
RULES: dict[str, Callable[[Any, list[Any], Any], Any]] = {
    "Add": lambda op, args, seed: functools.reduce(tf.add, args),
    "Mul": lambda op, args, seed: functools.reduce(tf.multiply, args),
    "DimShuffle": lambda op, args, seed: _dimshuffle(op, args[0]),
    "ViewOp": lambda op, args, seed: args[0],
    "NormalRV": lambda op, args, seed: tfd.Normal(*args).sample(seed=seed),
}


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

    # nodes in topological order; each node follows its inputs
    order: list[Any] = []

    def visit(var: Any) -> None:
        node = var.owner
        if node is None or node in order:
            return
        for inp in _inputs(node):
            visit(inp)
        order.append(node)

    for out in outputs:
        visit(out)

    for node in order:
        if _op_key(node.op) not in RULES:
            msg = f"The adapter does not support the op {_op_key(node.op)}."
            raise NotImplementedError(msg)
    rv_nodes = [node for node in order if hasattr(node.op, "dist_params")]

    class PyMCModel:
        """Generative model translated from a PyMC model"""

        def __call__(self, prior_samples: Any, seed: Any) -> dict[str, Any]:
            values: dict[Any, Any] = {
                x: prior_samples[:, :, i] for i, x in enumerate(batched)
            }
            seeds = tfp.random.split_seed(seed, n=max(1, len(rv_nodes)), salt="pymc")
            rv_seeds = dict(zip(rv_nodes, seeds))
            for node in order:
                args = [
                    values[v] if v in values else _constant(v) for v in _inputs(node)
                ]
                rule = RULES[_op_key(node.op)]
                values[node.default_output()] = rule(node.op, args, rv_seeds.get(node))
            return {name: values[out] for name, out in zip(names, outputs, strict=True)}

    return PyMCModel
