"""
Translate a PyTensor graph into TensorFlow

This module must not import elicito.
"""

import functools
from collections.abc import Callable
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

tfd = tfp.distributions


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


def _vectorize_rv(op: Any, node: Any, rng: Any, size: Any, *params: Any) -> Any:
    """
    Batch a random variable; take the new axes from every input

    PyTensor's `vectorize_random_variable` takes them from the first input
    only. `Binomial(n, p)` with a constant `n` then keeps its old size.
    """
    from pytensor.graph.basic import equal_computations
    from pytensor.tensor.basic import concatenate
    from pytensor.tensor.random.utils import compute_batch_shape
    from pytensor.tensor.type_other import NoneTypeT

    old_size = op.size_param(node)
    if not isinstance(old_size.type, NoneTypeT) and equal_computations(
        [old_size], [size]
    ):
        new_ndim = max(
            (
                new.type.ndim - old.type.ndim
                for new, old in zip(params, op.dist_params(node), strict=True)
            ),
            default=0,
        )
        if new_ndim > 0:
            batch = compute_batch_shape(params, ndims_params=op.ndims_params)
            size = concatenate([batch[:new_ndim], size])  # type: ignore[no-untyped-call]
    return op.make_node(rng, size, *params)


def vectorize(outputs: list[Any], replace: dict[Any, Any]) -> list[Any]:
    """
    Add batch axes on the left of a PyTensor graph

    This is `pytensor.graph.replace.vectorize_graph`, with a fix for
    random variables (see `_vectorize_rv`).

    Parameters
    ----------
    outputs
        PyTensor variables to batch

    replace
        each input variable, and the batched variable that replaces it

    Returns
    -------
    :
        the batched outputs
    """
    from pytensor.graph.basic import Apply
    from pytensor.graph.replace import _vectorize_node
    from pytensor.graph.traversal import toposort, truncated_graph_inputs
    from pytensor.tensor.random.op import RandomVariable

    inputs = truncated_graph_inputs(outputs, ancestors_to_include=replace.keys())
    batched = {inp: replace.get(inp, inp) for inp in inputs}
    for node in toposort(outputs, blockers=inputs):
        args = [batched.get(inp, inp) for inp in node.inputs]
        if isinstance(node.op, RandomVariable):
            new = _vectorize_rv(node.op, node, *args)
        else:
            new = _vectorize_node(node.op, node, *args)
        new_outputs = new.outputs if isinstance(new, Apply) else new
        for out, new_out in zip(node.outputs, new_outputs, strict=True):
            # keep a replacement that the caller gave
            batched.setdefault(out, new_out)
    return [batched[out] for out in outputs]


def translate(
    outputs: list[Any], inputs: list[Any]
) -> Callable[[list[Any], Any], list[Any]]:
    """
    Translate a PyTensor graph into a TensorFlow function

    Parameters
    ----------
    outputs
        PyTensor variables to compute

    inputs
        PyTensor variables whose values the caller gives

    Returns
    -------
    :
        function that takes the input values and a seed, and returns
        the output values

    Raises
    ------
    NotImplementedError
        the graph has an op without a rule
    """
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

    def run(args: list[Any], seed: Any) -> list[Any]:
        values: dict[Any, Any] = dict(zip(inputs, args, strict=True))
        seeds = tfp.random.split_seed(seed, n=max(1, len(rv_nodes)), salt="pymc")
        rv_seeds = dict(zip(rv_nodes, seeds))
        for node in order:
            node_args = [
                values[v] if v in values else _constant(v) for v in _inputs(node)
            ]
            rule = RULES[_op_key(node.op)]
            values[node.default_output()] = rule(node.op, node_args, rv_seeds.get(node))
        return [values[out] for out in outputs]

    return run
