"""
Translate a PyTensor graph into TensorFlow

This module must not import elicito.
"""

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

tfd = tfp.distributions


def _op_key(op: Any) -> str:
    """Name of the op, or of the scalar or core op that it wraps"""
    inner = getattr(op, "scalar_op", None) or getattr(op, "core_op", None)
    return type(inner or op).__name__


def _inputs(node: Any) -> list[Any]:
    """Return the inputs that carry values: for an RV, its size and parameters"""
    if hasattr(node.op, "dist_params"):
        return [node.op.size_param(node), *node.op.dist_params(node)]
    return list(node.inputs)


def _tf_dtype(dtype: str) -> Any:
    """TensorFlow dtype of a PyTensor dtype: bool, int32 or float32"""
    if dtype == "bool":
        return tf.bool
    if np.issubdtype(np.dtype(dtype), np.integer):
        return tf.int32
    return tf.float32


def _constant(var: Any) -> Any:
    """
    Value of a constant or of a pm.Data node; integers stay integers

    The size of an RV without a size is the constant None.
    """
    data = var.get_value() if hasattr(var, "get_value") else var.data
    if data is None:
        return None
    return tf.constant(data, dtype=_tf_dtype(var.dtype))


def _upcast(node: Any, args: list[Any]) -> list[Any]:
    """Give the numeric inputs of an elementwise op one dtype, as PyTensor does"""
    if not hasattr(node.op, "scalar_op"):
        return args
    numeric = [inp.dtype != "bool" for inp in node.inputs]
    dtypes = [inp.dtype for inp, num in zip(node.inputs, numeric) if num]
    if not dtypes:
        return args
    dtype = _tf_dtype(np.result_type(*dtypes).name)
    return [
        tf.cast(arg, dtype) if num else arg
        for arg, num in zip(args, numeric, strict=True)
    ]


def _dimshuffle(op: Any, x: Any) -> Any:
    """Drop, permute and add axes, as a PyTensor DimShuffle does"""
    if op.drop:
        x = tf.squeeze(x, axis=list(op.drop))
    x = tf.transpose(x, perm=list(op.shuffle))
    for axis in op.augment:
        x = tf.expand_dims(x, axis)
    return x


def _subtensor(op: Any, x: Any, *idx: Any) -> Any:
    """Index x as Subtensor does; the idx_list of the op holds positions in idx"""

    def pick(pos: int | None) -> Any:
        return None if pos is None else idx[pos]

    key = tuple(
        slice(pick(e.start), pick(e.stop), pick(e.step))
        if isinstance(e, slice)
        else idx[e]
        for e in op.idx_list
    )
    return x[key]


def _broadcast(param: Any, size: Any, ndim: int) -> Any:
    """Broadcast an input of an RV to the size, plus its core shape"""
    core = tf.shape(param)[tf.rank(param) - ndim :]
    return tf.broadcast_to(param, tf.concat([size, core], axis=0))


def _rv(sample: Callable[..., Any]) -> Callable[[Any, list[Any], Any], Any]:
    """Rule for an RV: broadcast the parameters to the size, then sample"""

    def rule(op: Any, args: list[Any], seed: Any) -> Any:
        size, *params = args
        if size is not None:
            params = [
                _broadcast(p, size, ndim)
                for p, ndim in zip(params, op.ndims_params, strict=True)
            ]
        return sample(*params, seed=seed)

    return rule


# op name -> TensorFlow rule; a rule takes the op, its input values and a seed.
# The input values of an RV are its size (None if not given) and parameters.
RULES: dict[str, Callable[[Any, list[Any], Any], Any]] = {
    "Add": lambda op, args, seed: functools.reduce(tf.add, args),
    "Mul": lambda op, args, seed: functools.reduce(tf.multiply, args),
    "Sub": lambda op, args, seed: tf.subtract(*args),
    "Neg": lambda op, args, seed: tf.negative(args[0]),
    "TrueDiv": lambda op, args, seed: tf.divide(*args),
    "Reciprocal": lambda op, args, seed: tf.math.reciprocal(args[0]),
    "Pow": lambda op, args, seed: tf.pow(*args),
    "Exp": lambda op, args, seed: tf.exp(args[0]),
    "Log": lambda op, args, seed: tf.math.log(args[0]),
    "Sqrt": lambda op, args, seed: tf.sqrt(args[0]),
    "Abs": lambda op, args, seed: tf.abs(args[0]),
    "Sign": lambda op, args, seed: tf.sign(args[0]),
    "Erf": lambda op, args, seed: tf.math.erf(args[0]),
    "Sigmoid": lambda op, args, seed: tf.sigmoid(args[0]),
    "Minimum": lambda op, args, seed: tf.minimum(*args),
    "Maximum": lambda op, args, seed: tf.maximum(*args),
    "Clip": lambda op, args, seed: tf.minimum(tf.maximum(args[0], args[1]), args[2]),
    "Switch": lambda op, args, seed: tf.where(tf.cast(args[0], tf.bool), *args[1:]),
    "EQ": lambda op, args, seed: tf.equal(*args),
    "LT": lambda op, args, seed: tf.less(*args),
    "GT": lambda op, args, seed: tf.greater(*args),
    # the translated value takes the dtype of the PyTensor output
    "Cast": lambda op, args, seed: args[0],
    "Second": lambda op, args, seed: args[1] + tf.zeros_like(args[0]),
    "DimShuffle": lambda op, args, seed: _dimshuffle(op, args[0]),
    "Join": lambda op, args, seed: tf.concat(args, axis=op.axis),
    "Shape": lambda op, args, seed: tf.shape(args[0]),
    "Subtensor": lambda op, args, seed: _subtensor(op, *args),
    "ScalarFromTensor": lambda op, args, seed: args[0],
    "ViewOp": lambda op, args, seed: args[0],
    "NormalRV": _rv(lambda loc, scale, seed: tfd.Normal(loc, scale).sample(seed=seed)),
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
            node_args = _upcast(node, node_args)
            rule = RULES[_op_key(node.op)]
            out = node.default_output()
            value = rule(node.op, node_args, rv_seeds.get(node))
            # the TensorFlow value takes the dtype of the PyTensor output
            values[out] = tf.cast(value, _tf_dtype(out.dtype))
        return [values[out] for out in outputs]

    return run
