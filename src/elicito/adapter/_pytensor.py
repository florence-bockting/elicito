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
    # a reduction such as Sum also has a scalar_op; it keeps its own name
    if type(op).__name__ == "Elemwise":
        return type(op.scalar_op).__name__
    return type(getattr(op, "core_op", None) or op).__name__


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


def _batch_ndim(op: Any, x: Any) -> int:
    """Return the batch axes of the first input of a Blockwise; 0 for another op"""
    if not hasattr(op, "inputs_sig"):
        return 0
    return len(x.shape) - len(op.inputs_sig[0])


def _core(op: Any, i: int, x: Any) -> Any:
    """Input i of a Blockwise at batch element 0"""
    return x[(0,) * (len(x.shape) - len(op.inputs_sig[i]))]


def _reshape(op: Any, x: Any, shape: Any) -> Any:
    """Reshape x; a Blockwise keeps the batch axes of x"""
    nb = _batch_ndim(op, x)
    if nb:
        # every batch element has the same shape
        shape = tf.concat([tf.shape(x)[:nb], _core(op, 1, shape)], axis=0)
    return tf.reshape(x, shape)


def _alloc(op: Any, value: Any, *shape: Any) -> Any:
    """Broadcast value to shape; a Blockwise keeps the batch axes of value"""
    nb = _batch_ndim(op, value)
    if not nb:
        return tf.broadcast_to(value, tf.stack(shape))
    # every batch element has the same shape
    target = tf.stack([_core(op, i + 1, s) for i, s in enumerate(shape)])
    batch, core = tf.shape(value)[:nb], tf.shape(value)[nb:]
    # the core axes of value align with the last axes of the shape
    ones = tf.ones([len(shape) - (len(value.shape) - nb)], dtype=tf.int32)
    value = tf.reshape(value, tf.concat([batch, ones, core], axis=0))
    return tf.broadcast_to(value, tf.concat([batch, target], axis=0))


def _advanced_subtensor(op: Any, x: Any, *idx: Any) -> Any:
    """
    Index x as AdvancedSubtensor does

    The op may hold full slices and integer arrays at consecutive axes. A
    Blockwise may index axis 0 of its core with one integer array.
    """
    nb = _batch_ndim(op, x)
    if nb:
        if op.core_op.idx_list != (0,):
            msg = f"The adapter does not support the Blockwise index {op.core_op}."
            raise NotImplementedError(msg)
        [i] = idx
        batch = tf.broadcast_dynamic_shape(tf.shape(x)[:nb], tf.shape(i)[:nb])
        x = tf.broadcast_to(x, tf.concat([batch, tf.shape(x)[nb:]], axis=0))
        i = tf.broadcast_to(i, tf.concat([batch, tf.shape(i)[nb:]], axis=0))
        return tf.gather(x, i, axis=nb, batch_dims=nb)

    entries = op.idx_list
    is_array = [not isinstance(e, slice) for e in entries]
    first, n = is_array.index(True), sum(is_array)
    full = all(
        (e.start, e.stop, e.step) == (None, None, None)
        for e in entries
        if isinstance(e, slice)
    )
    keys = [idx[e] for e in entries[first : first + n]]
    if (
        not full
        or not all(is_array[first : first + n])
        or any(k.dtype == tf.bool for k in keys)
    ):
        msg = f"The adapter does not support the index {op}."
        raise NotImplementedError(msg)

    r = max(len(k.shape) for k in keys)
    shape = functools.reduce(tf.broadcast_dynamic_shape, [tf.shape(k) for k in keys])
    keys = [
        tf.broadcast_to(tf.where(k < 0, k + tf.shape(x)[first + j], k), shape)
        for j, k in enumerate(keys)
    ]
    # gather_nd indexes the leading axes, so move the indexed axes first
    rank = len(x.shape)
    perm = [*range(first, first + n), *range(first), *range(first + n, rank)]
    out = tf.gather_nd(tf.transpose(x, perm), tf.stack(keys, axis=-1))
    # move the sliced axes back in front of the index axes
    back = [*range(r, r + first), *range(r), *range(r + first, len(out.shape))]
    return tf.transpose(out, back)


def _broadcast(param: Any, size: Any, ndim: int) -> Any:
    """Broadcast an input of an RV to the size, plus its core shape"""
    core = tf.shape(param)[tf.rank(param) - ndim :]
    return tf.broadcast_to(param, tf.concat([size, core], axis=0))


def _rv(sample: Callable[..., Any]) -> Callable[[Any, list[Any], Any], Any]:
    """Rule for an RV: broadcast the parameters to the size, then sample"""

    def rule(op: Any, args: list[Any], seed: Any) -> Any:
        size, *params = args
        # every tfd parameter is a float, also the n of a Binomial
        params = [tf.cast(p, tf.float32) for p in params]
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
    "Sum": lambda op, args, seed: tf.reduce_sum(args[0], axis=op.axis),
    "Prod": lambda op, args, seed: tf.reduce_prod(args[0], axis=op.axis),
    "ArgSortOp": lambda op, args, seed: tf.argsort(
        args[0], axis=int(tf.get_static_value(args[1]))
    ),
    "Alloc": lambda op, args, seed: _alloc(op, *args),
    "Reshape": lambda op, args, seed: _reshape(op, *args),
    "AdvancedSubtensor": lambda op, args, seed: _advanced_subtensor(op, *args),
    "MakeVector": lambda op, args, seed: tf.stack(
        [tf.cast(arg, _tf_dtype(op.dtype)) for arg in args]
    ),
    "DimShuffle": lambda op, args, seed: _dimshuffle(op, args[0]),
    "Join": lambda op, args, seed: tf.concat(args, axis=op.axis),
    "Shape": lambda op, args, seed: tf.shape(args[0]),
    "Subtensor": lambda op, args, seed: _subtensor(op, *args),
    "ScalarFromTensor": lambda op, args, seed: args[0],
    "ViewOp": lambda op, args, seed: args[0],
    "NormalRV": _rv(lambda loc, scale, seed: tfd.Normal(loc, scale).sample(seed=seed)),
    "HalfNormalRV": _rv(
        lambda loc, scale, seed: loc + tfd.HalfNormal(scale).sample(seed=seed)
    ),
    "TruncatedNormalRV": _rv(
        lambda loc, scale, low, high, seed: tfd.TruncatedNormal(
            loc, scale, low, high
        ).sample(seed=seed)
    ),
    "UniformRV": _rv(lambda low, high, seed: tfd.Uniform(low, high).sample(seed=seed)),
    "BetaRV": _rv(lambda a, b, seed: tfd.Beta(a, b).sample(seed=seed)),
    # GammaRV and ExponentialRV take a scale; tfd takes a rate
    "GammaRV": _rv(
        lambda shape, scale, seed: tfd.Gamma(shape, 1 / scale).sample(seed=seed)
    ),
    "ExponentialRV": _rv(
        lambda scale, seed: tfd.Exponential(1 / scale).sample(seed=seed)
    ),
    "BernoulliRV": _rv(lambda p, seed: tfd.Bernoulli(probs=p).sample(seed=seed)),
    "BinomialRV": _rv(lambda n, p, seed: tfd.Binomial(n, probs=p).sample(seed=seed)),
    "CategoricalRV": _rv(lambda p, seed: tfd.Categorical(probs=p).sample(seed=seed)),
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
