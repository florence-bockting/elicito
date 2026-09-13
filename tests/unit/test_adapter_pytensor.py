import numpy as np
import pytest
import tensorflow as tf

from elicito.adapter._pytensor import _constant, translate, vectorize

pt = pytest.importorskip("pytensor.tensor")

SEED = tf.constant([1, 2], dtype=tf.int32)
M = np.arange(6.0).reshape(2, 3)


def test_integer_constant_stays_integer():
    assert _constant(pt.constant(np.array([0, 2]))).dtype == tf.int32
    assert _constant(pt.constant(np.array([0.5]))).dtype == tf.float32


def test_elementwise_op_upcasts_an_integer_input():
    x = pt.vector("x")
    [out] = translate([x + np.arange(3)], [x])([tf.ones(3)], SEED)
    assert out.dtype == tf.float32
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda m: m[1:, 0], id="Subtensor"),
        pytest.param(lambda m: m[::-1], id="Subtensor-step"),
        pytest.param(lambda m: pt.concatenate([m, m], axis=1), id="Join"),
        pytest.param(lambda m: m.shape, id="Shape"),
        pytest.param(lambda m: pt.fill(m, 2.0), id="Second"),
        pytest.param(lambda m: m - 1.0, id="Sub"),
        pytest.param(lambda m: -m, id="Neg"),
        pytest.param(lambda m: m / 2, id="TrueDiv"),
        pytest.param(lambda m: pt.reciprocal(m + 1), id="Reciprocal"),
        pytest.param(lambda m: m**2, id="Pow"),
        pytest.param(lambda m: pt.exp(m), id="Exp"),
        pytest.param(lambda m: pt.log(m + 1), id="Log"),
        pytest.param(lambda m: pt.sqrt(m), id="Sqrt"),
        pytest.param(lambda m: pt.abs(m - 2.5), id="Abs"),
        pytest.param(lambda m: pt.sign(m - 2.5), id="Sign"),
        pytest.param(lambda m: pt.erf(m / 4), id="Erf"),
        pytest.param(lambda m: pt.sigmoid(m - 2.5), id="Sigmoid"),
        pytest.param(lambda m: pt.minimum(m, 2.5), id="Minimum"),
        pytest.param(lambda m: pt.maximum(m, 2.5), id="Maximum"),
        pytest.param(lambda m: pt.clip(m, 1, 4), id="Clip"),
        pytest.param(lambda m: pt.switch(m > 2, m, -m), id="Switch"),
        pytest.param(lambda m: pt.eq(m, 2), id="EQ"),
        pytest.param(lambda m: pt.lt(m, 2), id="LT"),
        pytest.param(lambda m: pt.gt(m, 2), id="GT"),
        pytest.param(lambda m: pt.cast(m, "int64"), id="Cast"),
        pytest.param(lambda m: pt.sum(m, axis=1), id="Sum"),
        pytest.param(lambda m: pt.sum(m), id="Sum-all"),
        pytest.param(lambda m: pt.prod(m + 1, axis=0), id="Prod"),
        pytest.param(lambda m: pt.argsort(-m, axis=1), id="ArgSortOp"),
        pytest.param(lambda m: pt.reshape(m, (3, -1)), id="Reshape"),
        pytest.param(lambda m: pt.stack([m[0, 0], m[1, 2]]), id="MakeVector"),
        pytest.param(lambda m: pt.repeat(m[0], 2), id="Alloc-repeat"),
        pytest.param(lambda m: pt.tile(m, (1, 2)), id="Alloc-tile"),
    ],
)
def test_rule_matches_pytensor(build):
    m = pt.matrix("m")
    out = build(m)
    [value] = translate([out], [m])([tf.constant(M, tf.float32)], SEED)
    expected = out.eval({m: M})
    assert value.dtype.is_bool == (expected.dtype == bool)
    np.testing.assert_allclose(
        np.asarray(value, dtype=float), expected.astype(float), rtol=1e-6
    )


def test_rv_takes_its_size():
    x = pt.random.normal(0.0, 1.0, size=(2, 3))
    [out] = translate([x], [])([], SEED)
    assert out.shape == (2, 3)


def test_batched_rv_takes_the_new_axes():
    """the first parameter is constant, the second is batched"""
    scale = pt.vector("scale", shape=(3,))
    x = pt.random.normal(0.0, scale, size=(3,))
    scale_b = pt.matrix("scale_b", shape=(None, 3))
    [x_b] = vectorize([x], {scale: scale_b})
    [out] = translate([x_b], [scale_b])([tf.ones((5, 3))], SEED)
    assert out.shape == (5, 3)
