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
    ],
)
def test_rule_matches_pytensor(build):
    m = pt.matrix("m")
    out = build(m)
    [value] = translate([out], [m])([tf.constant(M, tf.float32)], SEED)
    np.testing.assert_allclose(value, out.eval({m: M}))


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
