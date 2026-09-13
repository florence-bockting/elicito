import numpy as np
import pytest
import tensorflow as tf

from elicito.adapter._pytensor import _constant, translate

pt = pytest.importorskip("pytensor.tensor")

SEED = tf.constant([1, 2], dtype=tf.int32)


def test_integer_constant_stays_integer():
    assert _constant(pt.constant(np.array([0, 2]))).dtype == tf.int32
    assert _constant(pt.constant(np.array([0.5]))).dtype == tf.float32


def test_elementwise_op_upcasts_an_integer_input():
    x = pt.vector("x")
    [out] = translate([x + np.arange(3)], [x])([tf.ones(3)], SEED)
    assert out.dtype == tf.float32
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])
