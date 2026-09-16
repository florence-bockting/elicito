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
        pytest.param(lambda m: m[np.array([1, 0, 1])], id="AdvancedSubtensor"),
        pytest.param(lambda m: m[:, np.array([2, 0])], id="AdvancedSubtensor-slice"),
        pytest.param(
            lambda m: m[np.array([0, 1]), np.array([2, 0])], id="AdvancedSubtensor-two"
        ),
        pytest.param(
            lambda m: m[np.array([[0, 1], [1, 0]])], id="AdvancedSubtensor-2d"
        ),
        pytest.param(lambda m: m[:, np.array([-1])], id="AdvancedSubtensor-negative"),
        pytest.param(lambda m: -1 * (m > 2), id="Mul-bool"),
        pytest.param(lambda m: pt.sum(m > 2, axis=1), id="Sum-bool"),
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


COV = np.array([[1.0, 0.6], [0.6, 2.0]])


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda size: pt.random.dirichlet(np.array([1.0, 2.0, 3.0]), size=size),
            id="DirichletRV",
        ),
        pytest.param(
            lambda size: pt.random.multinomial(
                10, np.array([0.2, 0.3, 0.5]), size=size
            ),
            id="MultinomialRV",
        ),
        pytest.param(
            lambda size: pt.random.multivariate_normal(
                np.array([1.0, -1.0]), pt.linalg.inv(np.linalg.inv(COV)), size=size
            ),
            id="MvNormalRV-MatrixInverse",
        ),
    ],
)
def test_multivariate_rv_matches_pytensor(build):
    """mean and covariance of 20000 draws, with the event axis last"""
    out = build((20000,))
    [value] = translate([out], [])([], SEED)
    expected = out.eval()
    assert value.shape == expected.shape
    value = np.asarray(value, dtype=float)
    np.testing.assert_allclose(value.mean(0), expected.mean(0), atol=0.06)
    np.testing.assert_allclose(np.cov(value.T), np.cov(expected.T), atol=0.1)


def test_mvnormal_rv_takes_a_batched_mean():
    mu = pt.matrix("mu", shape=(None, 2))
    out = pt.random.multivariate_normal(mu, COV)
    loc = np.array([[0.0, 0.0], [10.0, -10.0], [5.0, 5.0]], dtype=np.float32)
    [value] = translate([out], [mu])([tf.constant(loc)], SEED)
    assert value.shape == (3, 2)
    np.testing.assert_allclose(value, loc, atol=7)


def test_batched_rv_takes_the_new_axes():
    """the first parameter is constant, the second is batched"""
    scale = pt.vector("scale", shape=(3,))
    x = pt.random.normal(0.0, scale, size=(3,))
    scale_b = pt.matrix("scale_b", shape=(None, 3))
    [x_b] = vectorize([x], {scale: scale_b})
    [out] = translate([x_b], [scale_b])([tf.ones((5, 3))], SEED)
    assert out.shape == (5, 3)


def test_blockwise_advanced_subtensor():
    """each batch element of v takes its own index"""
    v, idx = pt.vector("v"), pt.lvector("idx")
    v_b, idx_b = pt.matrix("v_b"), pt.lmatrix("idx_b")
    [out] = vectorize([v[idx]], {v: v_b, idx: idx_b})
    x, i = np.arange(6.0).reshape(2, 3), np.array([[2, 0], [1, 1]])
    args = [tf.constant(x, tf.float32), tf.constant(i, tf.int32)]
    [value] = translate([out], [v_b, idx_b])(args, SEED)
    np.testing.assert_allclose(value, np.take_along_axis(x, i, axis=1))


def test_blockwise_advanced_subtensor_two_arrays_and_a_slice():
    """a constant 3-d array; the index on axis 0 is batched"""
    t = np.arange(24.0).reshape(2, 3, 4)
    i, j = pt.lvector("i", shape=(3,)), np.array([2, 2, 0])
    i_b = pt.lmatrix("i_b", shape=(None, 3))
    [out] = vectorize([pt.as_tensor(t)[i, j, :]], {i: i_b})
    assert type(out.owner.op).__name__ == "Blockwise"
    rows = np.array([[1, 0, 1], [0, 0, 1]])
    [value] = translate([out], [i_b])([tf.constant(rows, tf.int32)], SEED)
    np.testing.assert_allclose(value, [t[r, j, :] for r in rows])


def test_blockwise_repeat_and_reshape():
    v = pt.vector("v", shape=(2,))
    v_b = pt.matrix("v_b", shape=(None, 2))
    [out] = vectorize([pt.reshape(pt.repeat(v, 3), (2, 3))], {v: v_b})
    x = np.arange(4.0).reshape(2, 2)
    [value] = translate([out], [v_b])([tf.constant(x, tf.float32)], SEED)
    np.testing.assert_allclose(value, np.repeat(x, 3, axis=1).reshape(2, 2, 3))


def test_blockwise_argsort_in_a_tf_function():
    """the axis of a batched ArgSortOp stays static inside tf.function"""
    v = pt.vector("v", shape=(3,))
    v_b = pt.matrix("v_b", shape=(None, 3))
    [out] = vectorize([pt.argsort(v, axis=0)], {v: v_b})
    run = tf.function(translate([out], [v_b]))
    x = np.array([[3.0, 1.0, 2.0], [0.0, 2.0, 1.0]])
    [value] = run([tf.constant(x, tf.float32)], SEED)
    np.testing.assert_array_equal(value, np.argsort(x, axis=1))


def test_constant_subgraph_needs_no_rule():
    """Cos has no rule, but its input is a constant"""
    x = pt.vector("x")
    [out] = translate([x + pt.cos(pt.constant(0.0))], [x])([tf.ones(2)], SEED)
    np.testing.assert_allclose(out, [2.0, 2.0])


@pytest.mark.parametrize(("size", "shape"), [((3,), (4, 5, 3)), (None, (4, 5))])
def test_vectorize_gives_an_unbatched_rv_the_batch_axes(size, shape):
    """an RV without a batched input draws once per batch element"""
    x = pt.random.normal(0.0, 1.0, size=size)
    batch = pt.lvector("batch", shape=(2,))
    [x_b] = vectorize([x], {}, batch)
    [out] = translate([x_b], [batch])([tf.constant([4, 5])], SEED)
    assert out.shape == shape
    assert float(tf.math.reduce_std(out)) > 0.5
