"""
Unittests for utils module
"""

import os
import pickle
import shutil

import cloudpickle
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr

import elicito as el
from elicito.utils import (
    DoubleBound,
    LowerBound,
    UpperBound,
    gumbel_softmax_trick,
    load,
    parallel,
    save,
    save_as_pkl,
)

tfd = tfp.distributions


def test_save_as_pkl():
    test_object = xr.DataTree(xr.Dataset({"a": 1, "b": [1, 2, 3]}))
    test_path = "tests/test-data/test_object.pkl"

    save_as_pkl(test_object, test_path)

    # check that file exists
    assert os.path.exists(test_path)

    # load file
    with open(test_path, "rb") as f:
        serialized_obj = pickle.load(f)  # noqa: S301
        loaded_obj = cloudpickle.loads(serialized_obj)

    assert loaded_obj == test_object


@pytest.fixture
def doublebound():
    return DoubleBound(lower=1.0, upper=5.0)


@pytest.mark.parametrize("x", [3.5, 1.0, 5.0, [1.5, 2.5, 3.5, 4.5]])
def test_forward_inverse_doublebound(doublebound, x):
    x = tf.constant(x, dtype=tf.float32)
    y = doublebound.forward(x)
    x_recovered = doublebound.inverse(y)

    np.testing.assert_allclose(x, x_recovered, rtol=1e-6)


@pytest.mark.parametrize("x", [-3.0, 1.0, 3.0, [-2.0, 0.0, 2.0]])
def test_inverse_forward_doublebound(doublebound, x):
    y = tf.constant(x, dtype=tf.float32)
    x = doublebound.inverse(y)
    y_recovered = doublebound.forward(x)

    np.testing.assert_allclose(y, y_recovered, rtol=1e-6)


@pytest.mark.parametrize("p", [0.4, 0.0, 1.0, [0.1, 0.5, 0.9]])
def test_logit_doublebound(doublebound, p):
    u = tf.constant(p, dtype=tf.float32)
    v = doublebound.logit(u)
    u_recovered = doublebound.inv_logit(v)

    np.testing.assert_allclose(u, u_recovered, rtol=1e-6)


@pytest.fixture
def lowerbound():
    return LowerBound(lower=1.0)


@pytest.mark.parametrize("x", [3.5, 1.0, 5.0, [1.5, 2.5, 3.5, 4.5]])
def test_forward_inverse_lowerbound(lowerbound, x):
    x = tf.constant(x, dtype=tf.float32)
    y = lowerbound.forward(x)
    x_recovered = lowerbound.inverse(y)

    np.testing.assert_allclose(x, x_recovered, rtol=1e-6)


@pytest.mark.parametrize("x", [-3.0, 1.0, 3.0, [-2.0, 0.0, 2.0]])
def test_inverse_forward_lowerbound(lowerbound, x):
    y = tf.constant(x, dtype=tf.float32)
    x = lowerbound.inverse(y)
    y_recovered = lowerbound.forward(x)

    np.testing.assert_allclose(y, y_recovered, rtol=1e-6)


class TestModel:
    def __call__(self, prior_samples):
        b0 = tfd.Normal(0, 1).sample((3, 3, 1))
        return dict(prior_samples=prior_samples, b0=b0)


@pytest.fixture
def upperbound():
    return UpperBound(upper=5.0)


@pytest.mark.parametrize("x", [3.5, 1.0, 5.0, [1.5, 2.5, 3.5, 4.5]])
def test_forward_inverse_upperbound(upperbound, x):
    x = tf.constant(x, dtype=tf.float32)
    y = upperbound.forward(x)
    x_recovered = upperbound.inverse(y)

    np.testing.assert_allclose(x, x_recovered, rtol=1e-6)


@pytest.mark.parametrize("x", [-3.0, 1.0, 3.0, [-2.0, 0.0, 2.0]])
def test_inverse_forward_upperbound(upperbound, x):
    y = tf.constant(x, dtype=tf.float32)
    x = upperbound.inverse(y)
    y_recovered = upperbound.forward(x)

    np.testing.assert_allclose(y, y_recovered, rtol=1e-6)


class DummyEliobj_empty:
    def __init__(self):
        self.model = el.model(TestModel)
        self.parameters = [
            el.parameter(
                name="b0",
                family=tfd.Normal,
                hyperparams=dict(loc=el.hyper(name="mu0"), scale=el.hyper("sigma0")),
            )
        ]
        self.targets = [
            el.target(name="b0", loss=el.losses.L2, query=el.queries.quantiles((0.5,)))
        ]
        self.expert = el.expert.data({"quantiles_b0": 0.5})
        self.optimizer = el.optimizer(
            optimizer=tf.keras.optimizers.Adam, learning_rate=0.1
        )
        self.trainer = el.trainer(method="parametric_prior", seed=42, epochs=1)
        self.initializer = el.initializer(
            "sobol",
            distribution=el.initialization.uniform(radius=1, mean=0),
            iterations=1,
        )
        self.network = None
        self.results = None
        self.history = None


class DummyEliobj_fitted:
    def __init__(self):
        self.model = el.model(TestModel)
        self.parameters = [
            el.parameter(
                name="b0",
                family=tfd.Normal,
                hyperparams=dict(loc=el.hyper(name="mu0"), scale=el.hyper("sigma0")),
            )
        ]
        self.targets = [
            el.target(name="b0", loss=el.losses.L2, query=el.queries.quantiles((0.5,)))
        ]
        self.expert = el.expert.data({"quantiles_b0": 0.5})
        self.optimizer = el.optimizer(
            optimizer=tf.keras.optimizers.Adam, learning_rate=0.1
        )
        self.trainer = el.trainer(method="parametric_prior", seed=42, epochs=1)
        self.initializer = el.initializer(
            "sobol",
            distribution=el.initialization.uniform(radius=1, mean=0),
            iterations=1,
        )
        self.network = None
        self.results = [1, 2, 3, 4]
        self.history = [1, 2, 3, 4]


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("eliobj", [DummyEliobj_empty(), DummyEliobj_fitted()])
@pytest.mark.parametrize(
    "test_path", ["tests/test-data/dummy_eliobj", "tests/test-data/dummy_eliobj.pkl"]
)
def test_save_and_load_path(monkeypatch, eliobj, test_path, overwrite):
    pytest.importorskip("scipy")

    # Ensure the directory exists
    os.makedirs("tests/test-data", exist_ok=True)

    expected_file = "tests/test-data/dummy_eliobj.pkl"

    # Check that assert statement works
    with pytest.raises(AssertionError):
        save(eliobj, name=None, file=None)

    if not overwrite:
        # Mock os.path.isfile to return True for our path
        monkeypatch.setattr("os.path.isfile", lambda p: True)

        # Mock input() to simulate user typing 'n'
        monkeypatch.setattr("builtins.input", lambda _: "n")

        called = {"used": False}
        monkeypatch.setattr(
            "elicito.utils.save", lambda eliobj, file: called.update("used", True)
        )

    # Check saving object works
    save(eliobj, file=test_path, overwrite=overwrite)

    if not overwrite:
        assert not called["used"]
    else:
        assert os.path.isfile(expected_file)

        # Check that loading object works
        loaded_eliobj = load(expected_file)

        assert loaded_eliobj.model["obj"] == TestModel
        assert loaded_eliobj.parameters[0]["name"] == "b0"
        assert loaded_eliobj.targets[0]["name"] == "b0"
        assert loaded_eliobj.trainer["method"] == "parametric_prior"
        assert loaded_eliobj.trainer["seed"] == 42
        assert loaded_eliobj.results == eliobj.results
        assert loaded_eliobj.history == eliobj.history

        # clean-up directory
        shutil.rmtree("tests/test-data")


@pytest.mark.parametrize("eliobj", [DummyEliobj_empty(), DummyEliobj_fitted()])
@pytest.mark.parametrize("test_file", ["dummy_eliobj.pkl", "dummy_eliobj"])
def test_save_and_load_name(eliobj, test_file):
    pytest.importorskip("scipy")

    save(eliobj, name=test_file, overwrite=True)

    expected_path = "results/parametric_prior/dummy_eliobj_42.pkl"

    assert os.path.exists(expected_path)

    loaded_eliobj = load(expected_path)

    # Check that loaded object is correct
    assert loaded_eliobj.model["obj"] == TestModel
    assert loaded_eliobj.parameters[0]["name"] == "b0"
    assert loaded_eliobj.targets[0]["name"] == "b0"
    assert loaded_eliobj.trainer["method"] == "parametric_prior"
    assert loaded_eliobj.trainer["seed"] == 42
    assert loaded_eliobj.results == eliobj.results
    assert loaded_eliobj.history == eliobj.history

    # clean-up directory
    shutil.rmtree("results/parametric_prior")


@pytest.mark.parametrize(
    "runs, cores",
    [
        (1, None),
        (2, 4),
        (10, 10),
    ],
)
def test_parallel(runs, cores):
    result = parallel(runs=runs, cores=cores)
    expected_cores = runs if cores is None else cores

    assert result == {"runs": runs, "cores": expected_cores, "seeds": None}


def test_gumble_softmax_trick_assert():
    likelihood = tfd.Poisson(rate=tf.ones((3, 2, 4)))

    # check assert statement
    with pytest.raises(ValueError, match="batch_shape"):
        gumbel_softmax_trick(likelihood, upper_thres=10)


def test_gumble_softmax_trick():
    B, S, N = 2, 3, 4

    valid_likelihood = tfd.Poisson(rate=tf.ones((B, S, N, 1)) * 2.0)

    upper_thres = 10
    temp = 1.6

    ypred = gumbel_softmax_trick(valid_likelihood, upper_thres=upper_thres, temp=temp)

    # Expected shape: (B, S, N)
    assert ypred.shape == (2, 3, 4)

    # Must be finite
    assert tf.math.reduce_all(tf.math.is_finite(ypred))

    # Values should be in [0, upper_thres]
    assert np.all(ypred.numpy() >= 0)
    assert np.all(ypred.numpy() <= upper_thres)

    # dtype should be float32
    assert ypred.dtype == tf.float32
