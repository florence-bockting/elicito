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


for boundary in ["lower-bound", "upper-bound", "double-bound"]:
    if boundary == "double-bound":

        @pytest.fixture
        def bound():
            return DoubleBound(lower=1.0, upper=5.0)

    if boundary == "lower-bound":

        @pytest.fixture
        def bound():
            return LowerBound(lower=1.0)

    if boundary == "upper-bound":

        @pytest.fixture
        def bound():
            return UpperBound(upper=5.0)

    @pytest.mark.parametrize("x", [3.5, 1.0, 5.0, [1.5, 2.5, 3.5, 4.5]])
    def test_forward_inverse_doublebound(bound, x):
        x = tf.constant(x, dtype=tf.float32)
        y = bound.forward(x)
        x_recovered = bound.inverse(y)

        np.testing.assert_allclose(x, x_recovered, rtol=1e-6)

    @pytest.mark.parametrize("x", [-3.0, 1.0, 3.0, [-2.0, 0.0, 2.0]])
    def test_inverse_forward_doublebound(bound, x):
        y = tf.constant(x, dtype=tf.float32)
        x = bound.inverse(y)
        y_recovered = bound.forward(x)

        np.testing.assert_allclose(y, y_recovered, rtol=1e-6)

    if boundary == "double-bound":

        @pytest.mark.parametrize("p", [0.4, 0.0, 1.0, [0.1, 0.5, 0.9]])
        def test_logit_doublebound(bound, p):
            u = tf.constant(p, dtype=tf.float32)
            v = bound.logit(u)
            u_recovered = bound.inv_logit(v)

            np.testing.assert_allclose(u, u_recovered, rtol=1e-6)


class TestModel:
    def __call__(self, prior_samples):
        b0 = tfd.Normal(0, 1).sample((3, 3, 1))
        return dict(prior_samples=prior_samples, b0=b0)


class DummyEliobj:
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


@pytest.fixture
def eliobj():
    return DummyEliobj()


def test_save_and_load_path(eliobj):
    file_path = "tests/test-data/dummy_eliobj.pkl"

    save(eliobj, file=file_path, overwrite=True)

    # Check that the file exists
    assert os.path.exists(file_path)

    loaded_eliobj = load(file_path)

    # Check that loaded object is correct
    assert loaded_eliobj.model["obj"] == TestModel
    assert loaded_eliobj.parameters[0]["name"] == "b0"
    assert loaded_eliobj.targets[0]["name"] == "b0"
    assert loaded_eliobj.trainer["method"] == "parametric_prior"
    assert loaded_eliobj.trainer["seed"] == 42


def test_save_and_load_name(eliobj):
    save(eliobj, name="dummy_eliobj", overwrite=True)

    expected_path = "results/parametric_prior/dummy_eliobj_42.pkl"
    assert os.path.exists(expected_path)

    loaded_eliobj = load(expected_path)

    # Check that loaded object is correct
    assert loaded_eliobj.model["obj"] == TestModel
    assert loaded_eliobj.parameters[0]["name"] == "b0"
    assert loaded_eliobj.targets[0]["name"] == "b0"
    assert loaded_eliobj.trainer["method"] == "parametric_prior"
    assert loaded_eliobj.trainer["seed"] == 42

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
