"""
Unittests for elicit.py module
"""

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el
from elicito.elicit import Dtype, VariableType, hyper, parameter
from elicito.utils import (
    DoubleBound,
    LowerBound,
    UpperBound,
    identity,
)

tfd = tfp.distributions


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "vtype, dim_exp",
    [
        (VariableType.real, 0),
        (VariableType.array, 1),
        (VariableType.cov, 2),
        (VariableType.cov2tril, 2),
    ],
)
@pytest.mark.parametrize("x", [0.1, 10.0, 100.0, 0.4])
def test_Dtype(x, vtype, dim, dim_exp):
    dtype = Dtype(vtype, dim)

    x_obs = dtype(x)

    assert isinstance(x_obs, tf.Tensor)
    assert x_obs.dtype == tf.float32
    assert x_obs.ndim == dim_exp


def test_hyper_checks():
    msg = "Lower must be either '-inf' or a float." "Other strings are not allowed."
    with pytest.raises(ValueError, match=msg):
        hyper("b0", lower="some_string", upper="inf")

    msg = "Upper must be either 'inf' or a float." "Other strings are not allowed."
    with pytest.raises(ValueError, match=msg):
        hyper("b0", lower="-inf", upper="some_string")

    msg = "The value for 'lower' must be smaller than the value for 'upper'."
    with pytest.raises(ValueError, match=msg):
        hyper("b0", lower=5, upper=1)

    msg = (
        "vtype must be either 'real', 'array', 'cov', 'cov2tril'. "  # type: ignore
        "You provided vtype='some_type'."
    )
    with pytest.raises(ValueError, match=msg):
        hyper("b0", vtype="some_type")

    msg = "For vtype='array', the 'dim' argument must have a value greater 1."  # type: ignore
    with pytest.raises(ValueError, match=msg):
        hyper("b0", vtype="array", dim=1)


@pytest.mark.parametrize(
    "lower, upper, expected_name, expected_type",
    [
        (0, "inf", "softplusL", LowerBound),
        ("-inf", 5, "softplusU", UpperBound),
        (1, 5, "invlogit", DoubleBound),
        ("-inf", "inf", "identity", identity),
    ],
)
def test_hyper_transformation(lower, upper, expected_name, expected_type):
    x = hyper("b0", lower, upper)

    if expected_name != "identity":
        assert isinstance(x["constraint"].__self__, expected_type)
    else:
        assert x["constraint"] == expected_type
    assert x["constraint_name"] == expected_name


def test_parameter_checks():
    msg = "'normal' family has no argument 'mu'. Check keys of " "'hyperparams' dict."
    with pytest.raises(ValueError, match=msg):
        parameter(
            "b0",
            family=tfd.Normal,
            hyperparams=dict(mu=el.hyper("mu"), scale=el.hyper("scale")),
        )


@pytest.mark.parametrize(
    "lower, upper, expected_name, expected_type",
    [
        (0, "inf", "softplusL", LowerBound),
        ("-inf", 5, "softplusU", UpperBound),
        (1, 5, "invlogit", DoubleBound),
        ("-inf", "inf", "identity", identity),
    ],
)
def test_parameter_transformation(lower, upper, expected_name, expected_type):
    x = parameter("b0", lower=float(lower), upper=float(upper))

    if expected_name != "identity":
        assert isinstance(x["constraint"].__self__, expected_type)
    else:
        assert x["constraint"] == expected_type
    assert x["constraint_name"] == expected_name


class TestModel1:
    """missing prior samples"""

    def __call__(self):
        return "Hello"


class TestModel2:
    """minimal model example"""

    def __call__(self, prior_samples, b):
        return b


def test_model():
    msg = (
        "The generative model class 'obj' requires the"
        " input variable 'prior_samples' but argument has not been found"
        " in 'obj'."
    )
    with pytest.raises(ValueError, match=msg):
        el.model(TestModel1)

    msg = (
        "The argument arg='b' required by the"
        "generative model class 'obj' is missing."
    )
    with pytest.raises(ValueError, match=msg):
        el.model(TestModel2)


def test_initializer():
    msg = "If method is None, 'distribution' must also be None."
    with pytest.raises(ValueError, match=msg):
        el.initializer(
            method=None, distribution=el.initialization.uniform(radius=1, mean=0)
        )

    msg = "If method is None, 'iterations' must also be None."
    with pytest.raises(ValueError, match=msg):
        el.initializer(method=None, iterations=12)

    msg = (  # type: ignore
        "Either 'method' or 'hyperparams' has"
        "to be specified. Use method for sampling from an"
        "initialization distribution and 'hyperparams' for"
        "specifying exact initial values per hyperparameter."
    )
    with pytest.raises(ValueError, match=msg):
        el.initializer(method=None, hyperparams=None)

    msg = "If 'distribution' is None, then 'method' must also be None."
    with pytest.raises(ValueError, match=msg):
        el.initializer(method="random", distribution=None, iterations=32)

    msg = (
        "Currently implemented initialization "
        "methods are 'random', 'sobol', and 'lhs', but got method='something'"
        " as input."
    )
    with pytest.raises(ValueError, match=msg):
        el.initializer(
            method="something",
            distribution=el.initialization.uniform(radius=1, mean=0),
            iterations=32,
        )


def test_trainer():
    msg = "Progress has to be either 0 or 1. Got progress=2."
    with pytest.raises(ValueError, match=msg):
        el.trainer(method="deep_prior", seed=1, epochs=3, progress=2)

    msg = "The number of epochs has to be greater 0." " Got epochs=0."
    with pytest.raises(ValueError, match=msg):
        el.trainer(method="deep_prior", seed=1, epochs=0)

    msg = (
        "Currently only the methods 'deep_prior' and"
        "'parametric prior' are implemented but got method='some_prior'."
    )
    with pytest.raises(ValueError, match=msg):
        el.trainer(method="some_prior", seed=1, epochs=3)
