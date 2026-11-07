"""
Unittests for specs.py module
"""

import inspect

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el
from elicito.parameters.bijections import (
    DoubleBound,
    LowerBound,
    UpperBound,
    identity,
)
from elicito.specs import Dtype, VariableType, hyper, parameter

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


def test_initializer_defaults():
    """no argument gives a Nelder-Mead search on the default uniform box"""
    init = el.initializer()
    assert init["method"] == "warmstart"
    assert init["iterations"] == 100
    assert init["distribution"] == el.initializers.uniform()


def test_initializer():
    msg = "If method is None, 'distribution' must also be None."
    with pytest.raises(ValueError, match=msg):
        el.initializer(
            method=None, distribution=el.initializers.uniform(radius=1, mean=0)
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

    # the list of valid names comes from the initialization registry
    msg = (
        "Currently implemented initialization methods are "
        "'cmaes', 'lhs', 'random', 'sobol', 'warmstart', but got "
        "method='something' as input."
    )
    with pytest.raises(ValueError, match=msg):
        el.initializer(
            method="something",
            distribution=el.initializers.uniform(radius=1, mean=0),
            iterations=32,
        )


@pytest.mark.parametrize(
    ("lower", "upper"), [(0, "inf"), ("-inf", 5), (1, 5), ("-inf", "inf")]
)
def test_hyper_unconstrain_inverts_the_constraint(lower, upper):
    x = hyper("b0", lower, upper)
    value = tf.constant([1.5, 2.0, 3.0])

    assert np.allclose(x["unconstrain"](x["constraint"](value)), value, atol=1e-5)


def test_custom_function_reads_the_signature():
    def func(y):
        return y + 1.0

    def renamed(**kw):
        return func(kw["y__e0"])

    renamed.__signature__ = inspect.Signature(
        [inspect.Parameter("y__e0", inspect.Parameter.KEYWORD_ONLY)]
    )
    out = el.targets.use_custom_functions({"y__e0": 1.0, "z": 5.0}, renamed)
    assert out == 2.0


def test_correlation_reads_its_output_if_the_model_gives_it():
    target = el.correlation("cor")
    prior_samples = tf.zeros((1, 4, 2))
    own = tf.ones((1, 4, 2))

    out = el.targets.computation_target_quantities({}, prior_samples, [target])
    assert out["cor"] is prior_samples
    out = el.targets.computation_target_quantities(
        {"cor": own}, prior_samples, [target]
    )
    assert out["cor"] is own
