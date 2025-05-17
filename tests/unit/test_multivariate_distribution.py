import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

dim = 3


@pytest.fixture(scope="module")
def param():
    return [
        el.parameter(
            name="multivariate_normal",
            family=tfd.MultivariateNormalTriL,
            hyperparams=dict(
                loc=el.hyper(name="mus", vtype="array", dim=dim),
                scale_tril=el.hyper(name="cov_matrix", vtype="cov2tril", dim=dim),
            ),
        )
    ]


def test_specification_multivariate_normal(param):
    param = param[0]
    loc_init = 1.0
    cov_init = [[0.36, 0.12, 0.06], [0.12, 0.29, -0.13], [0.06, -0.13, 0.26]]

    distr = param["family"]
    hyp_loc = param["hyperparams"]["loc"]["vtype"](loc_init)
    hyp_scale = param["hyperparams"]["scale_tril"]["vtype"](cov_init)

    np.testing.assert_equal(len(hyp_loc), dim)
    np.testing.assert_equal(hyp_scale.shape, (dim, dim))
    np.testing.assert_array_equal(hyp_scale, tf.linalg.cholesky(cov_init))
    np.testing.assert_equal(type(distr), type(tfd.MultivariateNormalTriL))


def test_initialization_multivariate_normal(param):
    init_matrix_slice = dict(
        mus=[0.5, 0.3, 0.1],
        cov_matrix=tf.linalg.cholesky(
            [[0.36, 0.12, 0.06], [0.12, 0.29, -0.13], [0.06, -0.13, 0.26]]
        ),
    )
    init_prior = el.simulations.intialize_priors(
        init_matrix_slice,
        method="parametric_prior",
        seed=1,
        parameters=param,
        network=None,
    )

    np.testing.assert_array_almost_equal(
        init_prior["loc_mus"], init_matrix_slice["mus"]
    )
    np.testing.assert_array_almost_equal(
        init_prior["scale_tril_cov_matrix"], init_matrix_slice["cov_matrix"]
    )


def test_sample_multivariate_normal(param):
    S = [2.5, 1.3, 0.8]
    M = [[1.0, 0.95, -0.99], [0.95, 1.0, -0.95], [-0.99, -0.95, 1.0]]
    cor = [0.95, -0.99, -0.95]
    covariance_matrix = (tf.linalg.diag(S) @ M) @ tf.linalg.diag(S)

    init_matrix_slice = dict(
        mus=[0.5, 0.3, 0.1], cov_matrix=tf.linalg.cholesky(covariance_matrix)
    )
    init_prior = el.simulations.intialize_priors(
        init_matrix_slice,
        method="parametric_prior",
        seed=1,
        parameters=param,
        network=None,
    )
    samples = el.simulations.sample_from_priors(
        initialized_priors=init_prior,
        ground_truth=False,
        num_samples=10_000,
        B=100,
        seed=1,
        method="parametric_prior",
        parameters=param,
        network=None,
        expert=None,
    )

    np.testing.assert_array_equal(samples.shape, (100, 10_000, 3))

    np.testing.assert_allclose(
        tf.reduce_mean(samples, (0, 1)), init_matrix_slice["mus"], rtol=1e-2
    )

    np.testing.assert_allclose(
        tf.reduce_mean(tf.math.reduce_std(samples, 1), 0), S, rtol=1e-2
    )
    np.testing.assert_allclose(
        tf.reduce_mean(el.targets.pearson_correlation(samples), 0), cor, rtol=1e-2
    )


def test_initialization_covariance_matrix(param):
    init_matrix = el.initializer(
        method=None,
        distribution=None,
        iterations=None,
        hyperparams=dict(mus=[0.5, 0.3, 0.1], cov_matrix=tf.eye(3)),
    )

    np.testing.assert_array_equal(init_matrix["hyperparams"]["mus"], [0.5, 0.3, 0.1])
    np.testing.assert_array_equal(init_matrix["hyperparams"]["cov_matrix"], tf.eye(3))


def test_trainable_variables_multivariate_normal(param):
    init_matrix = el.initializer(
        method=None,
        distribution=None,
        iterations=None,
        hyperparams=dict(mus=[0.5, 0.3, 0.1], cov_matrix=tf.eye(3)),
    )

    init_priors = el.simulations.Priors(
        ground_truth=False,
        init_matrix_slice=init_matrix["hyperparams"],
        trainer=dict(method="parametric_prior"),
        parameters=param,
        network=None,
        expert=None,
        seed=1,
    )

    np.testing.assert_equal(len(init_priors.trainable_variables), 2)
    np.testing.assert_array_almost_equal(
        init_priors.trainable_variables[0], init_matrix["hyperparams"]["mus"]
    )
    np.testing.assert_array_almost_equal(
        init_priors.trainable_variables[1], init_matrix["hyperparams"]["cov_matrix"]
    )
