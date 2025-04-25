import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el
from elicito import ExpertDict
from elicito.utils import (
    CorrelationMatrix,
    CovarianceMatrix,
)

tfd = tfp.distributions


@pytest.mark.parametrize("K", [2, 3, 4, 5])
def test_correlation_matrix(K):
    Cor = CorrelationMatrix(K)

    # from the unconstrained values y to a k x k correlation matrix X
    y = tf.random.normal(shape=(int((K * (K - 1)) / 2),))

    cor = Cor.inverse(y)

    np.testing.assert_array_equal(cor, np.transpose(cor))
    np.testing.assert_array_almost_equal(np.diagonal(cor), np.ones(K))
    assert all(
        tf.linalg.eigvalsh(cor) > 0.0
    ), "correlation matrix is not positive definite."

    np.testing.assert_array_equal(cor.shape, (K, K))
    np.testing.assert_array_almost_equal(np.diag(cor), np.ones(K))
    assert all(
        (Cor._get_upper_triangular(cor) >= -1.0)
        & (Cor._get_upper_triangular(cor) <= 1.0)
    ), "correlation matrix has values outside of [-1, 1]"

    # from a correlation matrix X to the unconstrained values y
    y_prime = Cor.forward(cor)

    np.testing.assert_array_almost_equal(y_prime, y, decimal=4)


@pytest.mark.parametrize(
    "S, M",
    [
        ([2.5, 1.3, 0.8], [[1.0, 0.3, -0.3], [0.3, 1.0, -0.2], [-0.3, -0.2, 1.0]]),
        ([1.0, 3.0, 1.0], [[1.0, 0.8, 0.2], [0.8, 1.0, 0.4], [0.2, 0.4, 1.0]]),
        ([0.5, 0.3, 0.8], [[1.0, -0.3, 0.0], [-0.3, 1.0, -0.6], [0.0, -0.6, 1.0]]),
    ],
)
def test_covariance_matrix(S, M):
    cov = (tf.linalg.diag(S) @ M) @ tf.linalg.diag(S)
    K = cov.shape[0]
    Cov = CovarianceMatrix(K)

    y = Cov.forward(cov)
    np.testing.assert_array_equal(y.shape, cov.shape)

    X = Cov.inverse(y)
    np.testing.assert_array_almost_equal(X, cov)


@pytest.mark.parametrize("K", [2, 3, 4, 5])
def test_cor2cov(K):
    vals = int(K * (K - 1) / 2)

    sd_unconstrained = tf.random.normal((K,))
    cor_unconstrained = tf.random.normal((vals,))

    cor = CorrelationMatrix(K)

    cor_constrained = cor.inverse(cor_unconstrained)
    sd_constrained = tf.exp(sd_unconstrained)

    cov_constrained = (
        tf.linalg.diag(sd_constrained) @ cor_constrained
    ) @ tf.linalg.diag(sd_constrained)
    assert all(
        tf.linalg.eigvalsh(cov_constrained) > 0.0
    ), "covariance matrix is not positive definite."


@pytest.fixture()
def param():
    return [
        el.parameter(
            name="betas",
            family=el.utils.MultivariateNormal,
            hyperparams=dict(
                loc=el.hyper("mu0s", vtype="array", dim=3),
                scale=el.hyper("sds", lower=0, vtype="array", dim=3),
                cor=el.hyper("cor", vtype="correlation", dim=3),
            ),
        ),
    ]


def test_initialize_multivariate_normal(param):
    init_values = dict(
        mu0s=tf.random.normal((3, 1)),
        sds=tf.random.normal((3, 1)),
        cor=tf.random.normal((3,)),
    )

    init_prior = el.simulations.intialize_priors(
        init_matrix_slice=init_values,
        method="parametric_prior",
        seed=1,
        parameters=param,
        network=None,
    )

    np.testing.assert_array_equal(len(init_prior), len(init_values.keys()))
    for key in init_prior.keys():
        assert init_prior[
            key
        ].trainable, f"initialized variable with {key=} is not trainable."


def test_sample_from_initialization(param):
    init_values = dict(
        mu0s=tf.random.normal((3,)),
        sds=tf.random.normal((3,)),
        cor=tf.random.normal((3,)),
    )

    init_prior = el.simulations.intialize_priors(
        init_matrix_slice=init_values,
        method="parametric_prior",
        seed=1,
        parameters=param,
        network=None,
    )

    samples = el.simulations.sample_from_priors(
        initialized_priors=init_prior,
        ground_truth=False,
        num_samples=1000,
        B=100,
        seed=1,
        method="parametric_prior",
        parameters=param,
        network=None,
        expert=ExpertDict,
    )

    np.testing.assert_equal(samples.shape[0], 100)
    np.testing.assert_equal(samples.shape[1], 1_000)
