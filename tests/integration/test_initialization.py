"""
test for initialization of hyperparameter values
"""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

# %% test uniform_samples
# check correct naming and output format of init_matrix computed from
# uniform_samples


@pytest.fixture
def parameters():
    return [
        el.parameter(
            name="beta0",
            family=tfd.Normal,
            hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)),
        ),
        el.parameter(
            name="beta1",
            family=tfd.Normal,
            hyperparams=dict(loc=el.hyper("mu1"), scale=el.hyper("sigma1", lower=0)),
        ),
        el.parameter(
            name="sigma",
            family=tfd.HalfNormal,
            hyperparams=dict(scale=el.hyper("sigma2", lower=0)),
        ),
    ]


def test_uniform_samples(parameters):
    pytest.importorskip("scipy")

    seed = 1
    hyppar = None
    n_samples = 10
    method = "sobol"
    mean = 0.0
    radius = 0.001

    init_matrix = el.initialization.uniform_samples(
        seed, hyppar, n_samples, method, mean, radius, parameters
    )

    # check names and order of hyperparameters in init_matrix
    assert list(init_matrix.keys()) == ["mu0", "sigma0", "mu1", "sigma1", "sigma2"]
    # check correct format
    assert init_matrix["mu0"].shape == [n_samples]

    # check that samples are drawn from a correctly specified
    # initialization distribution
    for key in init_matrix:
        assert tf.reduce_mean(init_matrix[key]).numpy() == pytest.approx(0.0, abs=0.01)


# %% test_uniform_samples_array
# constants
def test_uniform_samples_array(parameters):
    pytest.importorskip("scipy")

    seed = 1
    hyppar = ["mu0", "sigma0", "mu1", "sigma1", "sigma2"]
    n_samples = 10
    method = "sobol"
    mean = [0.0, 1.0, 2.0, 3.0, 4.0]
    radius = [0.001] * 5

    init_matrix = el.initialization.uniform_samples(
        seed, hyppar, n_samples, method, mean, radius, parameters
    )

    # check names and order of hyperparameters in init_matrix
    assert list(init_matrix.keys()) == hyppar
    # check correct format
    assert init_matrix["mu0"].shape == [n_samples]

    # check that samples are drawn from a correctly specified
    # initialization distribution
    for key, true in zip(init_matrix, mean):
        assert tf.reduce_mean(init_matrix[key]).numpy() == pytest.approx(
            true, abs=0.001
        )


# %% test_uniform_samples_array random order
# vary order of hyperparameter and use n_samples of 1
def test_uniform_samples_order(parameters):
    pytest.importorskip("scipy")

    seed = 1
    hyppar = ["mu0", "mu1", "sigma0", "sigma1", "sigma2"]
    n_samples = 1
    method = "sobol"
    mean = [0.0, 1.0, 2.0, 3.0, 4.0]
    radius = [0.001] * 5

    init_matrix = el.initialization.uniform_samples(
        seed, hyppar, n_samples, method, mean, radius, parameters
    )

    # check names and order of hyperparameters in init_matrix
    assert list(init_matrix.keys()) == hyppar
    # check correct format
    assert init_matrix["mu0"].shape == [n_samples]

    # check that samples are drawn from a correctly specified
    # initialization distribution
    for key, true in zip(init_matrix, mean):
        assert tf.reduce_mean(init_matrix[key]).numpy() == pytest.approx(
            true, abs=0.001
        )


# %% integration test


def test_integration_initialization():
    pytest.importorskip("scipy")

    # numeric, standardized predictor
    def std_predictor(N, quantiles):
        X = tf.cast(np.arange(N), tf.float32)
        X_std = (X - tf.reduce_mean(X)) / tf.math.reduce_std(X)
        X_sel = tfp.stats.percentile(X_std, [int(p * 100) for p in quantiles])
        return X_sel

    class ToyModel:
        def __call__(self, prior_samples, design_matrix):
            B = prior_samples.shape[0]
            S = prior_samples.shape[1]

            # preprocess shape of design matrix
            X = tf.broadcast_to(
                design_matrix[None, None, :], (B, S, len(design_matrix))
            )
            # linear predictor (= mu)
            epred = tf.add(
                prior_samples[:, :, 0][:, :, None],
                tf.multiply(prior_samples[:, :, 1][:, :, None], X),
            )
            # data-generating model
            likelihood = tfd.Normal(epred, scale=tf.ones(epred.shape))

            # prior predictive distribution
            ypred = likelihood.sample()

            # selected observations
            y_X0, y_X1 = (ypred[:, :, 0], ypred[:, :, 1])

            return dict(y_X0=y_X0, y_X1=y_X1)

    ground_truth = {
        "beta0": tfd.Normal(loc=0.1, scale=0.4),
        "beta1": tfd.Normal(loc=0.2, scale=0.2),
    }

    eliobj = el.Elicit(
        model=el.model(
            obj=ToyModel, design_matrix=std_predictor(N=200, quantiles=[0.25, 0.75])
        ),
        parameters=[
            el.parameter(
                name="beta0",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)
                ),
            ),
            el.parameter(
                name="beta1",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper("mu1"), scale=el.hyper("sigma1", lower=0)
                ),
            ),
        ],
        targets=[
            el.target(
                name="y_X0",
                query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
                loss=el.losses.MMD2(kernel="energy"),
                weight=1.0,
            ),
            el.target(
                name="y_X1",
                query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
                loss=el.losses.MMD2(kernel="energy"),
                weight=1.0,
            ),
        ],
        expert=el.expert.simulator(ground_truth=ground_truth, num_samples=10_000),
        optimizer=el.optimizer(
            optimizer=tf.keras.optimizers.Adam, learning_rate=0.001, clipnorm=1.0
        ),
        trainer=el.trainer(method="parametric_prior", seed=0, epochs=2),
        initializer=el.initializer(
            method="sobol",
            loss_quantile=0,
            iterations=1,
            distribution=el.initialization.uniform(
                radius=[0.01] * 4,
                mean=[0.0, 1.0, 2.0, 3.0],
                hyper=["mu0", "mu1", "sigma0", "sigma1"],
            ),
        ),
    )

    eliobj.fit()

    # check that initial values drawn from the initialization distribution
    # match with the specification of the initialization distr.
    for k, t in zip(eliobj.results[0]["init_matrix"], [0.0, 1.0, 2.0, 3.0]):
        assert tf.reduce_mean(
            eliobj.results[0]["init_matrix"][k]
        ).numpy() == pytest.approx(t, abs=0.01)

    # check whether initialized hyperparameter correspond to drawn initial
    # values
    hist = eliobj.history[0]
    res = eliobj.results[0]
    assert hist["hyperparameter"]["mu0"][0] == res["init_matrix"]["mu0"]
    assert hist["hyperparameter"]["mu1"][0] == res["init_matrix"]["mu1"]
    assert hist["hyperparameter"]["sigma0"][0] == pytest.approx(
        res["init_matrix"]["sigma0"], abs=0.5
    )
    assert hist["hyperparameter"]["sigma1"][0] == pytest.approx(
        res["init_matrix"]["sigma1"], abs=0.5
    )

    # check whether prior samples reflect corresponding initial hyperparameter
    means = tf.reduce_mean(res["prior_samples"], (0, 1))
    stds = tf.reduce_mean(tf.math.reduce_std(res["prior_samples"], 1), 0)

    for m, t in zip(means, [0.0, 1.0]):
        assert abs(m.numpy()) == pytest.approx(t, abs=0.03)

    for s, t in zip(stds, [2.0, 3.0]):
        assert s.numpy() == pytest.approx(t, abs=0.13)
