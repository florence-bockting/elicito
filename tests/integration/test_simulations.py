"""
tests for model and prior simulations
"""

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el
from elicito.exceptions import MissingOptionalDependencyError

tfd = tfp.distributions


# %% test intialize_priors
# Fixtures for reusable test data
@pytest.fixture
def init_matrix_slice():
    """Fixture providing initial matrix slice values."""
    return dict(
        mu0=tf.constant(1.0),
        sigma0=0.5,
        mu1=tf.constant(1.0),
        mu2=tf.constant(2.0),
        sigma2=1.3,
    )


@pytest.fixture
def parameters():
    """Fixture providing a list of parameter definitions."""
    return [
        el.parameter(
            name="beta0",
            family=tfd.Normal,
            hyperparams=dict(
                loc=el.hyper("mu0"),
                scale=el.hyper("sigma0", lower=0, shared=True),
            ),
        ),
        el.parameter(
            name="beta1",
            family=tfd.Normal,
            hyperparams=dict(
                loc=el.hyper("mu1"),
                scale=el.hyper("sigma0", lower=0, shared=True),
            ),
        ),
        el.parameter(
            name="beta2",
            family=tfd.Gamma,
            hyperparams=dict(
                concentration=el.hyper("mu2"),
                rate=el.hyper("sigma2", lower=0),
            ),
        ),
    ]


@pytest.fixture
def expected_keys():
    """Expected keys for the initialized prior dictionary."""
    return ["loc_mu0", "scale_sigma0", "loc_mu1", "concentration_mu2", "rate_sigma2"]


@pytest.fixture
def expected_names(init_matrix_slice):
    """Expected names of initialized hyperparameters."""
    return [
        "identity.mu0",
        "softplusL.sigma0",
        "identity.mu1",
        "identity.mu2",
        "softplusL.sigma2",
    ]


@pytest.fixture
def expected_values():
    """Expected values of initialized hyperparameters."""
    return [1.0, 0.5, 1.0, 2.0, 1.3]


def test_initialize_priors_1(
    init_matrix_slice, parameters, expected_keys, expected_names, expected_values
):
    """Test the initialization of priors."""
    # Create a dictionary with initialized tf.Variables
    init_prior = el.simulations.intialize_priors(
        init_matrix_slice=init_matrix_slice,
        method="parametric_prior",
        seed=0,
        parameters=parameters,
        network=None,
    )

    # Re-run the function with the same seed
    init_prior_copy = el.simulations.intialize_priors(
        init_matrix_slice=init_matrix_slice,
        method="parametric_prior",
        seed=0,
        parameters=parameters,
        network=None,
    )

    # Check keys of initialized hyperparameter dictionary
    assert (
        list(init_prior.keys()) == expected_keys
    ), f"Expected keys: {expected_keys}, but got: {list(init_prior.keys())}"

    # Check names of initialized hyperparameters
    for i, key in enumerate(init_prior):
        assert init_prior[key].name[:-2] == expected_names[i], (
            f"Expected name for key '{key}': {expected_names[i]}, "
            f"but got: {init_prior[key].name[:-2]}"
        )

    # Check initial values of hyperparameters and their correct transformations
    for i, key in enumerate(init_prior):
        assert init_prior[key].numpy() == pytest.approx(expected_values[i], abs=0.01), (
            f"Expected value for key '{key}': {expected_values[i]}, "
            f"but got: {init_prior[key].numpy()}"
        )

    # Check that results are identical when using the same seed
    for key in init_prior:
        assert tf.reduce_all(init_prior[key] == init_prior_copy[key]), (
            f"Values for key '{key}' are not identical across repeated runs"
            + "with the same seed."
        )


# test deep_prior method
@pytest.fixture
def network():
    """Fixture providing definition of NF network."""
    pytest.importorskip("bayesflow")

    try:
        from bayesflow.inference_networks import InvertibleNetwork
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "bayesflow", requirement="bayesflow"
        ) from exc

    return el.networks.NF(
        inference_network=InvertibleNetwork,
        network_specs=dict(
            num_params=3,
            num_coupling_layers=3,
            coupling_design="affine",
            coupling_settings={
                "dropout": False,
                "dense_args": {
                    "units": 128,
                    "activation": "relu",
                    "kernel_regularizer": None,
                },
                "num_dense": 2,
            },
            permutation="fixed",
        ),
        base_distribution=el.networks.base_normal,
    )


def test_initialize_priors_2(network):
    pytest.importorskip("bayesflow")

    """Test the initialization of priors."""
    # Create a dictionary with initialized tf.Variables
    init_prior = el.simulations.intialize_priors(
        init_matrix_slice=None,
        method="deep_prior",
        seed=0,
        parameters=None,
        network=network,
    )

    # check that a bf.inference_networks,InvertibleNetwork has been constructed
    assert init_prior.name == "invertible_network"


# %% test sample_from_priors
@pytest.fixture
def expert():
    return dict(
        ground_truth=dict(
            beta0=tfd.Normal(-0.5, 0.8),
            beta1=tfd.Normal(0.0, 0.8),
            beta2=tfd.Gamma(2.0, 2.0),
        ),  # mean=1., sd =0.71
        num_samples=100_000,
    )


@pytest.fixture
def parameters_deep():
    """Fixture providing a list of parameter definitions for deep-prior."""
    return [
        el.parameter(name="beta0"),
        el.parameter(name="beta1"),
        el.parameter(name="beta2"),
    ]


# check: parametric_prior, oracle
def test_prior_samples_1(init_matrix_slice, parameters, expert):
    initialized_priors = el.simulations.intialize_priors(
        init_matrix_slice=init_matrix_slice,
        method="parametric_prior",
        seed=0,
        parameters=parameters,
        network=None,
    )

    prior_samples = el.simulations.sample_from_priors(
        initialized_priors, True, 10, 5, 0, "parametric_prior", parameters, None, expert
    )

    prior_samples_copy = el.simulations.sample_from_priors(
        initialized_priors, True, 10, 5, 0, "parametric_prior", parameters, None, expert
    )

    prior_samples_copy2 = el.simulations.sample_from_priors(
        initialized_priors, True, 10, 5, 1, "parametric_prior", parameters, None, expert
    )

    # check expected shape of prior samples (1, num_samples, num_params)
    assert prior_samples.shape == (1, 100_000, 3)
    # check that same seed yields same prior samples
    assert tf.reduce_all(prior_samples == prior_samples_copy)
    # check that different seed yields different prior samples
    assert not tf.reduce_all(prior_samples == prior_samples_copy2)

    # check (1) order of axes in prior samples correspond to order in
    # parameters-section // (2) numeric values of prior samples from
    # oracle approx. correctly the specified ground truth
    means = tf.reduce_mean(prior_samples, (0, 1))
    stds = tf.math.reduce_std(prior_samples, (0, 1))
    for m, t in zip(means, [-0.5, 0.0, 1.0]):
        assert t == pytest.approx(m, abs=0.01)
    for s, t in zip(stds, [0.8, 0.8, 0.71]):
        assert t == pytest.approx(s, abs=0.01)


# check: parametric_prior, training
def test_prior_samples_2(init_matrix_slice, parameters, expert):
    initialized_priors = el.simulations.intialize_priors(
        init_matrix_slice=init_matrix_slice,
        method="parametric_prior",
        seed=0,
        parameters=parameters,
        network=None,
    )

    prior_samples = el.simulations.sample_from_priors(
        initialized_priors,
        False,
        10,
        5,
        0,
        "parametric_prior",
        parameters,
        None,
        expert,
    )

    prior_samples_copy = el.simulations.sample_from_priors(
        initialized_priors,
        False,
        10,
        5,
        0,
        "parametric_prior",
        parameters,
        None,
        expert,
    )

    prior_samples_copy2 = el.simulations.sample_from_priors(
        initialized_priors,
        False,
        10,
        5,
        1,
        "parametric_prior",
        parameters,
        None,
        expert,
    )

    # check expected shape of prior samples (B, num_samples, num_params)
    assert prior_samples.shape == (5, 10, 3)
    # check that same seed yields same prior samples
    assert tf.reduce_all(prior_samples == prior_samples_copy)
    # check that different seed yields different prior samples
    assert not tf.reduce_all(prior_samples == prior_samples_copy2)


# check: deep_prior, oracle
def test_prior_samples_3(init_matrix_slice, parameters_deep, expert, network):
    pytest.importorskip("bayesflow")

    initialized_priors = el.simulations.intialize_priors(
        init_matrix_slice=init_matrix_slice,
        method="deep_prior",
        seed=0,
        parameters=parameters_deep,
        network=network,
    )

    prior_samples = el.simulations.sample_from_priors(
        initialized_priors,
        True,
        10,
        5,
        0,
        "deep_prior",
        parameters_deep,
        network,
        expert,
    )

    prior_samples_copy = el.simulations.sample_from_priors(
        initialized_priors,
        True,
        10,
        5,
        0,
        "deep_prior",
        parameters_deep,
        network,
        expert,
    )

    prior_samples_copy2 = el.simulations.sample_from_priors(
        initialized_priors,
        True,
        10,
        5,
        1,
        "deep_prior",
        parameters_deep,
        network,
        expert,
    )

    # check expected shape of prior samples (1, num_samples, num_params)
    assert prior_samples.shape == (1, 100_000, 3)
    # check that same seed yields same prior samples
    assert tf.reduce_all(prior_samples == prior_samples_copy)
    # check that different seed yields different prior samples
    assert not tf.reduce_all(prior_samples == prior_samples_copy2)

    # check (1) order of axes in prior samples correspond to order in
    # parameters-section // (2) numeric values of prior samples from
    # oracle approx. correctly the specified ground truth
    means = tf.reduce_mean(prior_samples, (0, 1))
    stds = tf.math.reduce_std(prior_samples, (0, 1))
    for m, t in zip(means, [-0.5, 0.0, 1.0]):
        assert t == pytest.approx(m, abs=0.01)
    for s, t in zip(stds, [0.8, 0.8, 0.71]):
        assert t == pytest.approx(s, abs=0.01)


# check: deep_prior, training
def test_prior_samples_4(init_matrix_slice, parameters_deep, expert, network):
    pytest.importorskip("bayesflow")

    initialized_priors = el.simulations.intialize_priors(
        init_matrix_slice=init_matrix_slice,
        method="deep_prior",
        seed=0,
        parameters=parameters_deep,
        network=network,
    )

    prior_samples = el.simulations.sample_from_priors(
        initialized_priors,
        False,
        10,
        5,
        0,
        "deep_prior",
        parameters_deep,
        network,
        expert,
    )

    prior_samples_copy = el.simulations.sample_from_priors(
        initialized_priors,
        False,
        10,
        5,
        0,
        "deep_prior",
        parameters_deep,
        network,
        expert,
    )

    prior_samples_copy2 = el.simulations.sample_from_priors(
        initialized_priors,
        False,
        10,
        5,
        1,
        "deep_prior",
        parameters_deep,
        network,
        expert,
    )

    # check expected shape of prior samples (B, num_samples, num_params)
    assert prior_samples.shape == (5, 10, 3)
    # check that same seed yields same prior samples
    assert tf.reduce_all(prior_samples == prior_samples_copy)
    # check that different seed yields different prior samples
    assert not tf.reduce_all(prior_samples == prior_samples_copy2)


# %% test simulate_from_generator
# constants
N = 20
B = 5
num_samples = 30


@pytest.fixture
def predictor():
    X = tf.concat([tf.ones(int(N // 2)), tf.zeros(int(N // 2))], 0)
    X_brcst = tf.broadcast_to(X[None, None, :], (B, num_samples, N))
    return X_brcst


@pytest.fixture
def prior_samples():
    return tf.concat(
        [
            tfd.Normal(0.0, 1.0).sample((B, num_samples, 1)),
            tfd.Normal(-0.5, 1.3).sample((B, num_samples, 1)),
        ],
        -1,
    )


@pytest.fixture
def model(predictor):
    class Model:
        def __call__(self, prior_samples, **kwargs):
            # model
            epred = tf.add(
                prior_samples[:, :, 0][:, :, None],
                prior_samples[:, :, 1][:, :, None] * predictor,
            )
            likelihood = tfd.Normal(loc=epred, scale=tf.ones(epred.shape))
            ypred = likelihood.sample()
            return dict(
                ypred=ypred,
                epred=epred,
                prior_samples=prior_samples,
                likelihood=likelihood,
            )

    return dict(obj=Model)


def test_model_samples(prior_samples, model):
    model_sim = el.simulations.simulate_from_generator(prior_samples, 0, model)

    # check whether required output format is correct
    for key in ["ypred", "epred", "likelihood", "prior_samples"]:
        assert key in list(model_sim.keys()), f"{key} not in model simulations"

    # check whether shape is correct
    assert model_sim["ypred"].shape == (B, num_samples, N)
    assert model_sim["epred"].shape == (B, num_samples, N)
