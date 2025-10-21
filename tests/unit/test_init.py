"""
Unittest for init.py module
"""

import re
from unittest.mock import patch

import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from tests.utils import eliobj as base_eliobj

import elicito as el
from elicito import Elicit
from elicito.utils import get_expert_datformat

tfd = tfp.distributions

scipy = pytest.importorskip("scipy")


class TestModel:
    def __call__(self, prior_samples):
        b0 = tfd.Normal(0, 1).sample((3, 3, 1))
        return dict(prior_samples=prior_samples, b0=b0)


@pytest.fixture
def eliobj():
    return Elicit(
        model=el.model(TestModel),
        parameters=[
            el.parameter(
                name=f"b{i}",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper(name=f"mu{i}"), scale=el.hyper(f"sigma{i}")
                ),
            )
            for i in range(2)
        ],
        targets=[
            el.target(name="b0", loss=el.losses.L2, query=el.queries.quantiles((0.5,)))
        ],
        expert=el.expert.data({"quantiles_b0": 0.5}),
        optimizer=el.optimizer(optimizer=tf.keras.optimizers.Adam, learning_rate=0.1),
        trainer=el.trainer(method="parametric_prior", seed=42, epochs=1),
        initializer=el.initializer(
            "sobol",
            distribution=el.initialization.uniform(radius=1, mean=0),
            iterations=1,
        ),
        network=None,
    )


@pytest.fixture
def network():
    return el.networks.NF(
        inference_network=el.networks.InvertibleNetwork,
        network_specs=dict(
            num_params=2,
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


def test_expert_input(eliobj):
    # test correct format
    dat_format = eliobj.expert
    expected_format = get_expert_datformat(eliobj.targets)

    assert dat_format["data"].keys() == expected_format.keys()

    # error; wrong format
    msg = (
        "Provided expert data is not in the "
        "correct format. Please use "
        "el.utils.get_expert_datformat to check expected format."
    )

    with pytest.raises(AssertionError, match=msg):
        eliobj.update(expert=el.expert.data({"b0": 0.5}))

    # error; wrong number of parameters
    msg2 = (
        "Dimensionality of ground truth in "  # type: ignore
        "'expert' is not the same  as number of model "
        "parameters. Got num_params=1, expected "
        "2."
    )
    with pytest.raises(AssertionError, match=msg2):
        eliobj.update(expert=el.expert.simulator(ground_truth={"b0": tfd.Normal(0, 1)}))


def test_initializer_network(eliobj, network):
    # parametric method
    # correct: initialized is configured
    assert eliobj.initializer is not None
    assert eliobj.network is None

    # error message if initializer is None
    msg1 = (
        "If method is 'parametric_prior', " " the section 'initializer' can't be None."
    )
    with pytest.raises(ValueError, match=msg1):
        eliobj.update(initializer=None, network=None)

    msg2 = (
        "If method is 'parametric prior' "
        "the 'network' is not used and should be set to None."
    )
    with pytest.raises(ValueError, match=msg2):
        eliobj.update(network=network)

    # correct
    eliobj.update(
        trainer=el.trainer(method="deep_prior", epochs=1, seed=123),
        network=network,
        initializer=None,
    )

    assert eliobj.network == network

    msg3 = "If method is 'deep prior', " " the section 'network' can't be None."
    with pytest.raises(ValueError, match=msg3):
        eliobj.update(network=None)

    msg4 = (
        "For method 'deep_prior' the "
        "'initializer' is not used and should be set to None."
    )
    with pytest.raises(ValueError, match=msg4):
        eliobj.update(
            initializer=el.initializer(
                "sobol",
                distribution=el.initialization.uniform(radius=1, mean=0),
                iterations=1,
            )
        )


def test_network_parameter(eliobj, network):
    # error due to wrong number of parameters
    msg = (
        "The number of model parameters as "
        "specified in the parameters section, must match the "
        "number of parameters specified in the network."
        "Expected 1 but got 2"
    )
    with pytest.raises(ValueError, match=msg):
        eliobj.update(
            trainer=el.trainer(method="deep_prior", epochs=1, seed=123),
            initializer=None,
            network=network,
            parameters=[
                el.parameter(
                    name="b0",
                    family=tfd.Normal,
                    hyperparams=dict(
                        loc=el.hyper(name="mu0"), scale=el.hyper("sigma0")
                    ),
                )
            ],
        )


def test_network_base(eliobj):
    # error; distribution is not basenormal
    msg = (
        "Currently only the standard normal distribution "
        "is implemented as base distribution. "
        "See GitHub issue #35."
    )
    with pytest.raises(NotImplementedError, match=msg):
        eliobj.update(
            trainer=el.trainer(method="deep_prior", epochs=1, seed=123),
            initializer=None,
            network=el.networks.NF(
                inference_network=el.networks.InvertibleNetwork,
                network_specs=dict(
                    num_params=2,
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
                base_distribution=tfd.Normal,
            ),
        )


def test_shared_hyperparameter(eliobj):
    # raise error if same name for hyperparameter
    # but shared is False (default)
    msg = (
        "The following hyperparameter have the same "
        "name but are not shared: ['mu0', 'sigma0']. \n"
        "Have you forgot to set shared=True?"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        eliobj.update(
            parameters=[
                el.parameter(
                    name=f"b{i}",
                    family=tfd.Normal,
                    hyperparams=dict(
                        loc=el.hyper(name="mu0"), scale=el.hyper("sigma0")
                    ),
                )
                for i in range(2)
            ]
        )

    # correct; specify shared=True
    eliobj.update(
        parameters=[
            el.parameter(
                name=f"b{i}",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper(name="mu0", shared=True),
                    scale=el.hyper("sigma0", shared=True),
                ),
            )
            for i in range(2)
        ]
    )

    for i in range(2):
        assert eliobj.parameters[i].hyperparams["loc"]["name"] == "mu0"
        assert eliobj.parameters[i].hyperparams["scale"]["name"] == "sigma0"


def test_initializer_setup(eliobj):
    # correct; hyperparameter names match
    eliobj.update(
        initializer=el.initializer(
            hyperparams=dict(mu0=0.1, sigma0=0.2, mu1=0.1, sigma1=0.2)
        )
    )

    assert eliobj.initializer["hyperparams"]["mu0"] == 0.1
    assert eliobj.initializer["hyperparams"]["sigma1"] == 0.2

    # error; hyperparameter names do not match
    msg = (
        "Hyperparameter name 'mu2' doesn't "
        "match any name specified in the parameters "
        "section. Have you misspelled the name?"
    )
    with pytest.raises(ValueError, match=msg):
        eliobj.update(
            initializer=el.initializer(
                hyperparams=dict(mu2=0.1, sigma2=0.2, mu1=0.1, sigma1=0.2)
            )
        )


def test_hyperparameters(eliobj):
    # error; as parametric prior needs family and hyperparams arg.
    msg = (
        "When using method='parametric_prior', the argument "
        "'hyperparams' of el.parameter "
        "cannot be None."
    )
    with pytest.raises(ValueError, match=msg):
        eliobj.update(parameters=[el.parameter(name=f"b{i}") for i in range(2)])


def test_checks_fit(capsys):
    base_eliobj.results = [1, 2, 3]

    # negate
    with patch("builtins.input", side_effect=["n"]):
        base_eliobj.fit(overwrite=False)

    captured = capsys.readouterr()
    assert "not re-fitted" in captured.out

    # wrong input
    msg = "Invalid input. Please use 'y' or 'n'."
    with pytest.raises(ValueError, match=msg):
        with patch("builtins.input", side_effect=["x"]):
            base_eliobj.fit(overwrite=False)

    # approve
    with patch("builtins.input", side_effect=["y"]):
        base_eliobj.fit(overwrite=False)

    epochs = base_eliobj.results["history_stats"]["epoch"].shape[0]
    assert base_eliobj.results["history_stats"]["loss"]["total_loss"].values.shape == (
        1,
        epochs,
    )
