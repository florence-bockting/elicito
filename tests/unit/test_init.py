"""
Unittest for init.py module
"""

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el
from elicito import Elicit
from elicito.utils import get_expert_datformat

tfd = tfp.distributions


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

    # test wrong format
    with pytest.raises(AssertionError):
        eliobj.update(expert=el.expert.data({"b0": 0.5}))

    # test ground truth
    eliobj.update(expert=el.expert.simulator(ground_truth={"b0": tfd.Normal(0, 1)}))

    assert "ground_truth" in eliobj.expert.keys()


def test_initializer_network(eliobj, network):
    # parametric method
    # correct: initialized is configured
    assert eliobj.initializer is not None
    assert eliobj.network is None

    # error message if initializer is None
    with pytest.raises(ValueError):
        eliobj.update(initializer=None, network=network)

    # deep_prior method
    # error message if network is None
    with pytest.raises(ValueError):
        eliobj.update(trainer=el.trainer(method="deep_prior", epochs=1, seed=123))

    # correct
    eliobj.update(
        trainer=el.trainer(method="deep_prior", epochs=1, seed=123),
        network=network,
        initializer=None,
    )

    assert eliobj.network == network


def test_shared_hyperparameter(eliobj):
    # raise error if same name for hyperparameter
    # but shared is False (default)
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        eliobj.update(
            initializer=el.initializer(
                hyperparams=dict(mu2=0.1, sigma2=0.2, mu1=0.1, sigma1=0.2)
            )
        )


def test_hyperparameters(eliobj):
    # error; as parametric prior needs family and hyperparams arg.
    with pytest.raises(ValueError):
        eliobj.update(parameters=[el.parameter(name=f"b{i}") for i in range(2)])
