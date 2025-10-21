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
                name="b0",
                family=tfd.Normal,
                hyperparams=dict(loc=el.hyper(name="mu0"), scale=el.hyper("sigma0")),
            )
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
    )


def test_expert_input(eliobj):
    # test correct format
    dat_format = eliobj.expert
    expected_format = get_expert_datformat(eliobj.targets)

    assert dat_format["data"].keys() == expected_format.keys()

    # test wrong format
    eliobj.update(expert=el.expert.data({"b0": 0.5}))
    dat_format_wrong = eliobj.expert

    with pytest.raises(AssertionError):
        assert dat_format_wrong["data"].keys() == expected_format.keys()

    # test ground truth
    eliobj.update(expert=el.expert.simulator(ground_truth={"b0": tfd.Normal(0, 1)}))

    assert "ground_truth" in eliobj.expert.keys()


def test_initializer_network(eliobj):
    # parametric method
    # correct: initialized is configured
    assert eliobj.initializer is not None
    assert eliobj.network is None

    # error message if initializer is None
    eliobj.update(initializer=None, network="some_network")
    ### TODO when updating the checks do not run
    with pytest.raises(ValueError):
        assert eliobj.initializer is not None
