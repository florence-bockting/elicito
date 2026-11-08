import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions


@pytest.mark.parametrize("keys", [("a", "b"), ("b", "a")])
def test_ground_truth_keys_name_the_parameters(keys):
    """The keys, not the dict order, decide which prior a parameter gets"""
    parameters = [el.parameter(name="a"), el.parameter(name="b")]
    truth = {"a": tfd.Normal(-10.0, 1.0), "b": tfd.Normal(10.0, 1.0)}
    expert = el.expert.simulator({key: truth[key] for key in keys}, num_samples=1_000)

    samples = el.parameters.priors.sample_from_priors(
        initialized_priors=None,
        ground_truth=True,
        num_samples=1_000,
        B=1,
        seed=1,
        method="parametric_prior",
        parameters=parameters,
        network=None,
        expert=expert,
    )

    np.testing.assert_allclose(tf.reduce_mean(samples, (0, 1)), [-10.0, 10.0], atol=0.5)
