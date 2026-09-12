"""Error messages must be strings, not tuples."""

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from elicito.initializers.sampling import uniform_samples
from elicito.targets import computation_elicited_statistics
from elicito.utils import gumbel_softmax_trick

tfd = tfp.distributions


@pytest.mark.parametrize("mean,radius", [([0.0], 1.0), (0.0, [1.0])])
def test_uniform_samples_hyppar_none(mean, radius):
    with pytest.raises(ValueError) as excinfo:
        uniform_samples(
            seed=1,
            hyppar=None,
            n_samples=2,
            method="random",
            mean=mean,
            radius=radius,
            parameters=[],
        )
    assert not str(excinfo.value).startswith("(")


def test_uniform_samples_hyppar_not_list():
    with pytest.raises(ValueError) as excinfo:
        uniform_samples(
            seed=1,
            hyppar=["a"],
            n_samples=2,
            method="random",
            mean=0.0,
            radius=1.0,
            parameters=[],
        )
    assert not str(excinfo.value).startswith("(")


def test_elicited_statistics_rank():
    targets = [{"name": "y", "query": {"name": "quantiles", "value": [50]}}]
    with pytest.raises(ValueError) as excinfo:
        computation_elicited_statistics({"y": tf.zeros((2, 3, 4, 5))}, targets)
    assert not str(excinfo.value).startswith("(")


def test_gumbel_softmax_trick_rank():
    likelihood = tfd.Normal(tf.zeros((2, 3)), 1.0)
    with pytest.raises(ValueError) as excinfo:
        gumbel_softmax_trick(likelihood, upper_thres=10.0)
    assert not str(excinfo.value).startswith("(")
