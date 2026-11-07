from functools import partial

import pytest

from elicito._checks import check_model


class NoPriorSamples:
    def __call__(self):
        return "Hello"


class NeedsB:
    def __call__(self, prior_samples, b):
        return b


class WithSeed:
    def __call__(self, prior_samples, seed):
        return prior_samples


def test_check_model_rejects_a_class():
    with pytest.raises(TypeError, match="instance of a class"):
        check_model(NeedsB)


def test_check_model_needs_prior_samples():
    with pytest.raises(ValueError, match="'prior_samples'"):
        check_model(NoPriorSamples())


def test_check_model_needs_bound_arguments():
    with pytest.raises(ValueError, match=r"\['b'\] have no value"):
        check_model(NeedsB())


def test_check_model_accepts_partial_and_seed():
    check_model(partial(NeedsB(), b=1))
    check_model(WithSeed())
