"""
Tests of the fit of a panel of experts
"""

import pytest
import tensorflow_probability as tfp

import elicito as el
from elicito.elicit import _is_panel
from elicito.fit import Fit, Panel

tfd = tfp.distributions

PARAMETERS = [
    el.parameter(
        name="beta0",
        family=tfd.Normal,
        hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)),
    ),
]
TARGETS = [el.quantiles("y", probs=(0.25, 0.50, 0.75))]
EXPERTS = {
    "expert 1": {"y": [-0.7, 0.0, 0.7]},
    "expert 2": {"y": [1.3, 2.0, 2.7]},
}


def _model(prior_samples):
    return dict(y=prior_samples[..., 0])


def _eliobj(pool=None):
    return el.Elicit(
        _model,
        TARGETS,
        PARAMETERS,
        pool=pool,
        epochs=40,
        B=4,
        num_samples=100,
        verbose=False,
        dry_run=False,
    )


def test_a_dict_of_dicts_is_a_panel():
    oracle = el.expert.simulator(ground_truth={"beta0": tfd.Normal(0, 1)})

    assert _is_panel(EXPERTS)
    assert not _is_panel(EXPERTS["expert 1"])
    assert not _is_panel(oracle)


def test_a_mixed_dict_raises():
    with pytest.raises(ValueError, match="mix answers and experts"):
        _is_panel({"expert 1": {"y": [0.0, 1.0, 2.0]}, "y": [0.0, 1.0, 2.0]})


def test_a_pool_for_one_expert_warns():
    with pytest.warns(UserWarning, match="answers are of one expert"):
        fit = _eliobj(pool="complete").fit(EXPERTS["expert 1"])
    assert isinstance(fit, Fit)


def test_a_pool_for_a_panel_of_one_warns():
    with pytest.warns(UserWarning, match="panel has one expert"):
        panel = _eliobj(pool="complete").fit({"expert 1": EXPERTS["expert 1"]})
    assert panel.pool is None


def test_no_pool_fits_each_expert_alone():
    panel = _eliobj().fit(EXPERTS)

    assert isinstance(panel, Panel)
    assert panel.experts == ["expert 1", "expert 2"]
    assert panel["expert 1"].hyperparameters()["mu0"] < 1.0
    assert panel["expert 2"].hyperparameters()["mu0"] > 1.0


def test_complete_pooling_fits_one_prior():
    panel = _eliobj(pool="complete").fit(EXPERTS)

    assert set(panel.joint.hyperparameters()) == {"mu0", "sigma0"}
    with pytest.raises(KeyError, match="panel.joint"):
        panel["expert 1"]


def test_partial_pooling_with_a_fixed_tau_learns_offsets():
    panel = _eliobj(pool=el.partial(over=["mu0"], tau=1.0)).fit(EXPERTS)

    hyper = panel.joint.hyperparameters()
    assert set(hyper) == {"mu0", "sigma0", "mu0__e0", "mu0__e1"}
    assert hyper["mu0__e0"] < hyper["mu0__e1"]
    assert panel.tau == {"mu0": 1.0}


def test_partial_pooling_estimates_tau_in_two_stages():
    panel = _eliobj(pool=el.partial(over=["mu0"])).fit(EXPERTS)

    assert set(panel.tau) == {"mu0"}
    assert panel.tau["mu0"] > 0.1


def test_a_panel_takes_an_oracle():
    experts = {
        "expert 1": EXPERTS["expert 1"],
        "oracle": el.expert.simulator(
            ground_truth={"beta0": tfd.Normal(2.0, 1.0)}, num_samples=1_000
        ),
    }
    panel = _eliobj(pool=el.partial(over=["mu0"], tau=1.0)).fit(experts)

    hyper = panel.joint.hyperparameters()
    assert hyper["mu0__e0"] < hyper["mu0__e1"]
