"""
Tests of the expansion of one expert into a panel of experts
"""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el
from elicito import _panel

tfd = tfp.distributions

PARAMETERS = [
    el.parameter(
        name="beta0",
        family=tfd.Normal,
        hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)),
    ),
    el.parameter(
        name="sigma",
        family=tfd.HalfNormal,
        hyperparams=dict(scale=el.hyper("s", lower=0)),
    ),
]


def test_parameters_are_copied_in_the_order_of_the_experts():
    panel = _panel.parameters(PARAMETERS, n_experts=2, over=("mu0",))

    assert [p["name"] for p in panel] == [
        "beta0__e0",
        "sigma__e0",
        "beta0__e1",
        "sigma__e1",
    ]


def test_only_the_over_names_get_an_offset():
    panel = _panel.parameters(PARAMETERS, n_experts=2, over=("mu0",))
    hyper = panel[2]["hyperparams"]

    assert hyper["loc__offset"]["name"] == "mu0__e1"
    assert not hyper["loc__offset"]["shared"]
    assert {key: h["name"] for key, h in hyper.items() if h["shared"]} == {
        "loc": "mu0",
        "scale": "sigma0",
    }
    assert set(panel[3]["hyperparams"]) == {"scale"}


def test_the_offset_shifts_a_location():
    (par,) = _panel.parameters(PARAMETERS[:1], n_experts=1, over=("mu0",))
    prior = par["family"](loc=2.0, loc__offset=0.5, scale=1.0)

    assert float(prior.loc) == pytest.approx(2.5)


def test_the_offset_of_a_scale_keeps_the_bound():
    (par,) = _panel.parameters(PARAMETERS[:1], n_experts=1, over=("sigma0",))
    prior = par["family"](loc=0.0, scale=1.0, scale__offset=-50.0)

    assert float(prior.scale) > 0.0
    assert float(prior.scale) < 1.0


def test_targets_are_copied_for_each_expert():
    targets = [el.quantiles("y", probs=(0.5,)), el.correlation("cor")]
    panel = _panel.targets(targets, n_experts=2)

    assert [t["name"] for t in panel] == ["y__e0", "cor__e0", "y__e1", "cor__e1"]
    assert [t["output"] for t in panel] == ["y__e0", "cor__e0", "y__e1", "cor__e1"]


def test_a_custom_target_reads_the_quantities_of_its_expert():
    def stat(y):
        return tf.reduce_mean(y)

    target = el.custom("y", stat, loss=el.losses.L2)
    panel = _panel.targets([target], n_experts=2)
    quantities = {"y__e0": tf.constant([1.0]), "y__e1": tf.constant([3.0])}

    out = el.targets.use_custom_functions(quantities, panel[1]["query"]["value"])
    assert float(out) == 3.0


def _model(prior_samples):
    return dict(y=prior_samples[..., 0] + 10.0 * prior_samples[..., 1])


def test_the_model_runs_for_each_expert_on_its_own_samples():
    model = _panel.Model(_model, n_experts=2, pooled=False, correlations=["cor"])
    samples = tf.constant([[[1.0, 2.0, 3.0, 4.0]]])

    out = model(samples, seed=tf.constant([0, 0]))
    assert out["y__e0"].numpy().item() == 21.0
    assert out["y__e1"].numpy().item() == 43.0
    np.testing.assert_array_equal(out["cor__e1"], [[[3.0, 4.0]]])


def test_a_pooled_model_runs_once_for_all_experts():
    calls = []

    def model(prior_samples):
        calls.append(1)
        return _model(prior_samples)

    panel = _panel.Model(model, n_experts=3, pooled=True, correlations=[])
    out = panel(tf.constant([[[1.0, 2.0]]]), seed=tf.constant([0, 0]))

    assert len(calls) == 1
    assert sorted(out) == ["y__e0", "y__e1", "y__e2"]


def test_the_model_gives_each_expert_its_own_seed():
    def model(prior_samples, seed):
        return dict(y=tf.random.stateless_normal((1,), seed=seed))

    panel = _panel.Model(model, n_experts=2, pooled=False, correlations=[])
    out = panel(tf.zeros((1, 1, 2)), seed=tf.constant([0, 1]))

    assert out["y__e0"].numpy().item() != out["y__e1"].numpy().item()


def test_answers_are_flattened():
    out = _panel.answers([{"y": [1.0]}, {"y": [2.0]}])
    assert out == {"y__e0": [1.0], "y__e1": [2.0]}
