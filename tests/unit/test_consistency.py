"""
Unittest for consistency.py module
"""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions


class LinearModel:
    def __call__(self, prior_samples):
        epred = prior_samples[:, :, 0:1] + prior_samples[:, :, 1:2] * tf.constant(
            [-1.0, 0.0, 1.0]
        )
        ypred = tfd.Normal(epred, 1.0).sample()
        return dict(y0=ypred[:, :, 0], y1=ypred[:, :, 1], y2=ypred[:, :, 2])


@pytest.fixture(scope="module")
def eliobj():
    eliobj = el.Elicit(
        model=el.model(LinearModel),
        parameters=[
            el.parameter(
                name=f"b{i}",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper(f"mu{i}"), scale=el.hyper(f"sigma{i}", lower=0)
                ),
            )
            for i in range(2)
        ],
        targets=[
            el.target(
                name=f"y{i}",
                query=el.queries.quantiles((0.25, 0.5, 0.75)),
                loss=el.losses.MMD2(kernel="energy"),
                weight=1.0,
            )
            for i in range(3)
        ],
        expert=el.expert.data(
            {
                "quantiles_y0": [-1.0, 0.0, 1.0],
                "quantiles_y1": [0.0, 1.0, 2.0],
                "quantiles_y2": [1.0, 2.0, 3.0],
            }
        ),
        optimizer=el.optimizer(optimizer=tf.keras.optimizers.Adam, learning_rate=0.1),
        trainer=el.trainer(
            method="parametric_prior", seed=1, epochs=2, B=8, num_samples=50
        ),
        initializer=el.initializer(
            "random",
            distribution=el.initializers.uniform(radius=1, mean=0),
            iterations=2,
        ),
    )
    eliobj.fit()
    return eliobj


def test_detect_returns_one_row_per_target(eliobj):
    result = el.consistency.detect(eliobj, num_draws=5, num_samples=100)

    assert list(result.target.values) == ["y0", "y1", "y2"]
    assert set(result.data_vars) == {"observed", "null_mean", "null_q95", "p_value"}
    assert np.all(result.p_value > 0)
    assert np.all(result.p_value <= 1)


def test_detect_default_num_samples_for_expert_data(eliobj):
    result = el.consistency.detect(eliobj, num_draws=1)

    assert result.attrs["num_samples"] == el.consistency.EXPERT_NUM_SAMPLES


def test_detect_is_reproducible(eliobj):
    first = el.consistency.detect(eliobj, num_draws=3, num_samples=100, seed=7)
    second = el.consistency.detect(eliobj, num_draws=3, num_samples=100, seed=7)

    np.testing.assert_array_equal(first.null_mean, second.null_mean)


def test_best_replication_is_a_valid_index(eliobj):
    best = el.consistency.best_replication(eliobj)

    assert best == 0


def test_calibrate_penalty_is_positive(eliobj):
    penalty = el.consistency.calibrate_penalty(eliobj, num_draws=5, num_samples=100)

    assert penalty > 0


def test_slack_default_penalty_is_the_calibrated_one(eliobj):
    result = el.consistency.slack(eliobj, epochs=1)

    expected = el.consistency.calibrate_penalty(eliobj)
    np.testing.assert_allclose(result.penalty.values, [expected])


def test_slack_returns_one_row_per_penalty(eliobj):
    result = el.consistency.slack(eliobj, penalty=[0.1, 100.0], epochs=3)

    assert result["size"].dims == ("penalty", "target")
    assert result["size"].shape == (2, 3)
    assert result["quantiles_y1"].shape == (2, 3)
    assert result["loss"].shape == (2,)


def test_slack_large_penalty_keeps_every_shift_at_zero(eliobj):
    result = el.consistency.slack(eliobj, penalty=[100.0], epochs=3)

    np.testing.assert_array_equal(result["size"].values, 0.0)


@pytest.mark.parametrize(
    "function", ["detect", "slack", "calibrate_penalty", "best_replication"]
)
def test_unfitted_eliobj_raises(eliobj, function):
    unfitted = el.Elicit(
        model=eliobj.model,
        parameters=eliobj.parameters,
        targets=eliobj.targets,
        expert=eliobj.expert,
        optimizer=eliobj.optimizer,
        trainer=eliobj.trainer,
        initializer=eliobj.initializer,
    )

    with pytest.raises(AttributeError, match="Run 'eliobj.fit\\(\\)' first"):
        getattr(el.consistency, function)(unfitted)
