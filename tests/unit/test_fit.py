"""
Unittest for fit.py module
"""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el
from elicito.fit import Fit

tfd = tfp.distributions


class LinearModel:
    def __call__(self, prior_samples):
        epred = prior_samples[:, :, 0:1] + prior_samples[:, :, 1:2] * tf.constant(
            [-1.0, 0.0, 1.0]
        )
        ypred = tfd.Normal(epred, 1.0).sample()
        return dict(y0=ypred[:, :, 0], y1=ypred[:, :, 1], y2=ypred[:, :, 2])


@pytest.fixture(scope="module")
def fit():
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
                "y0": [-1.0, 0.0, 1.0],
                "y1": [0.0, 1.0, 2.0],
                "y2": [1.0, 2.0, 3.0],
            }
        ),
        optimizer=el.optimizer(optimizer=tf.keras.optimizers.Adam, learning_rate=0.1),
        trainer=el.trainer(
            method="parametric_prior",
            seed=1,
            epochs=2,
            B=8,
            num_samples=50,
            progress=0,
        ),
        initializer=el.initializer(
            "random",
            distribution=el.initializers.uniform(radius=1, mean=0),
            iterations=2,
        ),
    )
    return eliobj.fit(parallel=el.utils.parallel(runs=2, cores=1))


def test_fit_returns_a_fit(fit):
    assert isinstance(fit, Fit)


def test_best_is_a_valid_index(fit):
    assert fit.best() in (0, 1)


def test_best_hyperparameters_are_those_of_the_best_replication(fit):
    assert fit.hyperparameters() == fit.hyperparameters(fit.best())


def test_default_sample_is_cached(fit):
    assert fit.sample() is fit.sample()


def test_sample_with_a_seed_is_not_cached(fit):
    assert fit.sample(seed=3) is not fit.sample()


def test_summary_has_one_group_per_table(fit):
    summary = fit.summary()

    assert set(summary.children) == {
        "hyperparameters",
        "priors",
        "expert_fit",
        "target_loss",
        "loss",
    }
    assert summary.attrs["reps"] == [fit.best()]


def test_summary_names_the_technique_of_each_target(fit):
    target_loss = fit.summary()["target_loss"]

    assert list(target_loss["technique"].values) == ["quantiles"] * 3
    assert "technique" in repr(fit)


def test_summary_of_all_replications_keeps_each_replication(fit):
    summary = fit.summary(replication="all")

    np.testing.assert_array_equal(summary["priors"]["replication"].values, [0, 1])
    assert "rep_sd" in summary["hyperparameters"]


def test_summary_error_is_elicited_minus_expert(fit):
    table = fit.summary()["expert_fit"].to_dataset()

    np.testing.assert_allclose(table.error, table.elicited - table.expert)


def test_repr_shows_the_summary(fit):
    text = repr(fit)

    assert text.startswith("Method: parametric_prior (2 epochs)")
    assert "Loss by target:" in text


def test_plot_lists_the_bound_plots(fit):
    assert "loss" in dir(fit.plot)


def test_unknown_plot_raises(fit):
    with pytest.raises(AttributeError, match="No plot named 'missing'"):
        fit.plot.missing()
