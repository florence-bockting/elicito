"""
The consistency check finds a target that contradicts the others

In the linear model y_x = b0 + b1 * x + noise, the median at x = 0 lies
halfway between the medians at x = -1 and x = 1. Expert quantiles at x = 0
that are shifted upwards therefore contradict the other two targets.
"""

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

EPOCHS = 600


class LinearModel:
    def __call__(self, prior_samples):
        epred = prior_samples[:, :, 0:1] + prior_samples[:, :, 1:2] * tf.constant(
            [-1.0, 0.0, 1.0]
        )
        ypred = tfd.Normal(epred, 1.0).sample()
        return dict(y0=ypred[:, :, 0], y1=ypred[:, :, 1], y2=ypred[:, :, 2])


def fit(shift: float):
    # quantiles of the prior predictive under b0 ~ N(1, 1), b1 ~ N(1, 0.5)
    consistent = {
        "quantiles_y0": [-1.0, 0.0, 1.0],
        "quantiles_y1": [0.05, 1.0, 1.95],
        "quantiles_y2": [1.0, 2.0, 3.0],
    }
    data = {
        **consistent,
        "quantiles_y1": [q + shift for q in consistent["quantiles_y1"]],
    }
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
        expert=el.expert.data(data),
        optimizer=el.optimizer(
            optimizer=tf.keras.optimizers.Adam,
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                0.1, EPOCHS, 0.05
            ),
            clipnorm=1.0,
        ),
        trainer=el.trainer(
            method="parametric_prior", seed=3, epochs=EPOCHS, progress=0
        ),
        initializer=el.initializer(
            "random",
            distribution=el.initializers.uniform(radius=1, mean=0),
            iterations=5,
        ),
    )
    eliobj.fit()
    return eliobj


@pytest.fixture(scope="module")
def consistent():
    return fit(shift=0.0)


@pytest.fixture(scope="module")
def shifted():
    return fit(shift=3.0)


def test_shifted_target_has_the_largest_excess_loss(shifted):
    result = el.consistency.detect(shifted, num_draws=50)

    excess = result.observed - result.null_q95
    assert excess.idxmax().item() == "y1"
    assert float(result.p_value.sel(target="y1")) < 0.05


def test_slack_shifts_only_the_shifted_target(shifted):
    result = el.consistency.slack(shifted, penalty=[0.3])

    size = result["size"].sel(penalty=0.3)
    assert float(size.sel(target="y0")) == 0
    assert float(size.sel(target="y2")) == 0
    # the expert quantiles of y1 are 3 too high
    shift = result["quantiles_y1"].sel(penalty=0.3).values
    assert all(2 < s < 4 for s in shift)


def test_slack_keeps_consistent_targets_at_zero(consistent):
    result = el.consistency.slack(consistent, penalty=[0.3])

    assert float(result["size"].max()) == 0
