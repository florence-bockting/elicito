"""
test that the consistency diagnostics find a statistic that conflicts
"""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions


class _Copies:
    """Return the prior samples of one parameter under three names"""

    def __call__(self, prior_samples):
        theta = prior_samples[:, :, 0]
        return dict(a=theta, b=theta, c=theta)


def _fit(quantiles_c):
    # the expert quantiles of a and b are fixed; c agrees with them or not
    quantiles = (0.25, 0.50, 0.75)
    eliobj = el.Elicit(
        model=el.model(obj=_Copies),
        parameters=[
            el.parameter(
                name="theta",
                family=tfd.Normal,
                hyperparams=dict(loc=el.hyper("mu"), scale=el.hyper("sigma", lower=0)),
            )
        ],
        targets=[
            el.target(
                name=name,
                query=el.queries.quantiles(quantiles),
                loss=el.losses.MMD2(kernel="energy"),
                weight=1.0,
            )
            for name in ("a", "b", "c")
        ],
        expert=el.expert.data(
            dat={
                "quantiles_a": [-0.67, 0.0, 0.67],
                "quantiles_b": [-0.67, 0.0, 0.67],
                "quantiles_c": quantiles_c,
            }
        ),
        optimizer=el.optimizer(optimizer=tf.keras.optimizers.Adam, learning_rate=0.1),
        # start values on the unconstrained scale
        initializer=el.initializer(method=None, hyperparams=dict(mu=0.0, sigma=0.0)),
        trainer=el.trainer(
            method="parametric_prior",
            seed=1,
            epochs=100,
            B=32,
            num_samples=100,
            progress=0,
        ),
    )
    eliobj.fit()
    return eliobj


TOLERANCE = {f"quantiles_{name}": 0.1 for name in "abc"}


@pytest.fixture(scope="module")
def eliobj():
    return _fit([4.33, 5.0, 5.67])


@pytest.fixture(scope="module")
def consistent():
    return _fit([-0.67, 0.0, 0.67])


def test_leave_one_out_finds_the_statistic_that_conflicts(eliobj):
    reference = el.diagnostics.baseline(eliobj)
    diagnostic = el.diagnostics.leave_one_out(eliobj, reference, TOLERANCE)

    assert list(diagnostic.statistic.values) == [
        "quantiles_a",
        "quantiles_b",
        "quantiles_c",
    ]
    assert np.argmax(diagnostic["mean_miss"].values) == 2
    assert diagnostic.attrs["inconsistent"]
    assert list(diagnostic["leading"].values) == [False, False, True]


def test_parallel_replications_draw_with_their_own_seed(eliobj):
    refit = el.diagnostics._refit(eliobj, eliobj.targets, [1, 2])
    losses = el.diagnostics._final_losses(refit)["total_loss"].values

    # the same prior sample would give the same loss up to float noise
    assert abs(losses[0] - losses[1]) > 1e-4


def test_leave_one_out_does_not_flag_consistent_statistics(consistent):
    reference = el.diagnostics.baseline(consistent)
    diagnostic = el.diagnostics.leave_one_out(consistent, reference, TOLERANCE)

    assert not diagnostic.attrs["inconsistent"]
    assert not diagnostic["leading"].values.any()


def test_baseline_reads_the_statistics_of_a_simulator_expert(consistent):
    simulator = el.Elicit(
        model=consistent.model,
        parameters=consistent.parameters,
        targets=consistent.targets,
        expert=el.expert.simulator(
            ground_truth={"theta": tfd.Normal(0.0, 1.0)}, num_samples=10_000
        ),
        optimizer=consistent.optimizer,
        initializer=consistent.initializer,
        trainer=consistent.trainer,
    )
    simulator.fit()

    reference = el.diagnostics.baseline(simulator)

    assert np.isfinite(reference["distance"].values).all()
