import numpy as np
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions


class TestModel:
    def __call__(self, prior_samples):
        mu = prior_samples

        y_pred = tfd.Normal(loc=mu, scale=1.0).sample()

        return dict(mu=mu, y_pred=y_pred)


parameters = [
    el.parameter(
        name="mu",
        family=tfd.Normal,
        hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0.0)),
    )
]

model = el.model(obj=TestModel)

targets = [
    el.target(
        name="mu", query=el.queries.identity(), loss=el.losses.MMD2(kernel="energy")
    ),
    el.target(
        name="y_pred",
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
    ),
]

trainer = el.trainer(method="parametric_prior", seed=123, epochs=1)

initializer = el.initializer(
    method="sobol",
    iterations=32,
    distribution=el.initialization.uniform(radius=2.0, mean=0.0),
)


def test_dryrun():
    res_dry = el.utils.dry_run(
        model, parameters, targets, trainer, initializer, network=None
    )

    (elicited_statistics, prior_samples, *_) = res_dry

    prior_shape = prior_samples.shape
    elicits_shape = [elicited_statistics[k].shape for k in elicited_statistics]

    np.testing.assert_equal(
        prior_shape, (trainer["B"], trainer["num_samples"], len(parameters))
    )

    for sh, exp_sh in zip(
        elicits_shape,
        [(trainer["B"], trainer["num_samples"], len(parameters)), (trainer["B"], 5)],
    ):
        np.testing.assert_equal(sh, exp_sh)
