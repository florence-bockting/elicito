"""
Unit tests for output hyperparameter names.
"""

import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions


def test_output_hyperparameter_names():
    summary_data = {
        "quantiles_y_obs": [0, -0.012148, 0.049513, 0.100585, 0.153291, 0.634314]
    }

    class GenerativeModel:
        def __call__(self, prior_samples, N, se_y):
            sigma_j = tf.cast(se_y, tf.float32)
            mu = prior_samples[:, :, 0]
            tau = prior_samples[:, :, 1]

            theta_j = tfd.Normal(loc=mu, scale=tau).sample(N)
            y_pred = tfd.Normal(
                loc=tf.transpose(theta_j, perm=[1, 2, 0]), scale=sigma_j[None, None, :]
            ).sample()

            return dict(y_obs=y_pred)

    eliobj = el.Elicit(
        model=el.model(obj=GenerativeModel, N=50, se_y=tfd.Normal(0, 1).sample(50)),
        parameters=[
            el.parameter(
                name="mu",
                family=tfd.Normal,
                hyperparams=dict(loc=el.hyper("mu_mu"), scale=el.hyper("sigma_mu")),
            ),
            el.parameter(
                name="tau",
                family=tfd.Normal,
                hyperparams=dict(loc=el.hyper("l_tau"), scale=el.hyper("d_tau")),
            ),
        ],
        targets=[
            el.target(
                name="y_obs",
                query=el.queries.quantiles([0.05, 0.25, 0.5, 0.75, 0.95]),
                loss=el.losses.MMD2(kernel="energy"),
            ),
        ],
        expert=el.expert.data(summary_data),
        optimizer=el.optimizer(
            optimizer=tf.keras.optimizers.Adam, learning_rate=0.0, clipnorm=1.0
        ),
        trainer=el.trainer(method="parametric_prior", seed=123, epochs=1, progress=1),
        initializer=el.initializer(
            hyperparams=dict(mu_mu=1.0, sigma_mu=0.5, l_tau=0.1, d_tau=0.4)
        ),
    )

    eliobj.fit()

    res_dict = {}
    for var in ["mu_mu", "sigma_mu", "l_tau", "d_tau"]:
        res_dict[var] = eliobj.results.history_stats.hyperparameter[var].values.ravel()[
            -1
        ]

    expected_dict = dict(mu_mu=1.0, sigma_mu=0.5, l_tau=0.1, d_tau=0.4)

    for var in ["mu_mu", "sigma_mu", "l_tau", "d_tau"]:
        assert res_dict[var] == expected_dict[var]
