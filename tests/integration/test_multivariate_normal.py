import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions


# %% generative model
class GenerativeModel:
    def __call__(self, prior_samples, n_gr):
        B, num_samples, _ = prior_samples.shape

        design_matrix = tf.stack([[1.0] * (n_gr * 2), [0.0] * n_gr + [1] * n_gr], 1)
        mu = prior_samples @ tf.transpose(design_matrix)

        y_pred = tfd.Normal(mu, 1.0).sample()

        return dict(y_gr0=y_pred[:, :, :n_gr], y_gr1=y_pred[:, :, n_gr:])


# %% ground truth
sds = [2.5, 1.3]
cor = [[1.0, 0.3], [0.3, 1.0]]
cov = (tf.linalg.diag(sds) @ cor) @ tf.linalg.diag(sds)

betas = tfd.MultivariateNormalTriL(loc=[0.5, 1.0], scale_tril=tf.linalg.cholesky(cov))


# %% eliobj
@pytest.fixture
def eliobj():
    return el.Elicit(
        model=el.model(obj=GenerativeModel, n_gr=15),
        parameters=[
            el.parameter(
                name="betas",
                family=el.utils.MultivariateNormal,
                hyperparams=dict(
                    loc=el.hyper("mus", vtype="array", dim=2),
                    scale=el.hyper("sigmas", lower=0.0, vtype="array", dim=2),
                    cor=el.hyper("cor", vtype="correlation", dim=2),
                ),
            ),
        ],
        targets=[
            el.target(
                name=f"y_gr{i}",
                loss=el.losses.MMD2(kernel="energy"),
                query=el.queries.quantiles(quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)),
            )
            for i in range(2)
        ]
        + [
            el.target(
                name="cor",
                loss=el.losses.L2,
                weight=0.1,
                query=el.queries.correlation(),
            )
        ],
        expert=el.expert.simulator(ground_truth=dict(betas=betas), num_samples=10_000),
        trainer=el.trainer(method="parametric_prior", seed=1, epochs=300),
        optimizer=el.optimizer(
            optimizer=tf.keras.optimizers.Adam,
            learning_rate=0.05,
        ),
        network=None,
        initializer=el.initializer(
            hyperparams=dict(mus=[0.0, 0.0], sigmas=[0.5, 0.5], cor=[1.0])
        ),
    )


def test_fit_eliobj(eliobj):
    eliobj.fit()

    np.testing.assert_array_almost_equal(
        tf.reduce_mean(eliobj.history[0]["hyperparameter"]["sigmas"][-30:], 0),
        sds,
        decimal=1,
    )

    np.testing.assert_array_almost_equal(
        tf.reduce_mean(eliobj.history[0]["hyperparameter"]["mus"][-30:], 0),
        [0.5, 1.0],
        decimal=2,
    )

    cor_list = []
    for i in range(30):
        cov = el.utils.CorrelationMatrix(K=2)
        cor_list.append(
            cov._get_upper_triangular(
                eliobj.history[0]["hyperparameter"]["cor"][-(30 - i)]
            )
        )

    np.testing.assert_almost_equal(tf.reduce_mean(cor_list), 0.3, decimal=2)
