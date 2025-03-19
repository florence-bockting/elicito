"""
helper functions for tests
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

import elicito as el

tfd = tfp.distributions


# numeric, standardized predictor
def std_predictor(N, quantiles):
    X = tf.cast(np.arange(N), tf.float32)
    X_std = (X - tf.reduce_mean(X)) / tf.math.reduce_std(X)
    X_sel = tfp.stats.percentile(X_std, quantiles)
    return X_sel


def log_R2(ypred: tf.Tensor, epred: tf.Tensor) -> float:
    """
    compute log R2

    Parameters
    ----------
    ypred
        model predictions
    epred
        predictions of linear predictor

    Returns
    -------
    log R2 :
        log R2
    """
    var_epred = tf.math.reduce_variance(epred, -1)
    # variance of difference between ypred and epred
    var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
    var_total = var_epred + var_diff
    # variance of linear predictor divided by total variance
    log_R2 = tf.subtract(tf.math.log(var_epred), tf.math.log(var_total))
    return log_R2


# implemented, generative model
class ToyModel2:
    def __call__(self, prior_samples, design_matrix):
        B = prior_samples.shape[0]
        S = prior_samples.shape[1]

        # preprocess shape of design matrix
        X = tf.broadcast_to(design_matrix[None, None, :], (B, S, len(design_matrix)))

        # linear predictor (= mu)
        epred = tf.add(
            prior_samples[:, :, 0][:, :, None],
            tf.multiply(prior_samples[:, :, 1][:, :, None], X),
        )
        # data-generating model
        likelihood = tfd.Normal(
            loc=epred, scale=tf.expand_dims(prior_samples[:, :, -1], -1)
        )
        # prior predictive distribution (=height)
        ypred = likelihood.sample()

        # selected observations
        y_X0, y_X1, y_X2 = (ypred[:, :, 0], ypred[:, :, 1], ypred[:, :, 2])

        # log R2 (log for numerical stability)
        logR2 = log_R2(ypred, epred)

        return dict(
            likelihood=likelihood,
            ypred=ypred,
            epred=epred,
            prior_samples=prior_samples,
            y_X0=y_X0,
            y_X1=y_X1,
            y_X2=y_X2,
            log_R2=logR2,
        )


ground_truth = {
    "beta0": tfd.Normal(loc=5, scale=1),
    "beta1": tfd.Normal(loc=2, scale=1),
    "sigma": tfd.HalfNormal(scale=10.0),
}


eliobj = el.Elicit(
    model=el.model(
        obj=ToyModel2, design_matrix=std_predictor(N=200, quantiles=[25, 50, 75])
    ),
    parameters=[
        el.parameter(
            name="beta0",
            family=tfd.Normal,
            hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)),
        ),
        el.parameter(
            name="beta1",
            family=tfd.Normal,
            hyperparams=dict(loc=el.hyper("mu1"), scale=el.hyper("sigma1", lower=0)),
        ),
        el.parameter(
            name="sigma",
            family=tfd.HalfNormal,
            hyperparams=dict(scale=el.hyper("sigma2", lower=0)),
        ),
    ],
    targets=[
        el.target(
            name="y_X0",
            query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0,
        ),
        el.target(
            name="y_X1",
            query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0,
        ),
        el.target(
            name="y_X2",
            query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0,
        ),
        # el.target(
        #     name="log_R2",
        #     query=el.queries.quantiles((.05, .25, .50, .75, .95)),
        #     loss=el.losses.MMD2(kernel="energy"),
        #     weight=1.0
        # )
    ],
    # expert=el.expert.data(dat = expert_dat),
    expert=el.expert.simulator(ground_truth=ground_truth, num_samples=10_000),
    optimizer=el.optimizer(
        optimizer=tf.keras.optimizers.Adam, learning_rate=0.1, clipnorm=1.0
    ),
    trainer=el.trainer(method="parametric_prior", seed=0, epochs=100),
    initializer=el.initializer(
        method="random",
        loss_quantile=0.0,
        iterations=1,
        distribution=el.initialization.uniform(radius=1, mean=0),
    ),
    # network = el.networks.NF(...) # TODO vs. el.normalizing_flow(...)
)

eliobj.fit()
