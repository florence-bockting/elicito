# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting started with a non-parametric joint prior
#
# Here we introduce how to specify the elicitation method for
# a non-parametric joint prior.
#

# %% [markdown]
# ## Imports

# %%
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

# %% [markdown]
# ## The Model
# ### Probabilistic model
# $$
# \begin{align*}
#     (\beta_0, \beta_1, \sigma) &\sim p_\lambda(\cdot) \\
#     \mu &= \beta_0 + \beta_1X \\
#     y_{pred} &\sim \text{Normal}(\mu, \sigma)
# \end{align*}
# $$
#
# ### Implementation
# #### Predictor


# %%
# create a predictor ranging from 1 to 200
# standardize predictor
# select the 25th (X0), 50th (X1), and 75th (X2) quantile of the std.
# predictor for querying the expert
def X_design(N: int, quantiles: list[float]) -> np.ndarray:
    """
    Compute design matrix

    Parameters
    ----------
    N
        number of observations

    quantiles
        list of quantiles

    Returns
    -------
    :
        design matrix
    """
    X = tf.cast(np.arange(N), tf.float32)
    X_std = X / tf.math.reduce_std(X)
    X_sel = tfp.stats.percentile(X_std, quantiles)
    X_design = tf.stack([tf.ones(X_sel.shape), X_sel], -1)
    return X_design


X_design(N=200, quantiles=[25, 50, 75])

# %% [markdown]
# #### Generative model


# %%
class ToyModel:
    """
    generative model
    """

    def __call__(self, prior_samples: Any, design_matrix: Any) -> dict[str, Any]:
        """
        Compute target quantities from generative model

        Parameters
        ----------
        prior_samples
            prior samples

        design_matrix
            design matrix

        Returns
        -------
        :
            dictionary with target quantities
        """
        # linear predictor
        epred = tf.matmul(prior_samples[:, :, :-1], design_matrix, transpose_b=True)

        # data-generating model
        likelihood = tfd.Normal(
            loc=epred, scale=tf.expand_dims(prior_samples[:, :, -1], -1)
        )
        # prior predictive distribution
        ypred = likelihood.sample()

        # selected observations
        y_X0, y_X1, y_X2 = (ypred[:, :, 0], ypred[:, :, 1], ypred[:, :, 2])

        return dict(y_X0=y_X0, y_X1=y_X1, y_X2=y_X2, ypred=ypred, epred=epred)


# %% [markdown]
# ### Generative model

# %%
# specify the model
model = el.model(obj=ToyModel, design_matrix=X_design(N=30, quantiles=[25, 50, 75]))

# %% [markdown]
# ### Model parameters
# + intercept with normal prior $\beta_0$
# + slope with normal prior $\beta_1$
# + random noise with halfnormal prior $\sigma$
#
# **To be learned hyperparameters**
# $\lambda$ refer to weights of deep generative model to learn joint
# prior density function.
#
# + random noise parameter ($\sigma$) is constrained to be positive

# %%
parameters = [
    el.parameter(name="beta0"),
    el.parameter(name="beta1"),
    el.parameter(name="sigma", lower=0),
]


# %% [markdown]
# ### Target quantities and elicitation techniques
# **Target quantities**
# + query expert regarding
#     + **prior predictions** $y \mid X_{i}$ with $i$
# being the 25th, 50th, and 75th quantile of the predictor.
#     + **expected** $R^2$
#
# **Elicitation technique**
# + query each target quantity using **quantile-based elicitation** with
# $Q_p(y \mid X)$ for $p=25, 50, 75$
#
# **Specifying discrepancy measure and weight for single loss components**
# + prior predictions $y \mid X_i$:
#     + discrepancy measure: Maximum Mean Discrepancy with energy kernel
#     + weight of loss component: 1.0
# + $R^2$:
#     + discrepancy measure: Maximum Mean Discrepancy with energy kernel
#     + weight of loss component: 10.0
# + correlation between model parameters (assumed to be independent, thus zero)
#     + L2 loss
#     + weight of loss component: 0.1


# %%
def custom_r2(ypred, epred):
    """Compute coefficient of determination"""
    var_epred = tf.math.reduce_variance(epred, -1)
    # variance of difference between ypred and epred
    var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
    var_total = var_epred + var_diff
    # variance of linear predictor divided by total variance
    return tf.divide(var_epred, var_total)


targets = [
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
    el.target(
        name="R2",
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=10.0,
        target_method=custom_r2,
    ),
    el.target(
        name="cor",
        query=el.queries.correlation(),
        loss=el.losses.L2,
        weight=0.1,
    ),
]

# %% [markdown]
# ### Expert elicitation
#
# Instead of querying a "real" expert, we define a ground truth (i.e., oracle)
# and simulate the oracle-elicited statistics

# %%
# specify ground truth
ground_truth = {
    "beta0": tfd.Normal(loc=5, scale=1),
    "beta1": tfd.Normal(loc=2, scale=1),
    "sigma": tfd.HalfNormal(scale=7.0),
}

# define oracle
expert = el.expert.simulator(ground_truth=ground_truth, num_samples=10_000)
# %% [markdown]
# ### Normalizing Flow as deep generative model

# %%
network = el.networks.NF(
    inference_network=el.networks.InvertibleNetwork,
    network_specs=dict(
        num_params=3,
        num_coupling_layers=3,
        coupling_design="affine",
        coupling_settings={
            "dropout": False,
            "dense_args": {
                "units": 128,
                "activation": "relu",
                "kernel_regularizer": None,
            },
            "num_dense": 2,
        },
        permutation="fixed",
    ),
    base_distribution=el.networks.base_normal,
)

# %% [markdown]
# ### Training
# Learn prior distributions based on expert data

# %%
eliobj = el.Elicit(
    model=model,
    parameters=parameters,
    targets=targets,
    expert=expert,
    optimizer=el.optimizer(
        optimizer=tf.keras.optimizers.Adam, learning_rate=0.0001, clipnorm=1.0
    ),
    trainer=el.trainer(method="deep_prior", seed=2025, epochs=900, progress=0),
    initializer=None,
    network=network,
)

# %% [markdown]
# **Fit eliobj**

# %%
eliobj.fit()

# %% [markdown]
# ## Results

# %% [markdown]
# ### Convergence - Loss

# %%
el.plots.loss(eliobj, figsize=(6, 2))

# %% [markdown]
# ### Convergence - hyperparameters

# %%
el.plots.hyperparameter(eliobj, figsize=(6, 2))

# %% [markdown]
# ### Expert expectations

# %%
el.plots.elicits(eliobj, cols=5, figsize=(8, 2))

# %% [markdown]
# ### Learned prior distributions

# %%
el.plots.prior_marginals(eliobj, cols=3, figsize=(7, 2))

# %%
el.plots.prior_joint(eliobj)
