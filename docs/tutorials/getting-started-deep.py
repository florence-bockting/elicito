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
# This tutorial learns a **non-parametric joint prior** from expert knowledge.
# The expert does not state the prior. The expert states what they expect to
# observe, and `elicito` finds the prior that reproduces these expectations.
#
# A parametric prior gives each model parameter a prior family. A
# non-parametric joint prior has no prior family. A normalizing flow learns
# the joint distribution of all model parameters.
#
# **You learn how to:**
#
# 1. write a generative model,
# 2. declare the model parameters without a prior family,
# 3. define the target quantities that the expert is asked about,
# 4. set up the normalizing flow,
# 5. learn the weights of the flow with gradient descent,
# 6. check the result with the diagnostic plots.

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
# ## The model
#
# ### Probabilistic model
#
# We use a linear regression with one predictor $X$. The prior $p_\lambda$
# is a joint distribution of the three model parameters:
#
# $$
# \begin{align*}
#     (\beta_0, \beta_1, \sigma) &\sim p_\lambda(\cdot) \\
#     \mu &= \beta_0 + \beta_1 X \\
#     y_{pred} &\sim \text{Normal}(\mu, \sigma)
# \end{align*}
# $$
#
# ### Design matrix
#
# The predictor takes the values $0, 1, \dots, N-1$. We divide it by its
# standard deviation. We then ask the expert about the outcome at three
# values of the predictor: its 25th ($X_0$), 50th ($X_1$) and 75th ($X_2$)
# percentile.


# %%
def X_design(N: int, quantiles: list[float]) -> Any:
    """
    Compute the design matrix

    Parameters
    ----------
    N
        number of observations

    quantiles
        percentiles of the scaled predictor, from 0 to 100

    Returns
    -------
    :
        design matrix with one row per percentile
    """
    X = tf.cast(np.arange(N), tf.float32)
    X_scaled = X / tf.math.reduce_std(X)
    X_sel = tfp.stats.percentile(X_scaled, quantiles)
    design = tf.stack([tf.ones(X_sel.shape), X_sel], -1)
    return design


X_design(N=30, quantiles=[25, 50, 75])

# %% [markdown]
# ### Generative model
#
# The generative model is a class with a `__call__` method. It receives the
# prior samples and the design matrix. It returns a dictionary with every
# quantity that a target can use.
#
# The prior samples have the shape `(B, num_samples, number of parameters)`.
# The last column is $\sigma$. The other columns are the regression
# coefficients.


# %%
class ToyModel:
    """
    Generative model of the linear regression
    """

    def __call__(self, prior_samples: Any, design_matrix: Any) -> dict[str, Any]:
        """
        Compute the target quantities from the generative model

        Parameters
        ----------
        prior_samples
            prior samples

        design_matrix
            design matrix

        Returns
        -------
        :
            dictionary with the target quantities
        """
        # linear predictor
        epred = tf.matmul(prior_samples[:, :, :-1], design_matrix, transpose_b=True)

        # data-generating model
        likelihood = tfd.Normal(
            loc=epred, scale=tf.expand_dims(prior_samples[:, :, -1], -1)
        )
        # prior predictive distribution
        ypred = likelihood.sample()

        # outcome at the three selected predictor values
        y_X0, y_X1, y_X2 = (ypred[:, :, 0], ypred[:, :, 1], ypred[:, :, 2])

        return dict(y_X0=y_X0, y_X1=y_X1, y_X2=y_X2, ypred=ypred, epred=epred)


# %%
model = el.model(obj=ToyModel, design_matrix=X_design(N=30, quantiles=[25, 50, 75]))

# %% [markdown]
# ## Model parameters
#
# Each model parameter gets a name, but no prior family. The values that
# `elicito` learns are the weights $\lambda$ of the normalizing flow.
#
# | parameter | constraint |
# | :-- | :-- |
# | intercept $\beta_0$ | none |
# | slope $\beta_1$ | none |
# | noise $\sigma$ | $\sigma > 0$ |
#
# `lower=0` constrains $\sigma$ to positive values.

# %%
parameters = [
    el.parameter(name="beta0"),
    el.parameter(name="beta1"),
    el.parameter(name="sigma", lower=0),
]


# %% [markdown]
# ## Target quantities and elicitation techniques
#
# A **target quantity** is a quantity that the expert can judge. The
# **elicitation technique** is the question that we ask the expert about it.
#
# | target quantity | elicitation technique | discrepancy measure | weight |
# | :-- | :-- | :-- | --: |
# | $y \mid X_0$, $y \mid X_1$, $y \mid X_2$ | quantiles $Q_p$, $p = 5, 25, 50, 75, 95$ | MMD² with energy kernel | 1.0 |
# | $R^2$ | quantiles $Q_p$, $p = 5, 25, 50, 75, 95$ | MMD² with energy kernel | 10.0 |
# | model parameters | correlation | L2 loss | 0.1 |
#
# The model does not return $R^2$. The function `custom_r2` computes it from
# the model output. `elicito` matches its arguments to the keys of the model
# output by name.
#
# The expert assumes that the model parameters are independent. The
# correlation target therefore pulls each correlation to zero.


# %%
def custom_r2(ypred: Any, epred: Any) -> Any:
    """Compute the coefficient of determination"""
    var_epred = tf.math.reduce_variance(epred, -1)
    # variance of the difference between ypred and epred
    var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
    var_total = var_epred + var_diff
    # variance of the linear predictor divided by the total variance
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
# ## Expert elicitation
#
# We do not ask a real expert. We define a ground truth, the **oracle**, and
# simulate the answers that the oracle gives. Because we know the true
# prior, we can later see whether `elicito` recovers it.

# %%
ground_truth = {
    "beta0": tfd.Normal(loc=5, scale=1),
    "beta1": tfd.Normal(loc=2, scale=1),
    "sigma": tfd.HalfNormal(scale=7.0),
}

expert = el.expert.simulator(ground_truth=ground_truth, num_samples=10_000)

# %% [markdown]
# ## Normalizing flow
#
# The normalizing flow transforms samples from a standard normal base
# distribution into samples from the joint prior. It has one output for each
# model parameter, so `num_params=3`.

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
# ## Training
#
# We learn the weights of the flow with the **Adam** gradient descent.
#
# - `method="deep_prior"` tells the trainer to learn a normalizing flow.
# - `network` passes the flow to `elicito`.
# - `initializer=None`: the flow starts from its own random weights.
#
# > **Note:** CMA-ES does not support a deep prior. The flow has too many
# > weights for a search without a gradient.

# %%
eliobj = el.Elicit(
    model=model,
    parameters=parameters,
    targets=targets,
    expert=expert,
    optimizer=el.optimizer(
        optimizer=tf.keras.optimizers.Adam, learning_rate=0.0001, clipnorm=1.0
    ),
    trainer=el.trainer(method="deep_prior", seed=2025, epochs=500, progress=1),
    initializer=None,
    network=network,
)

# %% [markdown]
# Print the `eliobj` to see a summary of the settings.

# %%
eliobj

# %%
eliobj.fit()

# %% [markdown]
# The results keep the learned weights, the loss and the expert data.
# `sample()` computes the prior samples and the model simulations from them.
# We compute them once and give them to every plot.

# %%
eliobj.results

# %%
samples = eliobj.sample()

# %% [markdown]
# ## Results
#
# ### Convergence of the loss
#
# The loss must decrease and then stay flat.

# %%
el.plots.loss(eliobj, figsize=(7, 3));

# %% [markdown]
# ### Convergence of the prior marginals
#
# A deep prior has no hyperparameter to plot. The plot shows the mean and
# the standard deviation of each prior marginal. Each must settle on one
# value.

# %%
el.plots.marginals(eliobj, figsize=(8.5, 3.5));

# %% [markdown]
# ### Expert expectations
#
# Each point compares one elicited quantile of the expert with the same
# quantile of the learned model. The points lie on the diagonal when the
# learned prior reproduces the expert data.

# %%
el.plots.elicits(eliobj, cols=5, figsize=(11, 2.4), samples=samples);

# %% [markdown]
# ### Learned prior distributions

# %%
el.plots.prior_marginals(eliobj, cols=3, figsize=(8, 2.4), samples=samples);

# %% [markdown]
# ### Learned joint prior
#
# The joint prior shows the dependence between the model parameters. The
# expert assumes independence, so the pairs must show no correlation.

# %%
el.plots.prior_joint(eliobj, samples=samples);
