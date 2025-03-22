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
# # Using discrete random variables as likelihood
#
# Here we introduce how to specify the generative model
# when a discrete likelihood is used.

# %% [markdown]
# ## Imports

# %%
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

# %% [markdown]
# ## Gradients for discrete RVs
# ### Challenge
# + ToDo (describe the problem of computing gradients for discrete RVs)
# + possible solution: softmax-gumble trick
#
# ### Possible work around: Softmax-Gumble Trick
# + Describe the softmax gumble trick
#
# ## Example: Binomial model
#
# $$
# \begin{align*}
#     \beta_0 &\sim \text{Normal}(\mu_0, \sigma_0) &\text{priors}\\
#     \beta_1 &\sim \text{Normal}(\mu_1, \sigma_1) &\\
#     \mu &= \text{sigmoid}(\beta_0 + \beta_1X) &\text{link+linear predictor} \\
#     y_i &\sim \text{Binomial}(\mu, N) &\text{likelihood}
# \end{align*}
# $$
#
# + using the `el.utils.softmax_gumble_trick()` function in the generative model

# %%
help(el.utils.softmax_gumbel_trick)

# %% [markdown]
# ### The generative model


# %%
class ToyModel:
    """
    generative model with discrete likelihood
    """

    def __call__(
        self, prior_samples: Any, design_matrix: Any, total_count: int, temp: float
    ) -> dict[str, Any]:
        """
        Sample from the generative model

        Parameters
        ----------
        prior_samples
            samples from the prior distribution

        design_matrix
            design matrix

        total_count
            number of trials per draw from the Binomial

        temp
            temperature parameter for softmax gumbel trick

        Returns
        -------
        :
            dictionary with target quantities
        """
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
        likelihood = tfd.Binomial(
            total_count=total_count, probs=tf.math.sigmoid(tf.expand_dims(epred, -1))
        )

        # prior predictive distribution
        ypred = el.utils.softmax_gumbel_trick(likelihood, total_count, temp)

        # selected observations
        y_X0, y_X1, y_X2 = (ypred[:, :, 0], ypred[:, :, 1], ypred[:, :, 2])

        return dict(y_X0=y_X0, y_X1=y_X1, y_X2=y_X2)


# %% [markdown]

# ### Construct the predictor


# %%
# numeric, standardized predictor
def std_predictor(N: int, quantiles: list[float]) -> np.ndarray:
    """
    Compute the design matrix for the numeric predictor

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
    X_std = (X - tf.reduce_mean(X)) / tf.math.reduce_std(X)
    X_sel = tfp.stats.percentile(X_std, [int(p * 100) for p in quantiles])
    return X_sel


design_matrix = std_predictor(N=200, quantiles=[0.25, 0.50, 0.75])
design_matrix.numpy()


# %% [markdown]
# ### Oracle as expert
#
# + we simulate from a ground truth to obtain the expert data
#
# $$
# \begin{align*}
# \beta_0 &\sim \text{Normal}(0.1, 0.4) \\
#     \beta_1 &\sim \text{Normal}(0.2, 0.2)
# \end{align*}
# $$

# %%
ground_truth = {
    "beta0": tfd.Normal(loc=0.1, scale=0.4),
    "beta1": tfd.Normal(loc=0.2, scale=0.2),
}

# %% [markdown]
# ## Parametric approach
# ### Specify the eliobj

# %%
eliobj = el.Elicit(
    model=el.model(obj=ToyModel, design_matrix=design_matrix, total_count=30, temp=1.6),
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
    ],
    targets=[
        el.target(
            name=f"y_X{i}",
            query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0,
        )
        for i in range(3)
    ],
    expert=el.expert.simulator(ground_truth=ground_truth, num_samples=10_000),
    optimizer=el.optimizer(
        optimizer=tf.keras.optimizers.Adam, learning_rate=0.05, clipnorm=1.0
    ),
    trainer=el.trainer(
        method="parametric_prior",
        seed=0,
        epochs=3,  # 00
    ),
    initializer=el.initializer(
        method="sobol",
        loss_quantile=0,
        iterations=3,  # 2,
        distribution=el.initialization.uniform(radius=1.0, mean=0.0),
    ),
)

# %% [markdown]
# ### Train the eliobj

# %%
eliobj.fit()

# %% [markdown]
# ### Results
# #### Convergence - Loss

# %%
el.plots.loss(eliobj, figsize=(6, 2))

# %% [markdown]
# #### Convergence - Hyperparameter

# %%
el.plots.hyperparameter(eliobj, figsize=(6, 2))

# %% [markdown]
# #### Expert data (oracle) vs. simulated data

# %%
el.plots.elicits(eliobj, cols=3, figsize=(6, 2))

# %% [markdown]
# #### Learned prior distributions

# %%
el.plots.prior_marginals(eliobj, figsize=(6, 2))
