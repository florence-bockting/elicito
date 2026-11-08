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
# # Use a discrete likelihood
#
# This guide shows how to learn a prior for a model with a discrete
# likelihood, such as a Binomial.
# With the default optimizer, CMA-ES, the model samples the likelihood directly.
# If you use a gradient optimizer, such as Adam, the model needs a smooth
# version of the samples.

# %% tags=["hide"]
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# %% [markdown]
# ## Imports

# %%
from functools import partial
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

# %% [markdown]
# ## The model
#
# This guide uses a logistic regression with a Binomial likelihood:
#
# $$
# \begin{align*}
#     \beta_0 &\sim \text{Normal}(\mu_0, \sigma_0) \\
#     \beta_1 &\sim \text{Normal}(\mu_1, \sigma_1) \\
#     p &= \text{sigmoid}(\beta_0 + \beta_1 X) \\
#     y &\sim \text{Binomial}(N, p)
# \end{align*}
# $$
#
# The argument `temp` selects the sampling.
# With `temp=None`, the model samples the Binomial directly.
# With a number, it uses `el.utils.gumbel_softmax_trick`, which gives
# samples with a gradient.


# %%
class Model:
    """Logistic regression with a Binomial likelihood"""

    def __call__(
        self,
        prior_samples: Any,
        design_matrix: Any,
        total_count: int,
        temp: float | None = None,
    ) -> dict[str, Any]:
        """
        Simulate the outcome at the design points

        Parameters
        ----------
        prior_samples
            samples from the prior distributions

        design_matrix
            values of the predictor

        total_count
            number of trials of the Binomial

        temp
            temperature of the Gumbel-softmax trick; ``None`` samples directly

        Returns
        -------
        :
            dictionary with the model outputs
        """
        # linear predictor, shape (B, num_samples, number of design points)
        epred = prior_samples[:, :, :1] + prior_samples[:, :, 1:2] * design_matrix
        probs = tf.math.sigmoid(epred)
        if temp is None:
            ypred = tfd.Binomial(total_count=total_count, probs=probs).sample()
        else:
            likelihood = tfd.Binomial(total_count=total_count, probs=probs[..., None])
            ypred = el.utils.gumbel_softmax_trick(likelihood, total_count, temp)
        return dict(y_X0=ypred[:, :, 0], y_X1=ypred[:, :, 1], y_X2=ypred[:, :, 2])


# %%
X = tf.cast(np.arange(200), tf.float32)
X = (X - tf.reduce_mean(X)) / tf.math.reduce_std(X)
design_matrix = tfp.stats.percentile(X, [25, 50, 75])

parameters = [
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
]
targets = [
    el.quantiles(f"y_X{i}", probs=(0.05, 0.25, 0.50, 0.75, 0.95)) for i in range(3)
]

# %% [markdown]
# The answers come from an oracle with
# $\beta_0 \sim \text{Normal}(0.1, 0.4)$ and
# $\beta_1 \sim \text{Normal}(0.2, 0.2)$.

# %%
expert = el.expert.simulator(
    ground_truth={
        "beta0": tfd.Normal(loc=0.1, scale=0.4),
        "beta1": tfd.Normal(loc=0.2, scale=0.2),
    },
    num_samples=10_000,
)

# %% [markdown]
# ## Fit with CMA-ES
#
# Sample the likelihood directly: keep `temp=None`.

# %%
model = partial(Model(), design_matrix=design_matrix, total_count=30)

eliobj = el.Elicit(model, targets, parameters, epochs=300, verbose=False)
fit = eliobj.fit(expert)
fit

# %% [markdown]
# Check that the learned priors reproduce the answers.

# %%
fit.plot.elicits(cols=3, figsize=(6, 2))

# %% [markdown]
# ## Fit with a gradient optimizer
#
# A gradient cannot pass through a discrete sample.
# If you use Adam, set `temp` to use the Gumbel-softmax trick
# ([Jang et al., 2016](https://doi.org/10.48550/arXiv.1611.01144)).
# A small `temp` gives samples near the integers, but noisy gradients.
#
# Adam also needs an initializer.
# [Choose an initialization method][choose-an-initialization-method]
# helps you to pick one.

# %%
model_smooth = partial(Model(), design_matrix=design_matrix, total_count=30, temp=1.6)

eliobj_adam = el.Elicit(
    model_smooth,
    targets,
    parameters,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05, clipnorm=1.0),
    initializer=el.initializer(
        method="sobol",
        iterations=32,
        distribution=el.initializers.uniform(radius=1.0, mean=0.0),
    ),
    epochs=300,
    verbose=False,
)
fit_adam = eliobj_adam.fit(expert)
fit_adam

# %% [markdown]
# The oracle simulates its answers with the same model.
# Here the answers are therefore smooth too, and they differ from the CMA-ES fit.
# The answers of a real expert stay the same for both fits.

# %%
fit_adam.plot.elicits(cols=3, figsize=(6, 2))
