# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Share a hyperparameter
#
# Two priors can share one hyperparameter.
# In this guide, the means of two groups share one prior scale.
# The model then has three hyperparameters instead of four.

# %% tags=["hide"]
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# %% [markdown]
# ## Imports

# %%
from typing import Any

import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

# %% [markdown]
# ## The generative model
#
# The outcomes $y_a$ and $y_b$ are normal, with means $\mu_a$ and $\mu_b$
# and standard deviation 1.


# %%
class Model:
    """Normal outcomes of two groups"""

    def __call__(self, prior_samples: Any) -> dict[str, Any]:
        """
        Simulate the outcomes of both groups

        Parameters
        ----------
        prior_samples
            samples from the prior distributions

        Returns
        -------
        :
            dictionary with the model outputs
        """
        ya = tfd.Normal(loc=prior_samples[:, :, 0], scale=1.0).sample()
        yb = tfd.Normal(loc=prior_samples[:, :, 1], scale=1.0).sample()
        return dict(ya=ya, yb=yb)


# %% [markdown]
# ## The shared hyperparameter
#
# Give the hyperparameter the same name in both priors.
# Set `shared=True` in both.

# %%
scale = el.hyper("s", lower=0, shared=True)
parameters = [
    el.parameter("mu_a", tfd.Normal, dict(loc=el.hyper("m_a"), scale=scale)),
    el.parameter("mu_b", tfd.Normal, dict(loc=el.hyper("m_b"), scale=scale)),
]
targets = [
    el.quantiles("ya", probs=(0.25, 0.50, 0.75)),
    el.quantiles("yb", probs=(0.25, 0.50, 0.75)),
]

# %% [markdown]
# ## Training
#
# The answers come from an oracle.
# Its priors are $\mu_a \sim \text{Normal}(5, 2)$ and $\mu_b \sim \text{Normal}(10, 2)$.

# %%
expert = el.expert.simulator(
    ground_truth={"mu_a": tfd.Normal(5.0, 2.0), "mu_b": tfd.Normal(10.0, 2.0)}
)
eliobj = el.Elicit(
    Model(), targets, parameters, seeds=(2025,), epochs=400, verbose=False
)
fit = eliobj.fit(expert)
fit

# %% [markdown]
# The summary lists three hyperparameters.
# The learned values must be near $m_a = 5$, $m_b = 10$ and $s = 2$.

# %%
fit.plot.hyperparameter(cols=3, figsize=(8, 2.4))
