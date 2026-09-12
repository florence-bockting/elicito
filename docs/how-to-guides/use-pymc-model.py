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
# # Use a PyMC model
#
# Write the generative model once, in PyMC. `elicito` reads the priors from
# the PyMC model, learns their hyperparameters from the expert, and writes the
# learned values back. The same PyMC model then fits the data with the
# elicited priors.
#
# The example is the linear regression of the tutorial
# ["Getting started: parametric prior"](../tutorials/getting-started-param.py).
#
# > **Note:** the adapter needs Python 3.12 or later and the optional
# > dependency `pymc`. Install it with `pip install "elicito[pymc]"`.
# > This guide also uses CMA-ES, which needs `pip install "elicito[cma]"`.

# %% tags=["hide"]
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib


def _warm() -> None:
    import elicito  # noqa: F401


joblib.Parallel(n_jobs=5)(joblib.delayed(_warm)() for _ in range(5))

import tensorflow as tf

tf.constant(0.0)

# %% [markdown]
# ## Imports

# %%
from typing import Any

import arviz as az
import numpy as np
import pymc as pm
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el
from elicito.adapter import pymc as adapter_pymc

tfd = tfp.distributions

# %% [markdown]
# ## The PyMC model
#
# Follow these rules:
#
# - Make each hyperparameter a named `pm.Data` node. Its name becomes the name
#   of the hyperparameter in `elicito`.
# - Use Normal or HalfNormal priors and a Normal likelihood. The adapter
#   supports only these families for now.
# - Make the predictor and the data `pm.Data` nodes too. You can then replace
#   them with `pm.set_data` before the fit.
# - Do not give the likelihood a `shape` argument. It takes its shape from
#   `observed`.
#
# For the elicitation, the predictor holds the three design points of the
# tutorial. The data are placeholders.

# %%
x_scaled = np.arange(30) / np.std(np.arange(30))
x_design = np.percentile(x_scaled, [25, 50, 75])

with pm.Model() as pymc_model:
    x = pm.Data("x", x_design)
    y_obs = pm.Data("y_obs", np.zeros(3))

    mu0, sigma0 = pm.Data("mu0", 0.0), pm.Data("sigma0", 1.0)
    mu1, sigma1 = pm.Data("mu1", 0.0), pm.Data("sigma1", 1.0)
    sigma2 = pm.Data("sigma2", 1.0)

    beta0 = pm.Normal("beta0", mu0, sigma0)
    beta1 = pm.Normal("beta1", mu1, sigma1)
    sigma = pm.HalfNormal("sigma", sigma2)

    epred = pm.Deterministic("epred", beta0 + beta1 * x)
    pm.Normal("ypred", epred, sigma, observed=y_obs)

# %% [markdown]
# ## From PyMC to elicito
#
# `parameters` reads the priors. `model` translates the PyMC model into a
# generative model. The generative model returns every observed variable and
# every `pm.Deterministic`, by name: here `ypred` and `epred`.

# %%
parameters = adapter_pymc.parameters(pymc_model)
model = el.model(obj=adapter_pymc.model(pymc_model))

[(p["name"], p["family"].__name__) for p in parameters]

# %% [markdown]
# ## Target quantities
#
# The targets are those of the tutorial. `ypred` holds the outcome at the
# three design points, so a target function selects one of them. A target
# function receives the model output by the name of its argument.


# %%
def outcome_at(i: int) -> Any:
    """Return a target function that selects design point i of ypred"""

    def target(ypred: Any) -> Any:
        return ypred[:, :, i]

    return target


def custom_r2(ypred: Any, epred: Any) -> Any:
    """Compute the coefficient of determination"""
    var_epred = tf.math.reduce_variance(epred, -1)
    var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
    return tf.divide(var_epred, var_epred + var_diff)


targets = [
    el.target(
        name=f"y_X{i}",
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        target_method=outcome_at(i),
        weight=1.0,
    )
    for i in range(3)
]
targets.append(
    el.target(
        name="R2",
        query=el.queries.quantiles((0.05, 0.50, 0.95)),
        loss=el.losses.L2,
        target_method=custom_r2,
        weight=0.5,
    )
)

# %% [markdown]
# ## Expert and training
#
# A simulated expert answers with the ground truth of the tutorial. CMA-ES
# learns the hyperparameters. Five replications with different seeds show
# whether the result depends on the seed.

# %%
ground_truth = {
    "beta0": tfd.Normal(loc=5, scale=1),
    "beta1": tfd.Normal(loc=2, scale=1),
    "sigma": tfd.HalfNormal(scale=7.0),
}

eliobj = el.Elicit(
    model=model,
    parameters=parameters,
    targets=targets,
    expert=el.expert.simulator(ground_truth=ground_truth, num_samples=10_000),
    optimizer=el.optimizer(optimizer="cmaes"),
    trainer=el.trainer(
        method="parametric_prior", seed=2025, epochs=400, progress=1, kappa=0.0
    ),
)
eliobj.fit(parallel=el.utils.parallel(runs=5, seeds=[1, 2, 3, 4, 5]))

# %% [markdown]
# `hyperparameters(replication=r)` returns the learned values of one
# replication, by name. The true values are
# `mu0=5, sigma0=1, mu1=2, sigma1=1, sigma2=7`.

# %%
for r in range(5):
    values = eliobj.hyperparameters(replication=r)
    print(r, {name: round(value, 2) for name, value in values.items()})

# %% [markdown]
# ## Back to PyMC
#
# Replace the design points with the observed data. Here we simulate 50
# observations.

# %%
n_obs = 50
x_data = np.arange(n_obs) / np.std(np.arange(n_obs))
rng = np.random.default_rng(2025)
y_data = 5.3 + 1.8 * x_data + rng.normal(0.0, 2.0, size=n_obs)

pm.set_data({"x": x_data, "y_obs": y_data}, model=pymc_model)

# %% [markdown]
# `set_hyperparameters` writes the learned values of one replication into the
# `pm.Data` nodes. The priors of the PyMC model are then the elicited priors of
# that replication. For each replication, we draw from the prior and fit the
# model.

# %%
var_names = ["beta0", "beta1", "sigma"]
priors, posteriors = {}, {}
for r in range(5):
    adapter_pymc.set_hyperparameters(pymc_model, eliobj, replication=r)
    key = f"replication {r}"
    prior = pm.sample_prior_predictive(draws=2_000, random_seed=2025, model=pymc_model)
    # the same variable order as in the posterior plot
    prior["prior"] = prior["prior"].to_dataset()[var_names]
    priors[key] = prior
    posteriors[key] = pm.sample(
        draws=500,
        tune=500,
        chains=4,
        random_seed=2026,
        progressbar=False,
        model=pymc_model,
    )

# %% [markdown]
# The five elicited priors.

# %%
pc_prior = az.plot_dist(
    priors,
    group="prior",
    var_names=var_names,
    visuals={"point_estimate_text": False},
)
pc_prior.add_legend("model")

# %% [markdown]
# The posteriors of the five elicited priors.

# %%
pc_posterior = az.plot_dist(
    posteriors,
    var_names=var_names,
    visuals={"point_estimate_text": False},
)
pc_posterior.add_legend("model")
