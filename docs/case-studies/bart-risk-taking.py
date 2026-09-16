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
# # The BART model of risk taking
#
# This case study translates the models of chapter 16 of Lee & Wagenmakers
# (2013) with the PyMC adapter. The PyMC3 models come from the notebook
# [TheBARTModelofRiskTaking.ipynb](https://github.com/pymc-devs/pymc-resources/blob/main/BCM/CaseStudies/TheBARTModelofRiskTaking.ipynb)
# of pymc-resources (MIT licence). See the [case studies](index.md) for the
# credit.
#
# For each model, the notebook:
#
# 1. ports the model to the current PyMC,
# 2. samples the priors in TensorFlow and runs the translated model,
# 3. draws the same outputs with `pm.draw`,
# 4. compares the two samples.
#
# > **Note:** the adapter needs Python 3.12 or later and the optional
# > dependency `pymc`. Install it with `pip install "elicito[pymc]"`.

# %% tags=["hide"]
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# %% [markdown]
# ## Imports

# %%
from typing import Any

import numpy as np
import pymc as pm
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats

from elicito.adapter import pymc as adapter_pymc

SEED = tf.constant([2013, 16], dtype=tf.int32)
DRAWS = 10_000

# %% [markdown]
# ## The check
#
# The port of a model follows the rules of the how-to guide
# ["Use a PyMC model"](../how-to-guides/use-pymc-model.py): each input of a
# prior is a named `pm.Data` node. `adapter_pymc.parameters` reads the priors.
# `sample_priors` samples each prior in TensorFlow, with the current values
# of its `pm.Data` nodes.
#
# `compare` runs the translated model on these samples. It then draws the
# same outputs with `pm.draw`. For each element of an output, it computes the
# Kolmogorov-Smirnov distance `D` between the two samples. The table shows
# one row for each output: the element with the largest `D`, with the mean
# and the standard deviation of both samples. The check fails if `D` is 0.05
# or more.
#
# PyMC computes in float64, the translated model in float32. A value near 0
# or 1 then differs in the last digits. The check therefore rounds both
# samples to 6 decimals.


# %%
def sample_priors(pymc_model: Any, draws: int, seed: Any) -> Any:
    """Sample the priors of a PyMC model in TensorFlow, shape (1, draws, P)"""
    parameters = adapter_pymc.parameters(pymc_model)
    seeds = tfp.random.split_seed(seed, n=len(parameters))
    columns = []
    for parameter, param_seed in zip(parameters, seeds):
        args = {
            arg: float(pymc_model[hyper["name"]].get_value())
            for arg, hyper in parameter["hyperparams"].items()
        }
        prior = parameter["family"](**args)
        columns.append(prior.sample(draws, seed=param_seed))
    return tf.stack(columns, axis=-1)[None]


def compare(pymc_model: Any, names: list[str], tol: float = 0.05) -> None:
    """Compare the translated model with pm.draw, for each output in names"""
    seed_priors, seed_model = tfp.random.split_seed(SEED)
    generative_model = adapter_pymc.model(pymc_model)()
    prior_samples = sample_priors(pymc_model, DRAWS, seed_priors)
    tf_out = generative_model(prior_samples, seed=seed_model)
    pm_out = pm.draw([pymc_model[name] for name in names], draws=DRAWS, random_seed=1)

    print(f"{'output':<28}{'TensorFlow mean, sd':>22}{'PyMC mean, sd':>22}{'D':>7}")
    worst = 0.0
    for name, pm_draws in zip(names, pm_out):
        tf_values = np.round(tf_out[name][0], 6).reshape(DRAWS, -1)
        pm_values = np.round(pm_draws, 6).reshape(DRAWS, -1)
        distances = [
            stats.ks_2samp(tf_values[:, j], pm_values[:, j], method="asymp").statistic
            for j in range(tf_values.shape[1])
        ]
        j = int(np.argmax(distances))
        tf_stats, pm_stats = (
            f"{values[:, j].mean():.2f}, {values[:, j].std():.2f}"
            for values in (tf_values, pm_values)
        )
        label = f"{name}[{j}] of {tf_values.shape[1]}"
        print(f"{label:<28}{tf_stats:>22}{pm_stats:>22}{distances[j]:>7.3f}")
        worst = max(worst, distances[j])
    assert worst < tol, f"the largest distance D = {worst:.3f} is not below {tol}"


# %% [markdown]
# ## 16.1 The BART model
#
# In the Balloon Analogue Risk Task (BART), a participant pumps up a balloon.
# Each pump earns money, but the balloon can burst, and the money of the
# trial is then lost. The data hold 90 trials of one participant, George,
# in three conditions: sober, tipsy and drunk.
#
# `responses` turns the trials into one row per pump decision: `d` is 1 if
# George stops and takes the cash, and `k` counts the pumps of the trial.

# %%
BURST = 0.15  # (belief of) the bursting probability


def responses(path: str) -> tuple[Any, Any]:
    """Return the decisions d and the pump numbers k of one condition"""
    pumps, cash = np.loadtxt(path, skiprows=1, usecols=(5, 6)).T
    d: list[int] = []
    k: list[int] = []
    for n_pumps, stops in zip(pumps.astype(int), (cash != 0).astype(int)):
        d += [0] * n_pumps + [1] * stops
        k += list(range(1, n_pumps + stops + 1))
    return np.array(d), np.array(k)


d, k = responses("data/GeorgeSober.txt")

# %% [markdown]
# The priors are `gammap`, the risk propensity, and `beta`, the behavioral
# consistency.

# %%
with pm.Model() as model1:
    gammap = pm.Uniform(
        "gammap",
        lower=pm.Data("gammap_lower", 0.0),
        upper=pm.Data("gammap_upper", 10.0),
    )
    beta = pm.Uniform(
        "beta", lower=pm.Data("beta_lower", 0.0), upper=pm.Data("beta_upper", 10.0)
    )
    omega = pm.Deterministic("omega", -gammap / np.log(1 - BURST))
    thetajk = 1 - pm.math.invlogit(-beta * (k - omega))
    pm.Bernoulli("djk", p=thetajk, observed=d)

compare(model1, ["omega", "djk"])

# %% [markdown]
# ## 16.2 A hierarchical extension of the BART model
#
# The hierarchical model gives each condition its own `gammap` and `beta`.
# The group means and standard deviations are the priors. The condition
# effects have no `pm.Data` input, so the translated model samples them.

# %%
conditions = ["Sober", "Tipsy", "Drunk"]
data = [responses(f"data/George{condition}.txt") for condition in conditions]
dij = np.concatenate([d for d, _ in data])
kij = np.concatenate([k for _, k in data])
cij = np.concatenate([np.full(len(d), c) for c, (d, _) in enumerate(data)])


def uniform_prior(name: str) -> Any:
    """Add a Uniform(0, 10) prior with pm.Data bounds to the model"""
    lower = pm.Data(f"{name}_lower", 0.0)
    upper = pm.Data(f"{name}_upper", 10.0)
    return pm.Uniform(name, lower=lower, upper=upper)


with pm.Model() as model2:
    mu_g, sigma_g = uniform_prior("mu_g"), uniform_prior("sigma_g")
    mu_b, sigma_b = uniform_prior("mu_b"), uniform_prior("sigma_b")

    gammap = pm.Normal("gammap", mu=mu_g, sigma=sigma_g, shape=len(conditions))
    beta = pm.Normal("beta", mu=mu_b, sigma=sigma_b, shape=len(conditions))

    omega = -gammap[cij] / np.log(1 - BURST)
    thetajk = 1 - pm.math.invlogit(-beta[cij] * (kij - omega))
    pm.Bernoulli("djk", p=thetajk, observed=dij)

compare(model2, ["djk"])

# %% [markdown]
# ## References
#
# - Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
#   Practical Course*. Cambridge University Press.
