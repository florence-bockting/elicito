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
# # Heuristic decision-making
#
# This case study translates the models of chapter 18 of Lee & Wagenmakers
# (2013) with the PyMC adapter. The PyMC3 models come from the notebook
# [HeuristicDecisionMaking.ipynb](https://github.com/pymc-devs/pymc-resources/blob/main/BCM/CaseStudies/HeuristicDecisionMaking.ipynb)
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
import pytensor.tensor as pt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import io, stats

from elicito.adapter import pymc as adapter_pymc

SEED = tf.constant([2013, 18], dtype=tf.int32)
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
# samples to 6 decimals, in float64.


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
        tf_values = np.asarray(tf_out[name][0], dtype=float).reshape(DRAWS, -1)
        pm_values = np.asarray(pm_draws, dtype=float).reshape(DRAWS, -1)
        # round in float64: float32 cannot hold a value such as 0.873412
        tf_values, pm_values = tf_values.round(6), pm_values.round(6)
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
# ## Data
#
# In each of 30 questions, 20 participants choose which of two cities is
# larger. `m` holds 9 binary cues for each of 83 cities, `p` the two cities
# of each question, `v` the validity of each cue and `x` its weight. `y` is
# 1 if a participant chooses the first city.

# %%
matdata = io.loadmat("data/StopSearchData.mat")
y = np.squeeze(matdata["y"])
m = np.squeeze(np.float32(matdata["m"]))
p = np.squeeze(matdata["p"])
v = np.squeeze(np.float32(matdata["v"]))
x = np.squeeze(np.float32(matdata["x"]))
n, nc = m.shape  # stimuli and cues
nq = p.shape[0]  # questions
ns = y.shape[0]  # participants

# %% [markdown]
# `qccmat` holds the cue differences of each question. The take-the-best
# (TTB) heuristic decides with the most valid cue that differs; the weighted
# additive (WADD) heuristic sums the weighted cues. `tmat` and `wmat` code the
# decision: 0 for the second city, 1 for a guess, 2 for the first city.

# %%
qcc = m[p[:, 0] - 1] - m[p[:, 1] - 1]
qccmat = np.tile(qcc[None, :, :], (ns, 1, 1))

s = np.argsort(v)
ttmp = np.sum(qccmat * np.power(2, s), axis=2)
tmat = -1 * (-ttmp > 0) + (ttmp > 0) + 1

wtmp = np.sum(qccmat * x, axis=2)
wmat = -1 * (-wtmp > 0) + (wtmp > 0) + 1


def uniform_prior(name: str, lower: float, upper: float) -> Any:
    """Add a Uniform prior with pm.Data bounds to the model"""
    return pm.Uniform(
        name,
        lower=pm.Data(f"{name}_lower", lower),
        upper=pm.Data(f"{name}_upper", upper),
    )


def beta_prior(name: str) -> Any:
    """Add a Beta(1, 1) prior with pm.Data inputs to the model"""
    return pm.Beta(
        name, alpha=pm.Data(f"{name}_a", 1.0), beta=pm.Data(f"{name}_b", 1.0)
    )


def accuracy() -> Any:
    """Add the accuracy gamma; return the choice probability of each code"""
    gamma = uniform_prior("gamma", 0.5, 1.0)
    return pm.math.stack([1 - gamma, 0.5, gamma])


def search_order() -> Any:
    """Add a latent cue order for each participant; return the TTB codes"""
    v1 = pm.HalfNormal("v1", sigma=1.0, shape=ns * nc)
    s1 = pm.Deterministic("s1", pt.argsort(v1.reshape((ns, 1, nc)), axis=2))
    smat2 = pt.tile(s1, (1, nq, 1))
    ttmp = pm.math.sum(qccmat * 2**smat2, axis=2)
    return -1 * (-ttmp > 0) + (ttmp > 0) + 1


def stopping(tmat: Any) -> Any:
    """Add the latent heuristic of each participant; return the codes"""
    phi = beta_prior("phi")
    zi = pm.Bernoulli("zi", p=phi, shape=ns)
    zi_ = pm.math.reshape(pm.math.repeat(zi, nq), (ns, nq))
    return pm.math.switch(pm.math.eq(zi_, 1), tmat, wmat)


# %% [markdown]
# ## 18.1 Take-the-best
#
# Every participant uses TTB. A participant follows the TTB decision with
# the accuracy `gamma`.

# %%
with pm.Model() as model1:
    gammat = accuracy()
    pm.Bernoulli("yiq", p=gammat[tmat], observed=y)

compare(model1, ["yiq"])

# %% [markdown]
# ## 18.2 Stopping
#
# Each participant uses TTB (`zi = 1`) or WADD (`zi = 0`). The indicators
# `zi` have no `pm.Data` input, so the translated model samples them.

# %%
with pm.Model() as model2:
    phi_codes = stopping(tmat)
    gammat = accuracy()
    pm.Bernoulli("yiq", p=gammat[phi_codes], observed=y)

compare(model2, ["yiq"])

# %% [markdown]
# ## 18.3 Searching
#
# Each participant searches the cues in an own order `s1`, the argsort of a
# latent HalfNormal vector. The translated model samples the vector and
# sorts it.

# %%
with pm.Model() as model3:
    gammat = accuracy()
    pm.Bernoulli("yiq", p=gammat[search_order()], observed=y)

compare(model3, ["s1", "yiq"])

# %% [markdown]
# ## 18.4 Searching and stopping

# %%
with pm.Model() as model4:
    codes = stopping(search_order())
    gammat = accuracy()
    pm.Bernoulli("yiq", p=gammat[codes], observed=y)

compare(model4, ["s1", "yiq"])

# %% [markdown]
# ## References
#
# - Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
#   Practical Course*. Cambridge University Press.
