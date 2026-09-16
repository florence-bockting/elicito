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
# # The GCM model of categorization
#
# This case study translates the models of chapter 17 of Lee & Wagenmakers
# (2013) with the PyMC adapter. The PyMC3 models come from the notebook
# [TheGCMModelofCategorization.ipynb](https://github.com/pymc-devs/pymc-resources/blob/main/BCM/CaseStudies/TheGCMModelofCategorization.ipynb)
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
from scipy import io, stats

from elicito.adapter import pymc as adapter_pymc

SEED = tf.constant([2013, 17], dtype=tf.int32)
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
# In the experiment of Kruschke (1993), 40 participants learn to sort 8
# stimuli into two categories. `d1` and `d2` hold the distances between the
# stimuli on the two stimulus dimensions. `a` holds the category of each
# stimulus: 1 for A, 2 for B. `x` counts, for each participant and stimulus,
# the category A decisions in 8 trials.

# %%
matdata = io.loadmat("data/KruschkeData.mat")
nstim, nsubj = 8, 40
d1, d2 = matdata["d1"], matdata["d2"]
a = matdata["a"][0]
x = matdata["x"]
y = matdata["y"][:, 0]

# a1[i, j] is 1 if stimulus j is in category A
a1 = np.tile(2 - a, [nstim, 1])
b = 0.5  # the response bias


def uniform_prior(name: str, lower: float, upper: float, shape: Any = None) -> Any:
    """Add a Uniform prior with pm.Data bounds to the model"""
    return pm.Uniform(
        name,
        lower=pm.Data(f"{name}_lower", lower),
        upper=pm.Data(f"{name}_upper", upper),
        shape=shape,
    )


def response_rate(c: Any, w: Any, d1: Any, d2: Any, a1: Any) -> Any:
    """Return the probability of a category A decision for each stimulus"""
    sij = pm.math.exp(-c * (w * d1 + (1 - w) * d2))
    sum_ajsij = pm.math.sum(a1 * sij, axis=1)
    sum_majsij = pm.math.sum((1 - a1) * sij, axis=1)
    return (b * sum_ajsij) / (b * sum_ajsij + (1 - b) * sum_majsij)


# %% [markdown]
# ## 17.1 The GCM model
#
# The generalization `c` and the attention weight `w` are the same for all
# participants. `yi` sums the decisions of all participants, in
# `nstim * nsubj` trials for each stimulus.

# %%
with pm.Model() as model1:
    c = uniform_prior("c", 0.0, 5.0)
    w = uniform_prior("w", 0.0, 1.0)
    ri = pm.Deterministic("ri", response_rate(c, w, d1, d2, a1))
    pm.Binomial("yi", p=ri, n=nstim * nsubj, observed=y)

compare(model1, ["ri", "yi"])

# %% [markdown]
# ## 17.2 Individual differences in the GCM
#
# Each participant has an own `c` and `w`. The last axis of the distances
# and of the priors is the participant. A prior with 40 elements becomes 40
# parameters, with shared hyperparameters.

# %%
d1_t = np.tile(d1[:, :, None], [1, 1, nsubj])
d2_t = np.tile(d2[:, :, None], [1, 1, nsubj])
a1_t = a1[:, :, None]
y2 = x.T

with pm.Model() as model2:
    c = uniform_prior("c", 0.0, 5.0, shape=(1, 1, nsubj))
    w = uniform_prior("w", 0.0, 1.0, shape=(1, 1, nsubj))
    ri = pm.Deterministic("ri", response_rate(c, w, d1_t, d2_t, a1_t))
    pm.Binomial("yi", p=ri, n=nstim, observed=y2)

compare(model2, ["ri", "yi"])

# %% [markdown]
# ## 17.3 Latent groups in the GCM
#
# The participants fall into three latent groups. A contaminant, with
# `zck = 1`, guesses. The other participants attend to the first dimension
# (`zcg = 0`) or to the second (`zcg = 1`). The group indicators and the
# participant parameters have no `pm.Data` input, so the translated model
# samples them.

# %%
with pm.Model() as model3:
    mu1w = uniform_prior("mu1w", 0.0, 1.0)
    delta = uniform_prior("delta", 0.0, 1.0)
    mu2w = pm.Deterministic("mu2w", pm.math.clip(mu1w + delta, 0, 1))

    sigmaw = uniform_prior("sigmaw", 0.01, 1.0)
    muc = uniform_prior("muc", 0.0, 5.0)
    sigmac = uniform_prior("sigmac", 0.01, 3.0)

    phic = uniform_prior("phic", 0.0, 1.0)
    phig = uniform_prior("phig", 0.0, 1.0)

    zck = pm.Bernoulli("zck", p=phic, shape=nsubj)
    zcg = pm.Bernoulli("zcg", p=phig, shape=nsubj)

    c = pm.math.clip(pm.Normal("c", mu=muc, sigma=sigmac, shape=(1, 1, nsubj)), 0, 5)
    muw = pm.Deterministic("muw", pm.math.switch(pm.math.eq(zcg, 0), mu1w, mu2w))
    w = pm.math.clip(pm.Normal("w", mu=muw, sigma=sigmaw, shape=(1, 1, nsubj)), 0, 1)

    ri1 = pm.Deterministic("ri1", response_rate(c, w, d1_t, d2_t, a1_t))
    ri2 = np.full((nstim, nsubj), 0.5)
    ri = pm.Deterministic("ri", pm.math.switch(pm.math.eq(zck, 0), ri1, ri2))
    pm.Binomial("yi", p=ri, n=nstim, observed=y2)

compare(model3, ["muw", "ri", "yi"])

# %% [markdown]
# ## References
#
# - Kruschke, J. K. (1993). Human category learning: Implications for
#   backpropagation models. *Connection Science*, 5, 3-36.
# - Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
#   Practical Course*. Cambridge University Press.
