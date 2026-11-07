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
# # The SIMPLE model of memory
#
# This case study translates the models of chapter 15 of Lee & Wagenmakers
# (2013) with the PyMC adapter. The PyMC3 models come from the notebook
# [TheSIMPLEModelofMemory.ipynb](https://github.com/pymc-devs/pymc-resources/blob/main/BCM/CaseStudies/TheSIMPLEModelofMemory.ipynb)
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

# %% tags=["remove_input"]
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# mkdocs runs the notebook in the repository root; the data are next to it
if os.path.isdir("docs/case-studies"):
    os.chdir("docs/case-studies")

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

SEED = tf.constant([2013, 15], dtype=tf.int32)
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
# The data of Murdock (1962) come from six free recall conditions. Each row
# of `y` counts how often each serial position is recalled; a row ends with
# zeros after the list length. `m` holds the time since the study of each
# position, in seconds.

# %%
y = np.loadtxt("data/k_M.txt", delimiter=",")
n = np.array([1440, 1280, 1520, 1520, 1200, 1280])
listlength = np.array([10, 15, 20, 20, 30, 40])
lagall = np.array([2, 2, 2, 1, 1, 1])
offset = np.array([15, 20, 25, 10, 15, 20])
dsets, nitem = y.shape

m = np.zeros(y.shape)
for i in range(dsets):
    m[i, : listlength[i]] = offset[i] + np.arange(
        listlength[i] * lagall[i], 0, -lagall[i]
    )


def uniform_prior(name: str, lower: float, upper: float, shape: Any = None) -> Any:
    """Add a Uniform prior with pm.Data bounds to the model"""
    return pm.Uniform(
        name,
        lower=pm.Data(f"{name}_lower", lower),
        upper=pm.Data(f"{name}_upper", upper),
        shape=shape,
    )


# %% [markdown]
# The upstream notebook writes each model twice: vectorized over the six
# conditions, and with a loop over them. The two versions differ: the
# vectorized one computes `theta = 1 - prod(1 - resp)`, the loop one
# `theta = clip(sum(resp), 0, 0.999)`. The port keeps both.
#
# The vectorized version pads the positions after the list length with
# `m = 1e-5`, and observes only the recalled positions.

# %%
m_pad = np.where(m == 0, 1e-5, m)
nmat = np.repeat(n[:, None], nitem, axis=1)
mmat1 = np.repeat(m_pad[:, :, None], nitem, axis=2)
mmat2 = np.transpose(mmat1, (0, 2, 1))
mask = np.where(y > 0)


def theta_vectorized(cx: Any, sx: Any, tx: Any) -> Any:
    """Return the recall probability of each condition and position"""
    # similarities
    eta = pm.math.exp(-cx * pm.math.abs(pm.math.log(mmat1) - pm.math.log(mmat2)))
    etasum = pm.math.reshape(
        pm.math.repeat(pm.math.sum(eta, axis=2), nitem), (dsets, nitem, nitem)
    )
    # discriminabilities
    disc = eta / etasum
    # response probabilities
    resp = 1 / (1 + pm.math.exp(-sx * (disc - tx)))
    # free recall overall response probability
    return 1 - pm.math.prod(1 - resp, axis=2)


def theta_loop(x: int, cx: Any, sx: Any, tx: Any) -> Any:
    """Return the recall probability of each position of condition x"""
    sz = listlength[x]
    m1 = np.array([m[x, :sz]] * sz).T
    m2 = np.array([m[x, :sz]] * sz)
    eta = pm.math.exp(-cx * pm.math.abs(np.log(m1) - np.log(m2)))
    etasum = pm.math.reshape(pm.math.repeat(pm.math.sum(eta, axis=1), sz), (sz, sz))
    disc = eta / etasum
    resp = 1 / (1 + pm.math.exp(-sx * (disc - tx)))
    return pm.math.clip(pm.math.sum(resp, axis=1), 0, 0.999)


# %% [markdown]
# ## 15.1 The SIMPLE model
#
# Each condition has an own distinctiveness `cx`, threshold slope `sx` and
# threshold `tx`. A prior with six elements becomes six parameters, with
# shared hyperparameters.

# %%
with pm.Model() as simple1:
    cx = uniform_prior("cx", 0.0, 100.0, shape=dsets)
    sx = uniform_prior("sx", 0.0, 100.0, shape=dsets)
    tx = uniform_prior("tx", 0.0, 1.0, shape=dsets)
    theta = theta_vectorized(cx[:, None, None], sx[:, None, None], tx[:, None, None])
    pm.Binomial("yobs", p=theta[mask], n=nmat[mask], observed=y[mask])

compare(simple1, ["yobs"])

# %% [markdown]
# The loop version:

# %%
with pm.Model() as simple1_loop:
    cx = uniform_prior("cx", 0.0, 100.0, shape=dsets)
    sx = uniform_prior("sx", 0.0, 100.0, shape=dsets)
    tx = uniform_prior("tx", 0.0, 1.0, shape=dsets)
    for x in range(dsets):
        theta = theta_loop(x, cx[x], sx[x], tx[x])
        pm.Binomial(f"yobs_{x}", p=theta, n=n[x], observed=y[x, : listlength[x]])

compare(simple1_loop, [f"yobs_{x}" for x in range(dsets)])

# %% [markdown]
# ## 15.2 A hierarchical extension of SIMPLE
#
# All conditions share `cx` and `sx`. The threshold depends on the list
# length `W`: `tx = a1 * W + a2`.

# %%
W = listlength

with pm.Model() as simple2:
    cx = uniform_prior("cx", 0.0, 100.0)
    sx = uniform_prior("sx", 0.0, 100.0)
    a1 = uniform_prior("a1", -1.0, 0.0)
    a2 = uniform_prior("a2", 0.0, 1.0)
    tx = pm.Deterministic("tx", a1 * W + a2)
    theta = theta_vectorized(cx, sx, tx[:, None, None])
    pm.Binomial("yobs", p=theta[mask], n=nmat[mask], observed=y[mask])

compare(simple2, ["tx", "yobs"])

# %% [markdown]
# The loop version:

# %%
with pm.Model() as simple2_loop:
    cx = uniform_prior("cx", 0.0, 100.0)
    sx = uniform_prior("sx", 0.0, 100.0)
    a1 = uniform_prior("a1", -1.0, 0.0)
    a2 = uniform_prior("a2", 0.0, 1.0)
    tx = pm.Deterministic("tx", a1 * W + a2)
    for x in range(dsets):
        theta = theta_loop(x, cx, sx, tx[x])
        pm.Binomial(f"yobs_{x}", p=theta, n=n[x], observed=y[x, : listlength[x]])

compare(simple2_loop, ["tx", *[f"yobs_{x}" for x in range(dsets)]])

# %% [markdown]
# ## References
#
# - Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
#   Practical Course*. Cambridge University Press.
# - Murdock, B. B. (1962). The serial position effect of free recall. *Journal
#   of Experimental Psychology*, 64, 482-488.
