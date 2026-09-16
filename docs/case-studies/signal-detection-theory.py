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
# # Signal detection theory
#
# This case study translates the models of chapter 11 of Lee & Wagenmakers
# (2013) with the PyMC adapter. The PyMC3 models come from the notebook
# [SignalDetectionTheory.ipynb](https://github.com/pymc-devs/pymc-resources/blob/main/BCM/CaseStudies/SignalDetectionTheory.ipynb)
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

SEED = tf.constant([2013, 11], dtype=tf.int32)
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
# A rate with vague priors is mostly near 0 or near 1. Its median then jumps
# between the two modes from one sample to the next, so the table does not
# show it.
#
# PyMC computes in float64, the translated model in float32. A rate near 0 or
# 1 then differs: `phi(-5)` is exactly 0 in float32, but about 3e-7 in
# float64. The check therefore rounds both samples to 6 decimals, in float64.


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


def phi(x: Any) -> Any:
    """Cumulative distribution function of the standard normal distribution"""
    return 0.5 + 0.5 * pm.math.erf(x / np.sqrt(2))


# %% [markdown]
# ## 11.1 Signal detection theory
#
# The demo data have three cases. Each row holds the hits, false alarms,
# misses and correct rejections of one case.

# %%
data = np.array([70, 50, 30, 50, 7, 5, 3, 5, 10, 0, 0, 10]).reshape(3, -1)
h, f, mi, cr = data.T
s, n = h + mi, f + cr

# %% [markdown]
# PyMC3 gives the Normal priors a precision `tau`. The port gives the
# standard deviation `sigma = 1 / sqrt(tau)`.

# %%
with pm.Model() as model1:
    di = pm.Normal(
        "Discriminability",
        mu=pm.Data("d_mu", 0.0),
        sigma=pm.Data("d_sigma", np.sqrt(2.0)),
        shape=3,
    )
    ci = pm.Normal(
        "Bias", mu=pm.Data("c_mu", 0.0), sigma=pm.Data("c_sigma", np.sqrt(0.5)), shape=3
    )

    thetah = pm.Deterministic("Hit Rate", phi(di / 2 - ci))
    thetaf = pm.Deterministic("False Alarm Rate", phi(-di / 2 - ci))

    pm.Binomial("hi", p=thetah, n=s, observed=h)
    pm.Binomial("fi", p=thetaf, n=n, observed=f)

compare(model1, ["Hit Rate", "False Alarm Rate", "hi", "fi"])

# %% [markdown]
# ## 11.2 Hierarchical signal detection theory
#
# The data of Heit & Rotello (2005) hold an induction and a deduction
# condition. Each row is one participant. The translated model and `pm.draw`
# use the data only for the numbers of trials. Both conditions have 4 signal
# and 4 noise trials for each participant, so the check uses the induction
# condition only.

# %%
std_i = np.loadtxt("data/heit_rotello_std_i.csv", delimiter=",", skiprows=1)


def counts(table: Any) -> tuple[Any, ...]:
    """Return the hits, false alarms, signal trials and noise trials"""
    h, f, mi, cr = table[:, 1:].astype(int).T
    return h, f, h + mi, f + cr


# %% [markdown]
# The group means `mud` and `muc` and the precisions `lambdad` and `lambdac`
# are priors. The participant effects `di` and `ci` have no `pm.Data` input,
# so the translated model samples them.
#
# The book gives the precisions a `Gamma(0.001, 0.001)` prior. Most of its
# draws underflow to 0 in float32, and `1 / sqrt(0)` is infinite. The check
# therefore sets `Gamma(2, 2)` with `pm.set_data`. The translation does not
# depend on these values.

# %%
VAGUE_PRIORS = {
    "mud_mu": 0.0,
    "mud_sigma": 1 / np.sqrt(0.001),
    "muc_mu": 0.0,
    "muc_sigma": 1 / np.sqrt(0.001),
    "lambdad_alpha": 0.001,
    "lambdad_beta": 0.001,
    "lambdac_alpha": 0.001,
    "lambdac_beta": 0.001,
}
CHECK_PRIORS = {
    "lambdad_alpha": 2.0,
    "lambdad_beta": 2.0,
    "lambdac_alpha": 2.0,
    "lambdac_beta": 2.0,
}


def group_priors() -> tuple[Any, ...]:
    """Add the priors of the group means and precisions to the model"""
    data = {name: pm.Data(name, value) for name, value in VAGUE_PRIORS.items()}
    mud = pm.Normal("mud", mu=data["mud_mu"], sigma=data["mud_sigma"])
    muc = pm.Normal("muc", mu=data["muc_mu"], sigma=data["muc_sigma"])
    lambdad = pm.Gamma(
        "lambdad", alpha=data["lambdad_alpha"], beta=data["lambdad_beta"]
    )
    lambdac = pm.Gamma(
        "lambdac", alpha=data["lambdac_alpha"], beta=data["lambdac_beta"]
    )
    return mud, muc, lambdad, lambdac


def likelihood(di: Any, ci: Any, table: Any) -> None:
    """Add the hit and false alarm rates and the counts to the model"""
    h, f, s, n = counts(table)
    thetah = pm.Deterministic("Hit Rate", phi(di / 2 - ci))
    thetaf = pm.Deterministic("False Alarm Rate", phi(-di / 2 - ci))
    pm.Binomial("hi", p=thetah, n=s, observed=h)
    pm.Binomial("fi", p=thetaf, n=n, observed=f)


def model2(table: Any) -> Any:
    """Hierarchical signal detection theory"""
    k = len(table)
    with pm.Model() as model:
        mud, muc, lambdad, lambdac = group_priors()
        di = pm.Normal("di", mu=mud, sigma=1 / pm.math.sqrt(lambdad), shape=k)
        ci = pm.Normal("ci", mu=muc, sigma=1 / pm.math.sqrt(lambdac), shape=k)
        likelihood(di, ci, table)
    pm.set_data(CHECK_PRIORS, model=model)
    return model


OUTPUTS = ["Hit Rate", "False Alarm Rate", "hi", "fi"]

# %%
compare(model2(std_i), OUTPUTS)

# %% [markdown]
# ## 11.3 Parameter expansion
#
# The book scales the participant effects with the priors `xid` and `xic`.


# %%
def model3(table: Any) -> Any:
    """Hierarchical signal detection theory with parameter expansion"""
    k = len(table)
    with pm.Model() as model:
        mud, muc, lambdad, lambdac = group_priors()
        xid = pm.Uniform(
            "xid", lower=pm.Data("xid_lower", 0.0), upper=pm.Data("xid_upper", 1.0)
        )
        xic = pm.Uniform(
            "xic", lower=pm.Data("xic_lower", 0.0), upper=pm.Data("xic_upper", 1.0)
        )
        deltadi = pm.Normal("deltadi", mu=0.0, sigma=1 / pm.math.sqrt(lambdad), shape=k)
        deltaci = pm.Normal("deltaci", mu=0.0, sigma=1 / pm.math.sqrt(lambdac), shape=k)
        pm.Deterministic("sigmad", xid / pm.math.sqrt(lambdad))
        pm.Deterministic("sigmac", xic / pm.math.sqrt(lambdac))
        di = pm.Deterministic("di", mud + xid * deltadi)
        ci = pm.Deterministic("ci", muc + xic * deltaci)
        likelihood(di, ci, table)
    pm.set_data(CHECK_PRIORS, model=model)
    return model


# %%
compare(model3(std_i), OUTPUTS)

# %% [markdown]
# ## Non-centered parametrization
#
# The upstream notebook adds the usual non-centered parametrization: each
# participant effect is the group mean plus a standard Normal draw times the
# group standard deviation.


# %%
def model4(table: Any) -> Any:
    """Hierarchical signal detection theory, non-centered"""
    k = len(table)
    with pm.Model() as model:
        mud, muc, lambdad, lambdac = group_priors()
        deltadi = pm.Normal("deltadi", mu=0.0, sigma=1.0, shape=k)
        deltaci = pm.Normal("deltaci", mu=0.0, sigma=1.0, shape=k)
        pm.Deterministic("sigmad", 1 / pm.math.sqrt(lambdad))
        pm.Deterministic("sigmac", 1 / pm.math.sqrt(lambdac))
        di = pm.Deterministic("di", mud + deltadi / pm.math.sqrt(lambdad))
        ci = pm.Deterministic("ci", muc + deltaci / pm.math.sqrt(lambdac))
        likelihood(di, ci, table)
    pm.set_data(CHECK_PRIORS, model=model)
    return model


# %%
compare(model4(std_i), OUTPUTS)

# %% [markdown]
# ## References
#
# - Heit, E., & Rotello, C. M. (2005). Are there two kinds of reasoning? In
#   *Proceedings of the 27th Annual Meeting of the Cognitive Science Society*
#   (pp. 923-928).
# - Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
#   Practical Course*. Cambridge University Press.
