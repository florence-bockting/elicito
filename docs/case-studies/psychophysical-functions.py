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
# # Psychophysical functions
#
# This case study translates the models of chapter 12 of Lee & Wagenmakers
# (2013) with the PyMC adapter. The PyMC3 models come from the notebook
# [PsychophysicalFunctions.ipynb](https://github.com/pymc-devs/pymc-resources/blob/main/BCM/CaseStudies/PsychophysicalFunctions.ipynb)
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

SEED = tf.constant([2013, 12], dtype=tf.int32)
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


def phi(x: Any) -> Any:
    """Cumulative distribution function of the standard normal distribution"""
    return 0.5 + 0.5 * pm.math.erf(x / np.sqrt(2))


# %% [markdown]
# ## 12.1 Psychophysical functions
#
# Eight participants judge if a test interval is longer than a standard
# interval of 300 ms. For each participant and test interval, the data hold
# the interval in ms, the number of trials and the number of "long"
# responses. A row ends with `NA` if the participant saw fewer intervals.

# %%
x = np.genfromtxt("data/data_x.txt", delimiter="\t")
n = np.genfromtxt("data/data_n.txt", delimiter="\t")
r = np.genfromtxt("data/data_r.txt", delimiter="\t")
nsubjs, nstim = x.shape

# one entry per participant and test interval
valid = np.isfinite(x).flatten()
xij = x.flatten()[valid]
nij = n.flatten()[valid].astype(int)
rij = r.flatten()[valid].astype(int)
sbjid = np.repeat(np.arange(nsubjs), nstim)[valid]
# the test intervals are centered at the mean of each participant
xvect = np.repeat(np.nanmean(x, axis=1), nstim)[valid]


# %% [markdown]
# The priors are the group means and standard deviations of the intercepts
# `alpha` and the slopes `beta`. The participant effects have no `pm.Data`
# input, so the translated model samples them. PyMC3 gives the Normal priors a
# precision `tau = 0.001`; the port gives the standard deviation
# `sigma = 1 / sqrt(tau)`.


# %%
def uniform_prior(name: str, upper: float) -> Any:
    """Add a Uniform(0, upper) prior with pm.Data bounds to the model"""
    return pm.Uniform(
        name, lower=pm.Data(f"{name}_lower", 0.0), upper=pm.Data(f"{name}_upper", upper)
    )


def normal_prior(name: str) -> Any:
    """Add a vague Normal prior with pm.Data inputs to the model"""
    return pm.Normal(
        name,
        mu=pm.Data(f"{name}_mu", 0.0),
        sigma=pm.Data(f"{name}_sigma", 1 / np.sqrt(0.001)),
    )


def linear_predictor() -> Any:
    """Add the participant intercepts and slopes; return the linear predictor"""
    sigma_a, sigma_b = (
        uniform_prior("sigma_a", 1000.0),
        uniform_prior("sigma_b", 1000.0),
    )
    mu_a, mu_b = normal_prior("mu_a"), normal_prior("mu_b")
    alpha = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, shape=nsubjs)
    beta = pm.Normal("beta", mu=mu_b, sigma=sigma_b, shape=nsubjs)
    return alpha[sbjid] + beta[sbjid] * (xij - xvect)


with pm.Model() as model1:
    thetaij = pm.Deterministic("thetaij", pm.math.invlogit(linear_predictor()))
    pm.Binomial("rij", p=thetaij, n=nij, observed=rij)

compare(model1, ["thetaij", "rij"])

# %% [markdown]
# ## 12.2 Psychophysical functions under contamination
#
# A contaminated trial ignores the interval: its rate `pi_ij` is uniform. The
# latent indicator `zij` marks the contaminated trials, with a probability
# `phii` for each participant. The upstream notebook has two versions of this
# model. The first draws `zij` with a raw random stream, to initialize the
# sampler. The port uses the second, with a Bernoulli `zij`.

# %%
with pm.Model() as model2:
    linerpredi = linear_predictor()

    sigma_p = uniform_prior("sigma_p", 3.0)
    mu_p = normal_prior("mu_p")
    probitphi = pm.Normal("probitphi", mu=mu_p, sigma=sigma_p, shape=nsubjs)
    phii = pm.Deterministic("phii", phi(probitphi))

    pi_ij = pm.Uniform("pi_ij", lower=0.0, upper=1.0, shape=xij.shape)
    zij = pm.Bernoulli("zij", p=phii[sbjid], shape=xij.shape)
    thetaij = pm.Deterministic(
        "thetaij",
        pm.math.switch(pm.math.eq(zij, 0), pm.math.invlogit(linerpredi), pi_ij),
    )
    pm.Binomial("rij", p=thetaij, n=nij, observed=rij)

compare(model2, ["phii", "thetaij", "rij"])

# %% [markdown]
# ## References
#
# - Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
#   Practical Course*. Cambridge University Press.
