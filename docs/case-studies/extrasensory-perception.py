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
# # Extrasensory perception
#
# This case study translates the models of chapter 13 of Lee & Wagenmakers
# (2013) with the PyMC adapter. The PyMC3 models come from the notebook
# [ExtrasensoryPerception.ipynb](https://github.com/pymc-devs/pymc-resources/blob/main/BCM/CaseStudies/ExtrasensoryPerception.ipynb)
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

SEED = tf.constant([2013, 13], dtype=tf.int32)
DRAWS = 10_000

# %% [markdown]
# ## The check
#
# The port of a model follows the rules of the how-to guide
# [Use a PyMC model][use-a-pymc-model]: each input of a
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
# ## The correlation model
#
# All three models share a bivariate Normal with means `mu`, standard
# deviations `sigma1` and `sigma2`, and correlation `r`. PyMC3 gives the
# MvNormal a precision matrix `tau1`, the inverse of the covariance. The port
# keeps the inverse, and passes `tau=tau1`.
#
# The book gives the precisions `lambda1` and `lambda2` a
# `Gamma(0.001, 0.001)` prior. Most of its draws underflow to 0, and the
# covariance is then singular: `pm.draw` fails. The check therefore sets
# `Gamma(2, 2)` with `pm.set_data`. The translation does not depend on these
# values.


# %%
def uniform_prior(name: str, lower: float, upper: float) -> Any:
    """Add a Uniform prior with pm.Data bounds to the model"""
    return pm.Uniform(
        name,
        lower=pm.Data(f"{name}_lower", lower),
        upper=pm.Data(f"{name}_upper", upper),
    )


def gamma_prior(name: str) -> Any:
    """Add a Gamma(0.001, 0.001) prior with pm.Data inputs to the model"""
    alpha = pm.Data(f"{name}_alpha", 0.001)
    beta = pm.Data(f"{name}_beta", 0.001)
    return pm.Gamma(name, alpha=alpha, beta=beta)


CHECK_PRIORS = {
    "lambda1_alpha": 2.0,
    "lambda1_beta": 2.0,
    "lambda2_alpha": 2.0,
    "lambda2_beta": 2.0,
}


def correlation_priors(r_lower: float) -> tuple[Any, Any]:
    """Add the priors; return the means and the precision matrix"""
    r = uniform_prior("r", r_lower, 1.0)
    mu = pm.Normal(
        "mu",
        mu=pm.Data("mu_mu", 0.0),
        sigma=pm.Data("mu_sigma", 1 / np.sqrt(0.001)),
        shape=2,
    )
    lambda1, lambda2 = gamma_prior("lambda1"), gamma_prior("lambda2")
    sigma1 = pm.Deterministic("sigma1", 1 / pm.math.sqrt(lambda1))
    sigma2 = pm.Deterministic("sigma2", 1 / pm.math.sqrt(lambda2))
    cov = pm.math.stack(
        [[1 / lambda1, r * sigma1 * sigma2], [r * sigma1 * sigma2, 1 / lambda2]]
    )
    tau1 = pm.Deterministic("tau1", pm.math.matrix_inverse(cov))
    return mu, tau1


def phi(x: Any) -> Any:
    """Cumulative distribution function of the standard normal distribution"""
    return 0.5 + 0.5 * pm.math.erf(x / np.sqrt(2))


# %% [markdown]
# ## 13.1 Evidence for optional stopping
#
# The nine experiments of Bem (2011): the sample size `N` and the effect
# size `E`. A negative correlation between them suggests optional stopping.

# %%
N = np.array([100, 150, 97, 99, 100, 150, 200, 100, 50])
E = np.array([0.25, 0.20, 0.25, 0.20, 0.22, 0.15, 0.09, 0.19, 0.42])

with pm.Model() as model1:
    mu, tau1 = correlation_priors(-1.0)
    pm.MvNormal("yd", mu=mu, tau=tau1, observed=np.vstack([N, E]).T)

pm.set_data(CHECK_PRIORS, model=model1)
compare(model1, ["sigma1", "tau1", "yd"])

# %% [markdown]
# ## 13.2 Evidence for differences in ability
#
# 100 participants each do two sessions of 60 trials. `k` counts the correct
# responses on erotic pictures in each session. A positive correlation of the
# latent abilities `thetai` means stable differences in ability.

# %%
Nt = 60
k = np.array(
    [
        [
            36,
            32,
            36,
            36,
            28,
            40,
            40,
            24,
            36,
            36,
            28,
            40,
            28,
            36,
            20,
            24,
            24,
            16,
            20,
            32,
            40,
            32,
            36,
            24,
            28,
            44,
            40,
            36,
            40,
            32,
            32,
            40,
            28,
            20,
            24,
            32,
            24,
            24,
            20,
            28,
            24,
            28,
            28,
            32,
            20,
            44,
            16,
            36,
            32,
            28,
            24,
            32,
            40,
            28,
            32,
            32,
            28,
            24,
            28,
            40,
            28,
            20,
            20,
            20,
            24,
            24,
            36,
            28,
            20,
            20,
            40,
            32,
            20,
            36,
            28,
            28,
            24,
            20,
            28,
            32,
            48,
            24,
            32,
            32,
            40,
            40,
            40,
            36,
            36,
            32,
            20,
            28,
            40,
            32,
            20,
            20,
            16,
            16,
            28,
            40,
        ],
        [
            20,
            36,
            32,
            16,
            40,
            32,
            40,
            28,
            28,
            40,
            24,
            40,
            16,
            24,
            28,
            20,
            32,
            36,
            20,
            24,
            28,
            28,
            36,
            32,
            32,
            36,
            32,
            40,
            36,
            16,
            28,
            24,
            36,
            32,
            24,
            28,
            32,
            20,
            24,
            28,
            48,
            36,
            12,
            36,
            24,
            24,
            16,
            16,
            36,
            24,
            24,
            24,
            24,
            24,
            40,
            44,
            32,
            32,
            20,
            36,
            32,
            32,
            28,
            32,
            28,
            32,
            24,
            24,
            28,
            36,
            36,
            36,
            28,
            36,
            40,
            32,
            28,
            36,
            12,
            32,
            28,
            24,
            32,
            32,
            32,
            32,
            36,
            40,
            24,
            24,
            32,
            48,
            36,
            24,
            12,
            36,
            40,
            28,
            28,
            28,
        ],
    ]
).T

with pm.Model() as model2:
    mu, tau1 = correlation_priors(0.0)
    thetai = pm.MvNormal("thetai", mu=mu, tau=tau1, shape=k.shape)
    pm.Binomial("kij", p=phi(thetai), n=Nt, observed=k)

pm.set_data(CHECK_PRIORS, model=model2)
compare(model2, ["kij"])

# %% [markdown]
# ## 13.3 Evidence for the impact of extraversion
#
# The same participants: `k2` counts the correct responses of session 1, and
# `x2` holds an extraversion score between 0 and 100, measured with the
# standard deviation `sigmax = 3`.

# %%
k2 = k[:, 0]
x2 = np.array(
    [
        50,
        80,
        79,
        56,
        50,
        80,
        53,
        84,
        74,
        67,
        50,
        45,
        62,
        65,
        71,
        71,
        68,
        63,
        67,
        58,
        72,
        73,
        63,
        54,
        63,
        70,
        81,
        71,
        66,
        74,
        70,
        84,
        66,
        73,
        78,
        64,
        54,
        74,
        62,
        71,
        70,
        79,
        66,
        64,
        62,
        63,
        60,
        56,
        72,
        72,
        79,
        67,
        46,
        67,
        77,
        55,
        63,
        44,
        84,
        65,
        41,
        62,
        64,
        51,
        46,
        53,
        26,
        67,
        73,
        39,
        62,
        59,
        75,
        65,
        60,
        69,
        63,
        69,
        55,
        63,
        86,
        70,
        67,
        54,
        80,
        71,
        71,
        55,
        57,
        41,
        56,
        78,
        58,
        76,
        54,
        50,
        61,
        60,
        32,
        67,
    ]
)
sigmax = 3

with pm.Model() as model3:
    mu, tau1 = correlation_priors(0.0)
    thetai = pm.MvNormal("thetai", mu=mu, tau=tau1, shape=(len(k2), 2))
    pm.Binomial("ki", p=phi(thetai[:, 0]), n=Nt, observed=k2)
    pm.Normal("xi", mu=100 * phi(thetai[:, 1]), sigma=sigmax, observed=x2)

pm.set_data(CHECK_PRIORS, model=model3)
compare(model3, ["ki", "xi"])

# %% [markdown]
# ## References
#
# - Bem, D. J. (2011). Feeling the future: Experimental evidence for anomalous
#   retroactive influences on cognition and affect. *Journal of Personality
#   and Social Psychology*, 100, 407-425.
# - Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
#   Practical Course*. Cambridge University Press.
