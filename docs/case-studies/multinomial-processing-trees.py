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
# # Multinomial processing trees
#
# This case study translates the models of chapter 14 of Lee & Wagenmakers
# (2013) with the PyMC adapter. The PyMC3 models come from the notebook
# [MultinomialProcessingTrees.ipynb](https://github.com/pymc-devs/pymc-resources/blob/main/BCM/CaseStudies/MultinomialProcessingTrees.ipynb)
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

SEED = tf.constant([2013, 14], dtype=tf.int32)
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
# ## The pair-clustering model
#
# In a free recall task, a participant studies word pairs that are related
# in meaning. The multinomial processing tree has three parameters: the
# probability `c` to store a pair as a cluster, `r` to retrieve a stored
# cluster, and `u` to recall a word that is not in a cluster. Each response
# falls into one of four categories, with the probabilities `t1` to `t4`.


# %%
def phi(x: Any) -> Any:
    """Cumulative distribution function of the standard normal distribution"""
    return 0.5 + 0.5 * pm.math.erf(x / np.sqrt(2))


def category_probabilities(c: Any, r: Any, u: Any) -> Any:
    """Return the probabilities of the four response categories"""
    t1 = c * r
    t2 = (1 - c) * u**2
    t3 = 2 * u * (1 - c) * (1 - u)
    t4 = c * (1 - r) + (1 - c) * (1 - u) ** 2
    return [t1, t2, t3, t4]


# %% [markdown]
# ## 14.1 Multinomial processing model of pair-clustering
#
# The data hold the response counts of three trials. PyMC3 swaps the counts
# with a shared variable; the port uses a `pm.Data` node, and `pm.set_data`
# swaps them. The check uses the first trial: the counts do not change the
# prior predictive, and all three trials have 420 responses.

# %%
kall = np.array([[45, 24, 97, 254], [106, 41, 107, 166], [243, 64, 65, 48]])


def beta_prior(name: str) -> Any:
    """Add a Beta(1, 1) prior with pm.Data inputs to the model"""
    return pm.Beta(
        name, alpha=pm.Data(f"{name}_a", 1.0), beta=pm.Data(f"{name}_b", 1.0)
    )


with pm.Model() as model1:
    k = pm.Data("k", kall[0])
    c, r, u = beta_prior("c"), beta_prior("r"), beta_prior("u")
    t = pm.math.stack(category_probabilities(c, r, u))
    pm.Multinomial("kobs", p=t, n=k.sum(), observed=k)

compare(model1, ["kobs"])

# %% [markdown]
# ## 14.2 Latent-trait MPT model
#
# The data of Riefer et al. (2002): 21 participants, each with 20 word
# pairs, in trials 1, 2 and 6. Each participant has latent traits `delta`
# on the probit scale, with a Wishart prior on their precision matrix.
#
# The upstream notebook uses `pm.WishartBartlett`. It is deprecated in the
# current PyMC, and is a thin wrapper around `pm.Wishart`, so the port uses
# `pm.Wishart`. The traits and the Wishart matrix have no `pm.Data` input, so
# the translated model samples them.

# %%
nsubj, nitem, p = 21, 20, 3
nu = p + 2
response = {
    1: np.array(
        [
            [2, 4, 4, 10],
            [2, 1, 3, 14],
            [2, 2, 5, 11],
            [6, 0, 4, 10],
            [1, 0, 4, 15],
            [1, 0, 2, 17],
            [1, 2, 4, 13],
            [4, 1, 6, 9],
            [5, 1, 4, 10],
            [1, 0, 9, 10],
            [5, 0, 3, 12],
            [0, 1, 6, 13],
            [1, 5, 7, 7],
            [1, 1, 4, 14],
            [2, 2, 3, 13],
            [2, 1, 5, 12],
            [2, 0, 6, 12],
            [1, 0, 5, 14],
            [2, 1, 8, 9],
            [3, 0, 2, 15],
            [1, 2, 3, 14],
        ]
    ),
    2: np.array(
        [
            [7, 5, 3, 5],
            [5, 2, 3, 10],
            [6, 2, 7, 5],
            [9, 4, 2, 5],
            [2, 2, 7, 9],
            [1, 3, 3, 13],
            [5, 0, 5, 10],
            [7, 3, 4, 6],
            [7, 3, 6, 4],
            [4, 1, 10, 5],
            [9, 1, 2, 8],
            [3, 1, 6, 10],
            [3, 5, 9, 3],
            [2, 0, 6, 12],
            [8, 0, 3, 9],
            [3, 2, 7, 8],
            [7, 1, 5, 7],
            [2, 1, 6, 11],
            [5, 3, 5, 7],
            [5, 0, 6, 9],
            [6, 2, 2, 10],
        ]
    ),
    6: np.array(
        [
            [14, 3, 1, 2],
            [12, 3, 1, 4],
            [18, 0, 1, 1],
            [15, 3, 0, 2],
            [7, 1, 10, 2],
            [3, 6, 11, 0],
            [8, 4, 3, 5],
            [17, 1, 1, 1],
            [13, 4, 3, 0],
            [11, 6, 1, 2],
            [16, 1, 2, 1],
            [10, 1, 3, 6],
            [7, 13, 0, 0],
            [8, 4, 3, 5],
            [16, 1, 1, 2],
            [5, 4, 7, 4],
            [15, 0, 5, 0],
            [6, 3, 6, 5],
            [17, 2, 0, 1],
            [17, 1, 0, 2],
            [8, 3, 6, 3],
        ]
    ),
}


def latent_traits() -> Any:
    """Add the prior of the group means; return it"""
    mu = pm.Normal(
        "mu", mu=pm.Data("mu_mu", 0.0), sigma=pm.Data("mu_sigma", 1.0), shape=p
    )
    return mu


def likelihood(mu: Any, xi: Any, delta: Any, k: Any) -> None:
    """Add the group probabilities and the counts to the model"""
    c = phi(mu[0] + xi[0] * delta[:, 0])
    r = phi(mu[1] + xi[1] * delta[:, 1])
    u = phi(mu[2] + xi[2] * delta[:, 2])
    pm.Deterministic("muc", phi(mu[0]))
    pm.Deterministic("mur", phi(mu[1]))
    pm.Deterministic("muu", phi(mu[2]))
    t = pm.math.stack(category_probabilities(c, r, u))
    pm.Multinomial("kobs", p=t.T, n=nitem, observed=k)


with pm.Model() as model2:
    k = pm.Data("k", response[1])
    mu = latent_traits()
    xi = pm.Uniform(
        "xi", lower=pm.Data("xi_lower", 0.0), upper=pm.Data("xi_upper", 100.0), shape=p
    )
    wishart = pm.Wishart("wishart", nu=nu, V=np.eye(p))
    cov = pm.math.matrix_inverse(wishart)
    delta = pm.MvNormal("delta", mu=np.zeros(p), cov=cov, shape=(nsubj, p))
    likelihood(mu, xi, delta, k)

compare(model2, ["muc", "kobs"])

# %% [markdown]
# ## An LKJ prior for the covariance
#
# The upstream notebook discourages the Wishart prior. It reparametrizes the
# model with an `LKJCholeskyCov` prior on the Cholesky factor of the
# covariance, and uses trial 6. The factor is latent, so the translated model
# samples it.

# %%
with pm.Model() as model3:
    k = pm.Data("k", response[6])
    mu = latent_traits()
    xi = pm.HalfNormal("xi", sigma=pm.Data("xi_sigma", 1.0), shape=p)
    packed_chol = pm.LKJCholeskyCov(
        "chol_cov", n=p, eta=4, sd_dist=pm.Exponential.dist(1.0), compute_corr=False
    )
    chol = pm.math.expand_packed_triangular(p, packed_chol, lower=True)
    vals_raw = pm.Normal("vals_raw", mu=0.0, sigma=1.0, shape=(p, nsubj))
    delta = pm.math.dot(chol, vals_raw).T
    likelihood(mu, xi, delta, k)

compare(model3, ["muc", "kobs"])

# %% [markdown]
# ## References
#
# - Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
#   Practical Course*. Cambridge University Press.
# - Riefer, D. M., Knapp, B. R., Batchelder, W. H., Bamber, D., & Manifold, V.
#   (2002). Cognitive psychometrics: Assessing storage and retrieval deficits
#   in special populations with multinomial processing tree models.
#   *Psychological Assessment*, 14, 184-201.
