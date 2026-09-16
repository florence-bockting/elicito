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
# # Memory retention
#
# This case study translates the models of chapter 10 of Lee & Wagenmakers
# (2013) with the PyMC adapter. The PyMC3 models come from the notebook
# [MemoryRetention.ipynb](https://github.com/pymc-devs/pymc-resources/blob/main/BCM/CaseStudies/MemoryRetention.ipynb)
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
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# pm.draw runs the truncated Normal of PyMC in Numba object mode
warnings.filterwarnings("ignore", message="Numba will use object mode")

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

SEED = tf.constant([2013, 10], dtype=tf.int32)
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
# Participants study a list of items. At ten delays `t`, the data count how
# many of `n = 18` items each participant still recalls. Participant 4 has no
# data: the models predict the recall of this participant.

# %%
t = np.array([1, 2, 4, 7, 12, 21, 35, 59, 99, 200])
n = 18
k = np.array(
    [
        [18, 18, 16, 13, 9, 6, 4, 4, 4],
        [17, 13, 9, 6, 4, 4, 4, 4, 4],
        [14, 10, 6, 4, 4, 4, 4, 4, 4],
    ]
)
ns = 4  # participants, with participant 4 not observed
ns_obs, nt_obs = k.shape  # the observed participants and delays
t_obs = t[:nt_obs]


# %% [markdown]
# Each model predicts the recall rate `theta = exp(-alpha * t) + beta`, capped
# at 1. `kij` holds the observed counts. `kij_ppc` holds the predicted counts
# of all four participants at all ten delays. The upstream notebook draws
# `kij_ppc` with a raw random stream; the port draws it with
# `pm.Binomial.dist` inside a `pm.Deterministic`.
#
# The upstream notebook has two versions of model 10.1. The first observes
# a masked array, so PyMC imputes participant 4. The port uses the second,
# which predicts participant 4 with `kij_ppc`.


# %%
def beta_prior(name: str, shape: Any = None) -> Any:
    """Add a Beta(1, 1) prior with pm.Data inputs to the model"""
    a = pm.Data(f"{name}_a", 1.0)
    b = pm.Data(f"{name}_b", 1.0)
    return pm.Beta(name, alpha=a, beta=b, shape=shape)


def recall(alpha: Any, beta: Any) -> None:
    """Add the recall rates, the observed counts and the predicted counts"""
    theta = pm.math.exp(-alpha[:ns_obs] * t_obs[:, None]) + beta[:ns_obs]
    thetaj = pm.Deterministic("thetaj", pm.math.clip(theta.T, 0, 1))
    pm.Binomial("kij", p=thetaj, n=n, observed=k)
    theta2 = pm.math.minimum(pm.math.exp(-alpha * t[:, None]) + beta, 1.0)
    pm.Deterministic("kij_ppc", pm.Binomial.dist(n=n, p=theta2.T))


OUTPUTS = ["thetaj", "kij", "kij_ppc"]

# %% [markdown]
# ## 10.1 No individual differences
#
# All participants share one `alpha` and one `beta`.

# %%
with pm.Model() as model1:
    alpha = beta_prior("alpha")
    beta = beta_prior("beta")
    recall(pm.math.stack([alpha] * ns), pm.math.stack([beta] * ns))

compare(model1, OUTPUTS)

# %% [markdown]
# ## 10.2 Full individual differences
#
# Each participant has an own `alpha` and `beta`. A prior with four elements
# becomes four parameters, `alpha[0]` to `alpha[3]`, with shared
# hyperparameters.

# %%
with pm.Model() as model2:
    recall(beta_prior("alpha", shape=ns), beta_prior("beta", shape=ns))

compare(model2, OUTPUTS)

# %% [markdown]
# ## 10.3 Structured individual differences
#
# The participant parameters come from group distributions: a Normal
# truncated to [0, 1], with a Beta prior on the mean and a Gamma prior on the
# precision. The participant parameters have no `pm.Data` input, so the
# translated model samples them.
#
# The book gives the precisions a `Gamma(0.001, 0.001)` prior. Most of its
# draws underflow to 0, and `pm.draw` then fails: the scale of the truncated
# Normal is not finite. The check therefore sets `Gamma(2, 2)` with
# `pm.set_data`. The translation does not depend on these values.


# %%
def gamma_prior(name: str) -> Any:
    """Add a Gamma(0.001, 0.001) prior with pm.Data inputs to the model"""
    alpha = pm.Data(f"{name}_alpha", 0.001)
    beta = pm.Data(f"{name}_beta", 0.001)
    return pm.Gamma(name, alpha=alpha, beta=beta)


with pm.Model() as model3:
    mua, lambdaa = beta_prior("mua"), gamma_prior("lambdaa")
    mub, lambdab = beta_prior("mub"), gamma_prior("lambdab")
    alpha = pm.TruncatedNormal(
        "alpha", mu=mua, sigma=1 / pm.math.sqrt(lambdaa), lower=0, upper=1, shape=ns
    )
    beta = pm.TruncatedNormal(
        "beta", mu=mub, sigma=1 / pm.math.sqrt(lambdab), lower=0, upper=1, shape=ns
    )
    recall(alpha, beta)

pm.set_data(
    {
        "lambdaa_alpha": 2.0,
        "lambdaa_beta": 2.0,
        "lambdab_alpha": 2.0,
        "lambdab_beta": 2.0,
    },
    model=model3,
)
compare(model3, OUTPUTS)

# %% [markdown]
# ## References
#
# - Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
#   Practical Course*. Cambridge University Press.
