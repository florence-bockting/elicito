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
# # Number concept development
#
# This case study translates the models of chapter 19 of Lee & Wagenmakers
# (2013) with the PyMC adapter. The PyMC3 models come from the notebook
# [NumberConceptDevelopment.ipynb](https://github.com/pymc-devs/pymc-resources/blob/main/BCM/CaseStudies/NumberConceptDevelopment.ipynb)
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
from scipy import io, stats

from elicito.adapter import pymc as adapter_pymc

SEED = tf.constant([2013, 19], dtype=tf.int32)
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
# 20 children do two tasks. In the Give-N task, a child is asked for `q`
# items and gives `a` items, up to 15. In the Fast-Cards task, a child sees
# `q` items and says a number `a`, up to 50. Each child has a latent knower
# level `z`, from 0-knower to CP-knower, with 6 levels.

# %%
matdata = io.loadmat("data/fc_given.mat")
ns = int(matdata["ns"][0, 0])  # children
nz = int(matdata["nz"][0, 0])  # knower levels
gn = int(matdata["gn"][0, 0])  # possible answers in Give-N
fn = 50  # possible answers in Fast-Cards


def knower_index(n_answers: int) -> Any:
    """
    Return the code of each knower level, question and answer

    The code is 0 if the answer does not depend on the knower level, 1 if the
    level makes the answer less likely, and 2 if it makes it more likely.
    """
    i1 = np.arange(1, nz + 1)[:, None, None]
    j1 = np.arange(1, gn + 1)[None, :, None]
    k1 = np.arange(1, n_answers + 1)[None, None, :]
    # the knower level is the same as or greater than the answer
    ind1 = (i1 - 1 >= k1).astype(int)
    # the answer matches the question
    ind2 = (k1 == j1).astype(int)
    # 0-knower and CP-knower
    ind3 = (i1 == 1).astype(int)
    ind4 = (i1 == nz).astype(int)
    ind5 = ind3 + ind4 * (2 + ind2) + (1 - ind4) * (1 - ind3) * (ind1 * ind2 + ind1 + 1)
    return ind5 - 1


def responses(task: str) -> tuple[Any, Any, Any, int]:
    """Return the answers, the questions and the valid entries of a task"""
    a_obs = np.int64(matdata[f"{task}a"]).flatten() - 1
    q_obs = np.int64(matdata[f"{task}q"]).flatten() - 1
    valid = np.where(q_obs != -1)[0]
    return a_obs, q_obs, valid, matdata[f"{task}q"].shape[1]


# %% [markdown]
# Each task has an answer distribution `pi` and an evidence weight `nu`. The
# base rates `pi` have a Dirichlet prior with constant inputs, and the knower
# levels `zi` a Categorical prior with constant inputs, so the translated
# model samples them. `nu` is the only prior with `pm.Data` inputs.


# %%
def task_block(suffix: str, task: str, zi: Any, n_answers: int) -> None:
    """Add the answer model of one task to the model"""
    ind5r = knower_index(n_answers)
    a_obs, q_obs, valid, nq = responses(task)
    pi = pm.Dirichlet(f"pi{suffix}", a=np.ones(n_answers), shape=n_answers)
    nu = pm.Uniform(
        f"nu{suffix}",
        lower=pm.Data(f"nu{suffix}_lower", 1.0),
        upper=pm.Data(f"nu{suffix}_upper", 1000.0),
    )
    nu_vec = pm.math.stack([1.0, 1.0 / nu, nu])
    piprime = nu_vec[ind5r] * pi
    npiprime = piprime / pm.math.sum(piprime, axis=-1, keepdims=True)
    zi_vec = pm.math.repeat(zi, nq)
    pi_ij = npiprime[zi_vec[valid], q_obs[valid], :]
    pm.Categorical(f"a{suffix}ij", p=pi_ij, observed=a_obs[valid])


def knower_levels() -> Any:
    """Add the knower level of each child to the model"""
    return pm.Categorical("zi", p=np.ones(nz) / nz, shape=ns)


# %% [markdown]
# ## 19.1 Knower-level model for Give-N

# %%
with pm.Model() as model1:
    task_block("", "g", knower_levels(), gn)

compare(model1, ["aij"])

# %% [markdown]
# ## 19.2 Knower-level model for Fast-Cards

# %%
with pm.Model() as model2:
    task_block("", "f", knower_levels(), fn)

compare(model2, ["aij"])

# %% [markdown]
# ## 19.3 Knower-level model for Give-N and Fast-Cards
#
# Both tasks share the knower levels `zi`.

# %%
with pm.Model() as model3:
    zi = knower_levels()
    task_block("_g", "g", zi, gn)
    task_block("_f", "f", zi, fn)

compare(model3, ["a_gij", "a_fij"])

# %% [markdown]
# ## References
#
# - Lee, M. D., & Sarnecka, B. W. (2010). A model of knower-level behavior in
#   number concept development. *Cognitive Science*, 34, 51-67.
# - Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling: A
#   Practical Course*. Cambridge University Press.
