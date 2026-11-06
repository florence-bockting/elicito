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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # A hierarchical prior for a panel of experts
#
# The other tutorials learn a prior from one expert. This tutorial learns a
# prior from a **panel** of five experts who do not agree.
#
# Each expert gets a prior of their own. The priors of all experts share a
# panel mean, and each expert has an offset from it. The spread of the
# offsets is the disagreement in the panel. The result is the **population
# prior**: the prior of a new expert from the same panel.
#
# **You learn how to:**
#
# 1. give each expert a prior with a shared mean and an offset,
# 2. simulate the answers of a panel of experts,
# 3. add a term of your own to the loss with `el.trainer(penalty=...)`,
# 4. estimate the disagreement in two stages, and why one stage fails,
# 5. build the population prior from the result.

# %% tags=["hide"]
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib


def _warm() -> None:
    import elicito  # noqa: F401


joblib.Parallel(n_jobs=4)(joblib.delayed(_warm)() for _ in range(4))

import tensorflow as tf

tf.constant(0.0)

# %% [markdown]
# ## Imports

# %%
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

# %% [markdown]
# ## The model
#
# ### Probabilistic model
#
# We use the linear regression of the tutorial on parametric priors. Each
# expert $e = 1, \dots, 5$ has their own coefficients:
#
# $$
# \begin{align*}
#     \beta_{j,e} &\sim \text{Normal}(\mu_j + \delta_{j,e}, \sigma_j)
#         \qquad j = 0, 1 \\
#     \delta_{j,e} &\sim \text{Normal}(0, \tau_j) \\
#     \sigma &\sim \text{HalfNormal}(s) \\
#     \mu_e &= \beta_{0,e} + \beta_{1,e} X \\
#     y_{pred,e} &\sim \text{Normal}(\mu_e, \sigma)
# \end{align*}
# $$
#
# - $\mu_j$ is the panel mean of coefficient $j$. All experts share it.
# - $\sigma_j$ is the uncertainty of one expert. All experts share it.
# - $\delta_{j,e}$ is the offset of expert $e$ from the panel mean.
# - $\tau_j$ is the spread of the offsets: the disagreement in the panel.
# - The noise $\sigma$ is the same for all experts.
#
# A new expert from the same panel has the **population prior**
#
# $$
# \beta_j \sim \text{Normal}\left(\mu_j, \sqrt{\sigma_j^2 + \tau_j^2}\right).
# $$
#
# The disagreement $\tau_j$ makes it wider than the prior of one expert.
#
# ### Design matrix
#
# The design is the same as in the tutorial on parametric priors. We ask each
# expert about the outcome at three values of the predictor: its 25th
# ($X_0$), 50th ($X_1$) and 75th ($X_2$) percentile.


# %%
def X_design(N: int, quantiles: list[float]) -> Any:
    """
    Compute the design matrix

    Parameters
    ----------
    N
        number of observations

    quantiles
        percentiles of the scaled predictor, from 0 to 100

    Returns
    -------
    :
        design matrix with one row per percentile
    """
    X = tf.cast(np.arange(N), tf.float32)
    X_scaled = X / tf.math.reduce_std(X)
    X_sel = tfp.stats.percentile(X_scaled, quantiles)
    design = tf.stack([tf.ones(X_sel.shape), X_sel], -1)
    return design


DESIGN = X_design(N=30, quantiles=[25, 50, 75])
DESIGN

# %% [markdown]
# ### The panel
#
# We simulate the panel, so we know the truth. The disagreement is large for
# the intercept ($\tau_0 = 1.5$) and small for the slope ($\tau_1 = 0.5$).
# We draw one offset for each expert and coefficient.

# %%
EXPERTS = (1, 2, 3, 4, 5)
N_EXPERTS = len(EXPERTS)

MU = (5.0, 2.0)
SIGMA = (1.0, 1.0)
TAU = (3.0, 1.0)
S = 7.0

rng = np.random.default_rng(2025)
# one row per expert, one column per coefficient
DELTA = rng.normal(loc=0.0, scale=TAU, size=(N_EXPERTS, 2))

for e in EXPERTS:
    d0, d1 = DELTA[e - 1]
    print(f"expert {e}: delta0 = {d0:+.2f}, delta1 = {d1:+.2f}")

# %% [markdown]
# Five offsets are a small sample. Their standard deviation is the best
# estimate of $\tau_j$ that any method can reach from this panel. It is not
# the true $\tau_j$.

# %%
DELTA_SD = DELTA.std(axis=0, ddof=1)
print(
    f"sd of the offsets: {DELTA_SD[0]:.2f} (tau0 = {TAU[0]}),"
    f" {DELTA_SD[1]:.2f} (tau1 = {TAU[1]})"
)

# %% [markdown]
# ### Generative model
#
# The model receives the prior samples of all experts at once. Their shape is
# `(B, num_samples, 2 * 5 + 1)`. The order is $\beta_{0,1}, \beta_{1,1},
# \beta_{0,2}, \dots, \beta_{1,5}, \sigma$. The model reshapes the
# coefficients to one row per expert. It returns one target quantity per
# expert and design point, for example `y_X0_e1`.


# %%
class PanelModel:
    """
    Generative model of the linear regression, for every expert at once
    """

    def __call__(self, prior_samples: Any, design_matrix: Any) -> dict[str, Any]:
        """
        Compute the target quantities of every expert

        Parameters
        ----------
        prior_samples
            prior samples, the coefficients of every expert and the noise

        design_matrix
            design matrix

        Returns
        -------
        :
            dictionary with the target quantities
        """
        # coefficients with one row per expert: (B, num_samples, E, 2)
        shape = tf.concat([tf.shape(prior_samples)[:2], [N_EXPERTS, 2]], 0)
        betas = tf.reshape(prior_samples[:, :, :-1], shape)

        # linear predictor at the three design points: (B, num_samples, E, 3)
        epred = tf.einsum("bsej,kj->bsek", betas, design_matrix)

        # prior predictive distribution, with the noise of all experts
        sigma = prior_samples[:, :, -1][:, :, None, None]
        ypred = tfd.Normal(loc=epred, scale=sigma).sample()

        return {
            f"y_X{k}_e{e}": ypred[:, :, e - 1, k] for e in EXPERTS for k in range(3)
        }


# %%
model = el.model(obj=PanelModel, design_matrix=DESIGN)

# %% [markdown]
# ## Model parameters
#
# `elicito` learns the hyperparameters of a prior family. Here the loc of
# $\beta_{j,e}$ is a sum of two hyperparameters, $\mu_j + \delta_{j,e}$. A
# prior family can be any function that returns a distribution, so we write
# one:


# %%
def shifted_normal(mu: Any, delta: Any, scale: Any) -> Any:
    """Return the Normal prior of one expert, at the panel mean plus its offset"""
    return tfd.Normal(loc=mu + delta, scale=scale)


# %% [markdown]
# | hyperparameter | meaning | used by |
# | :-- | :-- | :-- |
# | $\mu_0$, $\mu_1$ | panel mean | all experts, `shared=True` |
# | $\sigma_0$, $\sigma_1$ | uncertainty of one expert | all experts, `shared=True` |
# | $\delta_{j,e}$ | offset of expert $e$ | expert $e$ only |
# | $s$ | scale of the noise prior | $\sigma$ |
#
# `shared=True` makes all experts use the same variable. That gives
# 2 + 2 + 10 + 1 = 15 hyperparameters for 11 model parameters.

# %%
parameters = [
    el.parameter(
        name=f"beta{j}_e{e}",
        family=shifted_normal,
        hyperparams=dict(
            mu=el.hyper(f"mu{j}", shared=True),
            delta=el.hyper(f"delta{j}_e{e}"),
            scale=el.hyper(f"sigma{j}", lower=0, shared=True),
        ),
    )
    for e in EXPERTS
    for j in (0, 1)
]
parameters.append(
    el.parameter(
        name="sigma",
        family=tfd.HalfNormal,
        hyperparams=dict(scale=el.hyper("s", lower=0)),
    )
)

# %% [markdown]
# ## Target quantities and elicitation techniques
#
# Each expert states five quantiles of the outcome at each of the three
# design points. That gives 5 x 3 = 15 targets.

# %%
targets = [
    el.target(
        name=f"y_X{k}_e{e}",
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    )
    for e in EXPERTS
    for k in range(3)
]

# %% [markdown]
# ## Expert elicitation
#
# The oracle of expert $e$ is the true prior of that expert. One ground
# truth holds the priors of all experts. Its order must be the order of
# `parameters`.

# %%
ground_truth = {}
for e in EXPERTS:
    for j in (0, 1):
        ground_truth[f"beta{j}_e{e}"] = tfd.Normal(
            loc=float(MU[j] + DELTA[e - 1, j]), scale=float(SIGMA[j])
        )
ground_truth["sigma"] = tfd.HalfNormal(scale=S)

expert = el.expert.simulator(ground_truth=ground_truth, num_samples=10_000)

# %% [markdown]
# A quick check of the model: one forward pass with draws from the ground
# truth must return 15 targets of shape `(1, num_samples)`.

# %%
draws = tf.stack([prior.sample((1, 200)) for prior in ground_truth.values()], -1)
check = PanelModel()(draws, DESIGN)
print(len(check), "targets, shape", check["y_X0_e1"].shape)

# %% [markdown]
# ## A term of your own in the loss
#
# The model says $\delta_{j,e} \sim \text{Normal}(0, \tau_j)$. The targets
# cannot see this line, because they depend on $\mu_j + \delta_{j,e}$ only.
# So the targets cannot separate the panel mean from the offsets. A term in
# the loss must hold the offsets near 0.
#
# `el.trainer(penalty=...)` adds such a term. The function receives the
# hyperparameters by name, on the constrained scale. It returns one number.
#
# ### Why one fit cannot learn $\tau_j$
#
# The obvious term is the negative log density of the offsets,
# $\sum_{j,e} \left[\log \tau_j + \delta_{j,e}^2 / (2 \tau_j^2)\right]$.
# Set every offset to 0. The term is then $5 \log \tau_j$ per coefficient,
# and it goes to $-\infty$ as $\tau_j$ goes to 0. The MMD loss at
# $\delta = 0$ is finite. So the search makes all experts equal and $\tau_j$
# zero. This is the known collapse of a joint mode in a hierarchical model.
#
# ### Two stages
#
# We keep $\tau_j$ fixed in each fit. The term is then a ridge,
# $\sum_{j,e} \delta_{j,e}^2 / (2 \tau_j^2)$, and it has a minimum.
#
# 1. Fit with a weak ridge, $\tau_j = 10$. The offsets follow the experts.
# 2. Estimate $\hat\tau_j$ as the standard deviation of the learned offsets.
# 3. Fit again with $\tau_j = \hat\tau_j$. The ridge now pulls each offset
#    toward 0, as far as the disagreement in the panel allows.
#
# The loss is a discrepancy, not a likelihood. So the weight 1 of the ridge
# is a choice, not a result.


# %%
def ridge(tau: tuple[float, float]) -> Any:
    """Return the hierarchy term for a fixed spread of the offsets"""

    def penalty(hyper: dict[str, Any]) -> Any:
        """Sum the squared offsets, each scaled by its spread"""
        return sum(
            hyper[f"delta{j}_e{e}"] ** 2 / (2.0 * tau[j] ** 2)
            for e in EXPERTS
            for j in (0, 1)
        )

    return penalty


# %% [markdown]
# CMA-ES searches all 15 hyperparameters at once. `epochs` is its budget of
# objective evaluations. The history keeps the best candidate of each
# generation, so the generation with the lowest score holds the result.

# %%
BUDGET = 3000
SEEDS = [2025, 2026, 2027, 2028, 2029]


def fit_panel(tau: tuple[float, float], seed: int = 2025) -> Any:
    """Fit the panel with the ridge of a fixed spread of the offsets"""
    eliobj = el.Elicit(
        model=model,
        parameters=parameters,
        targets=targets,
        expert=expert,
        optimizer=el.optimizer(optimizer="cmaes", sigma0=1.0),
        trainer=el.trainer(
            method="parametric_prior",
            seed=seed,
            epochs=BUDGET,
            progress=0,
            penalty=ridge(tau),
        ),
    )
    eliobj.fit(
        parallel=el.utils.parallel(runs=len(SEEDS), cores=len(SEEDS), seeds=SEEDS)
    )
    return eliobj


def learned(eliobj: Any) -> dict[str, float]:
    """Return the hyperparameters of the best generation of all replications"""
    hyper = eliobj.results["history_stats/hyperparameter"]
    score = eliobj.results["history_stats/loss"]["total_loss"].values
    rep, best = np.unravel_index(np.nanargmin(score), score.shape)
    return {name: float(hyper[name].values[rep, best]) for name in hyper.data_vars}


def learned_runs(eliobj: Any) -> list[dict[str, float]]:
    """Return the hyperparameters of the best generation of each replication"""
    hyper = eliobj.results["history_stats/hyperparameter"]
    score = eliobj.results["history_stats/loss"]["total_loss"].values
    best = np.nanargmin(score, axis=1)
    return [
        {name: float(hyper[name].values[r, b]) for name in hyper.data_vars}
        for r, b in enumerate(best)
    ]


def offsets(hyper: dict[str, float]) -> Any:
    """Return the learned offsets, one row per expert"""
    return np.array([[hyper[f"delta{j}_e{e}"] for j in (0, 1)] for e in EXPERTS])


# %% [markdown]
# ### Stage 1: a weak ridge

# %%
stage1 = fit_panel(tau=(10.0, 10.0))
hyper1 = learned(stage1)

TAU_HAT = tuple(float(sd) for sd in offsets(hyper1).std(axis=0, ddof=1))
print(f"tau_hat = ({TAU_HAT[0]:.2f}, {TAU_HAT[1]:.2f})")
print(f"sd of the true offsets = ({DELTA_SD[0]:.2f}, {DELTA_SD[1]:.2f})")

# %% [markdown]
# ### Stage 2: the ridge of the panel

# %%
stage2 = fit_panel(tau=TAU_HAT)
hyper2 = learned(stage2)

# %%
el.plots.elicits(stage2, cols=3, figsize=(7, 11));

# %%
# the ridge moves the mean offset into mu, so centre the truth
truth = DELTA - DELTA.mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
for j, ax in enumerate(axes):
    ax.axhline(0, color="0.8", lw=1)
    ax.plot(EXPERTS, truth[:, j], "o", color="0.2", label="truth, centred")
    ax.plot(EXPERTS, offsets(hyper1)[:, j], "s", mfc="none", label="stage 1")
    ax.plot(EXPERTS, offsets(hyper2)[:, j], "^", label="stage 2")
    ax.set(title=rf"$\delta_{{{j},e}}$", xlabel="expert", xticks=EXPERTS)
axes[0].legend(frameon=False)
fig.tight_layout();

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
for j, ax in enumerate(axes):
    mu, sd = hyper2[f"mu{j}"], hyper2[f"sigma{j}"]
    pop_sd = float(np.sqrt(sd**2 + TAU_HAT[j] ** 2))
    true_sd = float(np.sqrt(SIGMA[j] ** 2 + TAU[j] ** 2))
    grid = np.linspace(mu - 4 * pop_sd, mu + 4 * pop_sd, 300, dtype=np.float32)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for r, run in enumerate(learned_runs(stage2)):
        mu_r, sd_r = run[f"mu{j}"], run[f"sigma{j}"]
        for e in EXPERTS:
            dens = tfd.Normal(mu_r + run[f"delta{j}_e{e}"], sd_r).prob(grid).numpy()
            ax.plot(
                grid,
                dens,
                color=colors[e - 1],
                lw=1,
                alpha=0.5,
                label=f"expert {e}" if r == 0 else None,
            )
        pop_r = float(np.sqrt(sd_r**2 + TAU_HAT[j] ** 2))
        ax.plot(
            grid,
            tfd.Normal(mu_r, pop_r).prob(grid).numpy(),
            color="0.5",
            lw=1,
            label="population, one run" if r == 0 else None,
        )

    pop = tfd.Normal(mu, pop_sd).prob(grid).numpy()
    true = tfd.Normal(MU[j], true_sd).prob(grid).numpy()
    ax.plot(grid, pop, color="black", lw=2.5, label="population")
    ax.plot(grid, true, color="black", ls="--", lw=1, label="true population")
    ax.set(title=rf"$\beta_{j}$", yticks=[])
    ax.spines[["left", "right", "top"]].set_visible(False)
axes[1].legend(frameon=False, fontsize="small")
fig.tight_layout();

