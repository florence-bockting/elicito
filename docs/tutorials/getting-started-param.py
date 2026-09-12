# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Getting started with parametric priors
#
# This tutorial learns a **parametric prior** from expert knowledge. The
# expert does not state the prior. The expert states what they expect to
# observe, and `elicito` finds the prior that reproduces these expectations.
#
# **You learn how to:**
#
# 1. write a generative model,
# 2. give each model parameter a prior family with hyperparameters,
# 3. define the target quantities that the expert is asked about,
# 4. learn the hyperparameters with CMA-ES,
# 5. check the result with the diagnostic plots.
#
# Two variants follow at the end: a hyperparameter that two priors share,
# and expert data that you enter by hand.

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
# We use a linear regression with one predictor $X$:
#
# $$
# \begin{align*}
#     \beta_0 &\sim \text{Normal}(\mu_0, \sigma_0) \\
#     \beta_1 &\sim \text{Normal}(\mu_1, \sigma_1) \\
#     \sigma &\sim \text{HalfNormal}(\sigma_2) \\
#     \mu &= \beta_0 + \beta_1 X \\
#     y_{pred} &\sim \text{Normal}(\mu, \sigma)
# \end{align*}
# $$
#
# ### Design matrix
#
# The predictor takes the values $0, 1, \dots, N-1$. We divide it by its
# standard deviation. We then ask the expert about the outcome at three
# values of the predictor: its 25th ($X_0$), 50th ($X_1$) and 75th ($X_2$)
# percentile.


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


X_design(N=30, quantiles=[25, 50, 75])

# %% [markdown]
# ### Generative model
#
# The generative model is a class with a `__call__` method. It receives the
# prior samples and the design matrix. It returns a dictionary with every
# quantity that a target can use.
#
# The prior samples have the shape `(B, num_samples, number of parameters)`.
# The last column is $\sigma$. The other columns are the regression
# coefficients.


# %%
class ToyModel:
    """
    Generative model of the linear regression
    """

    def __call__(self, prior_samples: Any, design_matrix: Any) -> dict[str, Any]:
        """
        Compute the target quantities from the generative model

        Parameters
        ----------
        prior_samples
            prior samples

        design_matrix
            design matrix

        Returns
        -------
        :
            dictionary with the target quantities
        """
        # linear predictor
        epred = tf.matmul(prior_samples[:, :, :-1], design_matrix, transpose_b=True)

        # data-generating model
        likelihood = tfd.Normal(
            loc=epred, scale=tf.expand_dims(prior_samples[:, :, -1], -1)
        )
        # prior predictive distribution
        ypred = likelihood.sample()

        # outcome at the three selected predictor values
        y_X0, y_X1, y_X2 = (ypred[:, :, 0], ypred[:, :, 1], ypred[:, :, 2])

        return dict(y_X0=y_X0, y_X1=y_X1, y_X2=y_X2, ypred=ypred, epred=epred)


# %%
model = el.model(obj=ToyModel, design_matrix=X_design(N=30, quantiles=[25, 50, 75]))

# %% [markdown]
# ## Model parameters
#
# Each model parameter gets a prior family. Each hyperparameter of that
# family is a value that `elicito` learns.
#
# | parameter | prior family | hyperparameters | constraint |
# | :-- | :-- | :-- | :-- |
# | intercept $\beta_0$ | Normal | $\mu_0$, $\sigma_0$ | $\sigma_0 > 0$ |
# | slope $\beta_1$ | Normal | $\mu_1$, $\sigma_1$ | $\sigma_1 > 0$ |
# | noise $\sigma$ | HalfNormal | $\sigma_2$ | $\sigma_2 > 0$ |
#
# `lower=0` constrains a scale hyperparameter to positive values.

# %%
parameters = [
    el.parameter(
        name="beta0",
        family=tfd.Normal,
        hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)),
    ),
    el.parameter(
        name="beta1",
        family=tfd.Normal,
        hyperparams=dict(loc=el.hyper("mu1"), scale=el.hyper("sigma1", lower=0)),
    ),
    el.parameter(
        name="sigma",
        family=tfd.HalfNormal,
        hyperparams=dict(scale=el.hyper("sigma2", lower=0)),
    ),
]

# titles for the plots, in the order of the hyperparameters above
TITLES = [r"$\mu_0$", r"$\sigma_0$", r"$\mu_1$", r"$\sigma_1$", r"$\sigma_2$"]

# %% [markdown]
# ## Target quantities and elicitation techniques
#
# A **target quantity** is a quantity that the expert can judge. The
# **elicitation technique** is the question that we ask the expert about it.
#
# | target quantity | elicitation technique | discrepancy measure | weight |
# | :-- | :-- | :-- | --: |
# | $y \mid X_0$, $y \mid X_1$, $y \mid X_2$ | quantiles $Q_p$, $p = 5, 25, 50, 75, 95$ | MMD² with energy kernel | 1.0 |
# | $R^2$ | quantiles $Q_p$, $p = 5, 50, 95$ | L2 loss | 0.5 |
#
# The model does not return $R^2$. The function `custom_r2` computes it from
# the model output. `elicito` matches its arguments to the keys of the model
# output by name.


# %%
def custom_r2(ypred: Any, epred: Any) -> Any:
    """Compute the coefficient of determination"""
    var_epred = tf.math.reduce_variance(epred, -1)
    # variance of the difference between ypred and epred
    var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
    var_total = var_epred + var_diff
    # variance of the linear predictor divided by the total variance
    return tf.divide(var_epred, var_total)


targets = [
    el.target(
        name="y_X0",
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    ),
    el.target(
        name="y_X1",
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    ),
    el.target(
        name="y_X2",
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    ),
    el.target(
        name="R2",
        query=el.queries.quantiles((0.05, 0.50, 0.95)),
        loss=el.losses.L2,
        weight=0.5,
        target_method=custom_r2,
    ),
]

# %% [markdown]
# ## Expert elicitation
#
# We do not ask a real expert. We define a ground truth, the **oracle**, and
# simulate the answers that the oracle gives. Because we know the true
# prior, we can later see whether `elicito` recovers it.

# %%
ground_truth = {
    "beta0": tfd.Normal(loc=5, scale=1),
    "beta1": tfd.Normal(loc=2, scale=1),
    "sigma": tfd.HalfNormal(scale=7.0),
}

expert = el.expert.simulator(ground_truth=ground_truth, num_samples=10_000)

# %% [markdown]
# ## Training
#
# We learn the hyperparameters with **CMA-ES**, a global search that needs
# no gradient.
#
# - `el.optimizer(optimizer="cmaes")` replaces the gradient descent by the
#   CMA-ES search.
# - CMA-ES needs no `initializer`. The search starts at 0 on the unconstrained
#   scale, and `elicito` chooses the first step size. Set it with
#   `el.optimizer(sigma0=...)`.
# - `epochs` is the budget in **forward simulations**, not the number of
#   gradient steps. One generation of CMA-ES tests several candidates, so the
#   loss curve has one point per generation.
# - `el.utils.parallel()` runs four replications with different seeds. The
#   replications show whether the result depends on the seed.
# - `kappa` in `el.trainer()` sets the weight of a spread penalty in the loss.
#   The penalty keeps a prior from collapsing to a point mass. `kappa=0`
#   turns it off.
#
# > **Note:** CMA-ES needs the optional dependency `cma`. Install it with
# > `pip install "elicito[cma]"`.

# %%
eliobj = el.Elicit(
    model=model,
    parameters=parameters,
    targets=targets,
    expert=expert,
    optimizer=el.optimizer(optimizer="cmaes"),
    trainer=el.trainer(
        method="parametric_prior", seed=2025, epochs=800, progress=1, kappa=0.1
    ),
)

# %% [markdown]
# Print the `eliobj` to see a summary of the settings.

# %%
eliobj

# %%
eliobj.fit(parallel=el.utils.parallel(runs=4))

# %% [markdown]
# The results keep the learned hyperparameters, the loss and the expert
# data. `sample()` computes the prior samples and the model simulations from
# them. We compute them once and give them to every plot.

# %%
eliobj.results

# %%
samples = eliobj.sample()

# %% [markdown]
# ## Results
#
# ### Convergence of the loss
#
# The loss must decrease and then stay flat. Each line is one replication.

# %%
el.plots.loss(eliobj, figsize=(7, 3));

# %% [markdown]
# ### Convergence of the hyperparameters
#
# Each hyperparameter must settle on one value. When the replications settle
# on the same value, the result does not depend on the seed.

# %%
el.plots.hyperparameter(eliobj, titles=TITLES, cols=5, figsize=(11, 2.4));

# %% [markdown]
# ### Expert expectations
#
# Each point compares one elicited quantile of the expert with the same
# quantile of the learned model. The points lie on the diagonal when the
# learned prior reproduces the expert data.

# %%
el.plots.elicits(eliobj, cols=4, figsize=(9, 2.6), samples=samples);

# %% [markdown]
# ### Learned prior distributions

# %%
el.plots.prior_marginals(eliobj, cols=3, figsize=(8, 2.4), samples=samples);

# %% [markdown]
# ### Prior predictive distribution of $R^2$

# %%
el.plots.priorpredictive(eliobj, target="R2", samples=samples);

# %% [markdown]
# ### Averaged prior over the replications
#
# The four replications give four priors. The plot averages them. The
# weight of a replication comes from its mean loss over the last 30 history
# points.

# %%
el.plots.prior_averaging(eliobj, samples=samples);

# %% [markdown]
# ## Variant A: a shared scale hyperparameter
#
# The intercept and the slope can share one scale hyperparameter. Give both
# priors the same name, and set `shared=True`. The model then has four
# hyperparameters instead of five.

# %%
parameters_shared = [
    el.parameter(
        name="beta0",
        family=tfd.Normal,
        hyperparams=dict(
            loc=el.hyper("mu0"), scale=el.hyper("sigma1", lower=0, shared=True)
        ),
    ),
    el.parameter(
        name="beta1",
        family=tfd.Normal,
        hyperparams=dict(
            loc=el.hyper("mu1"), scale=el.hyper("sigma1", lower=0, shared=True)
        ),
    ),
    el.parameter(
        name="sigma",
        family=tfd.HalfNormal,
        hyperparams=dict(scale=el.hyper("sigma2", lower=0)),
    ),
]

eliobj_shared = el.Elicit(
    model=model,
    parameters=parameters_shared,
    targets=targets,
    expert=expert,
    optimizer=el.optimizer(optimizer="cmaes"),
    trainer=el.trainer(method="parametric_prior", seed=2025, epochs=800, progress=0),
)

eliobj_shared.fit()
samples_shared = eliobj_shared.sample()

# %% [markdown]
# ### Results

# %%
el.plots.loss(eliobj_shared, figsize=(7, 3));

# %%
el.plots.hyperparameter(eliobj_shared, cols=4, figsize=(9, 2.4));

# %%
el.plots.elicits(eliobj_shared, cols=4, figsize=(9, 2.6), samples=samples_shared);

# %%
el.plots.prior_marginals(
    eliobj_shared, cols=3, figsize=(8, 2.4), samples=samples_shared
);

# %% [markdown]
# ## Variant B: expert data as input
#
# With a real expert, you enter the elicited statistics by hand. Each key
# combines the elicitation technique and the target name.
# `el.utils.get_expert_datformat(targets)` shows the keys that the targets
# expect.

# %%
el.utils.get_expert_datformat(targets)

# %%
expert_dat = {
    "quantiles_y_X0": [-12.5, -0.6, 3.3, 7.1, 19.1],
    "quantiles_y_X1": [-11.2, 1.5, 5.0, 8.8, 20.4],
    "quantiles_y_X2": [-9.3, 3.1, 6.8, 10.5, 23.3],
    "quantiles_R2": [0.001, 0.09, 0.96],
}

eliobj_dat = el.Elicit(
    model=model,
    parameters=parameters,
    targets=targets,
    expert=el.expert.data(dat=expert_dat),
    optimizer=el.optimizer(optimizer="cmaes"),
    trainer=el.trainer(method="parametric_prior", seed=2025, epochs=800, progress=0),
)

eliobj_dat.fit()
samples_dat = eliobj_dat.sample()

# %% [markdown]
# ### Results

# %%
el.plots.loss(eliobj_dat, figsize=(7, 3));

# %%
el.plots.hyperparameter(eliobj_dat, titles=TITLES, cols=5, figsize=(11, 2.4));

# %%
el.plots.elicits(eliobj_dat, cols=4, figsize=(9, 2.6), samples=samples_dat);

# %%
el.plots.prior_marginals(eliobj_dat, cols=3, figsize=(8, 2.4), samples=samples_dat);
