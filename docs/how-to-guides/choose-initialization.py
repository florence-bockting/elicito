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

# %% [markdown]
# # Choose an initialization method
#
# Training a parametric prior starts from a value for every hyperparameter.
# This guide shows the four ways to provide that start value, on one toy
# model, and states when to use each one.

# %% [markdown]
# ## Imports

# %%
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

# %% [markdown]
# ## The toy model
#
# $$
# \begin{align*}
#     \beta_0 &\sim \text{Normal}(\mu_0, \sigma_0) \\
#     \beta_1 &\sim \text{Normal}(\mu_1, \sigma_1) \\
#     k &\sim \text{Weibull}(k_2, \lambda_2) \\
#     y &\sim \text{Weibull}(k, \lambda),
#         \quad \lambda = \frac{\exp(\beta_0 + \beta_1 X)}{\Gamma(1 + 1/k)}
# \end{align*}
# $$
#
# The likelihood follows the `brms` defaults for the Weibull family: a log link,
# and a mean parameterization. The linear predictor is therefore the log of the
# mean of $y$, not the log of the Weibull scale.
#
# The prior of the shape $k$ is a Weibull, not a HalfNormal. A HalfNormal puts
# mass at $k \approx 0$, where $\Gamma(1 + 1/k)$ overflows and the likelihood
# scale becomes zero. The loss is then NAN even at the true hyperparameters.
#
# The six hyperparameters $\mu_0, \sigma_0, \mu_1, \sigma_1, k_2, \lambda_2$ are
# learned. We query an oracle for the quantiles of $y$ at three values of the
# predictor.


# %%
def std_predictor(N: int, quantiles: list[int]) -> Any:
    """
    Compute a standardized predictor

    Parameters
    ----------
    N
        number of observations

    quantiles
        quantiles of the predictor used for the queries

    Returns
    -------
    :
        selected values of the standardized predictor
    """
    X = tf.cast(np.arange(N), tf.float32)
    X_std = (X - tf.reduce_mean(X)) / tf.math.reduce_std(X)
    return tfp.stats.percentile(X_std, quantiles)


class ToyModel:
    """Generative model of the guide"""

    def __call__(self, prior_samples: Any, design_matrix: Any) -> dict[str, Any]:
        """
        Compute the target quantities

        Parameters
        ----------
        prior_samples
            samples from the prior distributions

        design_matrix
            selected values of the predictor

        Returns
        -------
        :
            dictionary with the target quantities
        """
        B, S = prior_samples.shape[0], prior_samples.shape[1]
        X = tf.broadcast_to(design_matrix[None, None, :], (B, S, len(design_matrix)))

        eta = tf.add(
            prior_samples[:, :, 0][:, :, None],
            tf.multiply(prior_samples[:, :, 1][:, :, None], X),
        )
        # the log link gives the mean of y; the shape k turns it into the scale
        epred = tf.exp(eta)
        k = tf.expand_dims(prior_samples[:, :, -1], -1)
        scale = epred / tf.exp(tf.math.lgamma(1.0 + 1.0 / k))
        likelihood = tfd.Weibull(concentration=k, scale=scale)
        ypred = likelihood.sample()

        return dict(
            ypred=ypred,
            epred=epred,
            prior_samples=prior_samples,
            y_X0=ypred[:, :, 0],
            y_X1=ypred[:, :, 1],
            y_X2=ypred[:, :, 2],
        )


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
        name="k",
        family=tfd.Weibull,
        hyperparams=dict(
            concentration=el.hyper("k2", lower=0),
            scale=el.hyper("lambda2", lower=0),
        ),
    ),
]

targets = [
    el.target(
        name=name,
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    )
    for name in ("y_X0", "y_X1", "y_X2")
]

ground_truth = {
    "beta0": tfd.Normal(loc=1.0, scale=0.5),
    "beta1": tfd.Normal(loc=0.3, scale=0.2),
    "k": tfd.Weibull(concentration=2.0, scale=5.0),
}


# %%
def build(initializer: Any) -> el.Elicit:
    """
    Build the eliobj; only the initializer changes between the sections

    Parameters
    ----------
    initializer
        initialization method, as returned by
        [`initializer`][elicito.elicit.initializer]

    Returns
    -------
    :
        the unfitted eliobj
    """
    return el.Elicit(
        model=el.model(
            obj=ToyModel, design_matrix=std_predictor(N=200, quantiles=[25, 50, 75])
        ),
        parameters=parameters,
        targets=targets,
        expert=el.expert.simulator(ground_truth=ground_truth, num_samples=10_000),
        optimizer=el.optimizer(
            optimizer=tf.keras.optimizers.Adam, learning_rate=0.1, clipnorm=1.0
        ),
        trainer=el.trainer(method="parametric_prior", seed=0, epochs=100, progress=0),
        initializer=initializer,
    )


def fit_and_report(label: str, initializer: Any) -> tuple[str, float, float]:
    """
    Fit the model and read the final loss

    Parameters
    ----------
    label
        name of the initialization method, used in the summary table

    initializer
        initialization method

    Returns
    -------
    :
        the label, the final loss and the runtime in seconds
    """
    eliobj = build(initializer)
    start = time.time()
    eliobj.fit()
    seconds = time.time() - start
    loss = float(np.ravel(eliobj.results.history_stats.loss.total_loss.values)[-1])
    print(f"{label}: final loss {loss:.3f} in {seconds:.1f}s")
    return label, loss, seconds


results = []

# %% [markdown]
# ## Why the start value matters
#
# Every hyperparameter is learned on the **unconstrained** scale. A scale
# parameter is transformed with the inverse softplus, so a prior scale of
# $0.5$ is the value $\text{softplus}^{-1}(0.5) = -0.43$, and the true values of
# this model are $\mu_0=1, \sigma_0=-0.43, \mu_1=0.3, \sigma_1=-1.51,
# k_2=1.85, \lambda_2=4.99$.
#
# A box that sits far from these values starts the training in a bad place. With
# a Weibull likelihood it does not even reach the first gradient step. Every
# scale collapses to $\text{softplus}(-20) \approx 0$, the shape $k$ goes to
# zero, and the draws overflow. `fit` then finds no usable candidate, and it
# raises.

# %%
try:
    fit_and_report(
        "bad box",
        el.initializer(
            method="sobol",
            iterations=32,
            distribution=el.initialization.uniform(radius=2, mean=-20),
        ),
    )
except ValueError as error:
    print(error)

# %% [markdown]
# ## Option 1: exact values
#
# Use this when you know the values. Provide them on the unconstrained scale,
# with the `forward` method of
# [`LowerBound`][elicito.utils.LowerBound].

# %%
results.append(
    fit_and_report(
        "exact values",
        el.initializer(
            hyperparams=dict(
                mu0=1.0,
                sigma0=el.utils.LowerBound(lower=0.0).forward(0.5),
                mu1=0.3,
                sigma1=el.utils.LowerBound(lower=0.0).forward(0.2),
                k2=el.utils.LowerBound(lower=0.0).forward(2.0),
                lambda2=el.utils.LowerBound(lower=0.0).forward(5.0),
            )
        ),
    )
)

# %% [markdown]
# ## Option 2: sample a box
#
# `iterations` candidates are drawn from the box, and the candidate with the
# lowest loss starts the training. The sampler is `"sobol"`, `"lhs"` or
# `"random"`.
#
# The box must be centred by you. The default centre, `mean=0` with
# `radius=1`, raises for this model: the Weibull concentration is then
# $\text{softplus}(k_2) \in [0.31, 1.31]$, many draws of $k$ fall near zero,
# and $\Gamma(1 + 1/k)$ overflows. The box below is centred at 2, which
# covers the six true values.

# %%
results.append(
    fit_and_report(
        "uniform box",
        el.initializer(
            method="sobol",
            iterations=32,
            distribution=el.initialization.uniform(radius=3, mean=2),
        ),
    )
)

# %% [markdown]
# A candidate is scored at epoch 0 by default, which does not show whether its
# trajectory is stable. `el.initializer(warmup_epochs=10, ...)` trains every
# candidate for ten epochs first, and rejects one that diverges early. It costs
# `iterations * warmup_epochs` extra epochs.

# %% [markdown]
# ## Option 3: derive the box from the expert data
#
# [`from_elicits`][elicito.initialization.from_elicits] needs no `mean` and no
# `radius`. The box is built during `fit`, from the pooled median, spread and
# 95% quantile of the elicited statistics. Each hyperparameter then gets the
# box of its role:
#
# + a **location** is centred at the pooled median;
# + a **shape**, such as the Weibull concentration, covers the natural range
#   1 to 5. A shape has no relation to the scale of the data;
# + any other **magnitude** spans from `spread / 100` up to the pooled 95%
#   quantile. That covers a small prior scale and a large data scale.

# %%
results.append(
    fit_and_report(
        "from_elicits",
        el.initializer(
            method="sobol",
            iterations=32,
            distribution=el.initialization.from_elicits(),
        ),
    )
)

# %% [markdown]
# The derived box can be inspected with
# [`build_box`][elicito.initialization.build_box], which is the function `fit`
# uses.

# %%
eliobj = build(
    el.initializer(
        method="sobol", iterations=32, distribution=el.initialization.from_elicits()
    )
)
expert_elicits, _ = el.utils.get_expert_data(
    eliobj.trainer,
    eliobj.model,
    eliobj.targets,
    eliobj.expert,
    eliobj.parameters,
    eliobj.network,
    eliobj.trainer["seed"],
)
el.initialization.build_box(
    el.initialization.from_elicits(), expert_elicits, eliobj.parameters
)

# %% [markdown]
# ## Option 4: search the start value
#
# `method="warmstart"` runs a Nelder-Mead search on the unconstrained
# hyperparameters before the first gradient step. It evaluates the same loss on
# fewer prior draws, and needs no gradient, so it cannot diverge through an
# exploding gradient. Here `iterations` is the budget, in objective
# evaluations. The centre of `distribution` is the start point of the search.
#
# Nelder-Mead converges on its own tolerances, so the search restarts from its
# best point until the budget is spent. A point whose draws overflow is never
# returned. If every point fails, the search says so, and the centre of the box
# is used.

# %%
results.append(
    fit_and_report(
        "warmstart",
        el.initializer(
            method="warmstart",
            iterations=200,
            distribution=el.initialization.from_elicits(),
        ),
    )
)

# %% [markdown]
# ## Comparison

# %%
print(f"{'method':16s} {'final loss':>10s} {'seconds':>9s}")
for label, loss, seconds in results:
    print(f"{label:16s} {loss:10.3f} {seconds:9.1f}")

# %% [markdown]
# ## Which one to choose
#
# The numbers of this run are the following.
#
# | method       | final loss | seconds |
# | :----------- | ---------: | ------: |
# | exact values |      0.234 |     1.7 |
# | uniform box  |      0.228 |    35.8 |
# | from_elicits |     43.484 |     7.1 |
# | warmstart    |      0.226 |    64.3 |
#
# + Use **exact values** when you know them. Nothing is cheaper.
# + Use **`from_elicits`** when you have no numbers. It asks nothing of you,
#   and it puts the box in the right region. Here it reaches 43.5, while the
#   default box `mean=0, radius=1` cannot even start.
# + Use **`warmstart`** on top of that box when the model can overflow, as this
#   Weibull likelihood does. It is the best of the four here, at 0.226, and the
#   slowest, because it costs one forward simulation per evaluation.
# + Use a **`uniform` box** that you centre yourself when you know the order of
#   magnitude of the hyperparameters. Add `warmup_epochs` to reject a candidate
#   that diverges in the first epochs.
