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
# This guide shows the four ways to provide that start value, and states when
# to use each one.
#
# Part 1 uses a model with two hyperparameters. Two of them fit in a plane, so
# every method can be watched on the loss surface itself. Part 2 repeats the
# comparison on a model where a bad start value does not merely cost epochs,
# but stops the training before its first step.

# %% [markdown]
# ## Imports

# %%
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

# %% [markdown]
# ## Part 1: a model with two hyperparameters
#
# $$
# \begin{align*}
#     \mu &\sim \text{Normal}(\mu_0, \sigma_0) \\
#     y &\sim \text{Normal}(\mu, 1)
# \end{align*}
# $$
#
# Two hyperparameters are learned here, $\mu_0$ and $\sigma_0$. The oracle is
# asked for five quantiles of $y$. Two hyperparameters fit in a plane, so the
# loss of every possible start value can be drawn as a surface. Every method of
# this guide is then watched on that surface.
#
# The true values are $\mu_0 = 1$ and $\sigma_0 = 2$. A scale is learned on the
# unconstrained scale, so the true value of $\sigma_0$ in every figure below is
# $\text{softplus}^{-1}(2) = 1.85$.


# %%
class NormalModel:
    """Generative model of part 1"""

    def __call__(self, prior_samples: Any, N: int) -> dict[str, Any]:
        """
        Compute the target quantities

        Parameters
        ----------
        prior_samples
            samples from the prior distributions

        N
            number of observations

        Returns
        -------
        :
            dictionary with the target quantities
        """
        mu = prior_samples[:, :, 0][:, :, None]
        ypred = tfd.Normal(
            loc=tf.broadcast_to(mu, (*mu.shape[:2], N)), scale=1.0
        ).sample()

        return dict(ypred=ypred, prior_samples=prior_samples, y=ypred[:, :, 0])


# %%
parameters = [
    el.parameter(
        name="mu",
        family=tfd.Normal,
        hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)),
    )
]

targets = [
    el.target(
        name="y",
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    )
]

ground_truth = {"mu": tfd.Normal(loc=1.0, scale=2.0)}

forward = el.utils.LowerBound(lower=0.0).forward
# the true values, on the scale of the figures
TRUTH = (1.0, float(forward(2.0)))


# %%
def build(
    initializer: Any, epochs: int = 300, B: int = 128, num_samples: int = 200
) -> el.Elicit:
    """
    Build the eliobj; only the initializer changes between the sections

    Parameters
    ----------
    initializer
        initialization method, as returned by
        [`initializer`][elicito.elicit.initializer]

    epochs
        number of training epochs

    B
        batch size

    num_samples
        number of prior draws per batch

    Returns
    -------
    :
        the unfitted eliobj
    """
    return el.Elicit(
        model=el.model(obj=NormalModel, N=50),
        parameters=parameters,
        targets=targets,
        expert=el.expert.simulator(ground_truth=ground_truth, num_samples=10_000),
        optimizer=el.optimizer(optimizer=tf.keras.optimizers.Adam, learning_rate=0.1),
        trainer=el.trainer(
            method="parametric_prior",
            seed=0,
            epochs=epochs,
            B=B,
            num_samples=num_samples,
            progress=0,
        ),
        initializer=initializer,
    )


def fit_and_report(
    label: str,
    initializer: Any,
    epochs: int = 300,
    B: int = 128,
    num_samples: int = 200,
) -> float:
    """
    Fit the model, store it, and report the final loss

    Parameters
    ----------
    label
        name of the initialization method, used in the summary table

    initializer
        initialization method

    epochs
        number of training epochs

    B
        batch size

    num_samples
        number of prior draws per batch

    Returns
    -------
    :
        the final loss
    """
    eliobj = build(initializer, epochs, B, num_samples)
    start = time.time()
    eliobj.fit()
    seconds = time.time() - start
    loss = float(np.ravel(eliobj.results.history_stats.loss.total_loss.values)[-1])
    results.append((label, loss, seconds))
    fits[label] = eliobj
    return loss


results: list[tuple[str, float, float]] = []
fits: dict[str, el.Elicit] = {}

# %% [markdown]
# ### The loss surface
#
# [`score`][elicito.warmstart.score] computes the loss of one set of
# hyperparameter values, with no training at all. A grid of these values
# therefore draws the whole surface. The expert data is needed first, and
# [`get_expert_data`][elicito.utils.get_expert_data] provides it.
#
# The grid below costs 1600 evaluations, which is about 80 seconds.

# %%
eliobj = build(el.initializer(hyperparams=dict(mu0=1.0, sigma0=0.0)))
expert_elicits, _ = el.utils.get_expert_data(
    eliobj.trainer,
    eliobj.model,
    eliobj.targets,
    eliobj.expert,
    eliobj.parameters,
    eliobj.network,
    eliobj.trainer["seed"],
)

mu0_grid = np.linspace(-6.0, 6.0, 40)
sigma0_grid = np.linspace(-6.0, 4.5, 40)
surface = np.empty((len(sigma0_grid), len(mu0_grid)))
for i, sigma0_value in enumerate(sigma0_grid):
    for j, mu0_value in enumerate(mu0_grid):
        surface[i, j] = el.warmstart.score(
            hyperparams=dict(mu0=mu0_value, sigma0=sigma0_value),
            expert_elicited_statistics=expert_elicits,
            parameters=eliobj.parameters,
            trainer=eliobj.trainer,
            model=eliobj.model,
            targets=eliobj.targets,
            expert=eliobj.expert,
            seed=0,
        )


# %% [markdown]
# One figure follows every method below. It shows the surface, the candidates
# the method drew, the start value it picked, and the path the training then
# took. The code that draws it is not part of the guide, so it is hidden.


# %% tags=["remove_input"]
def landscape(
    title: str,
    candidates: Any = None,
    start: Any = None,
    path: Any = None,
    search: Any = None,
) -> None:
    """
    Draw the loss surface, and what one initialization method did on it

    Parameters
    ----------
    title
        title of the figure

    candidates
        start values the method drew, shape (candidates, 2)

    start
        the start value the method picked

    path
        the training path, shape (epochs, 2)

    search
        the points a search evaluated, shape (evaluations, 2)
    """
    fig, ax = plt.subplots(figsize=(6.5, 5), layout="tight")
    drawn = ax.contourf(
        mu0_grid, sigma0_grid, np.log10(surface), levels=30, cmap="viridis"
    )
    if candidates is not None:
        ax.scatter(
            candidates[:, 0],
            candidates[:, 1],
            s=25,
            color="white",
            edgecolor="0.3",
            linewidth=0.5,
            label="candidates",
        )
    if search is not None:
        ax.plot(
            search[:, 0],
            search[:, 1],
            color="white",
            linewidth=1.0,
            marker="o",
            markersize=3,
            label="search",
        )
    if path is not None:
        ax.plot(
            path[:, 0], path[:, 1], color="crimson", linewidth=1.5, label="training"
        )
        ax.scatter(path[-1, 0], path[-1, 1], marker="o", s=60, color="crimson")
    if start is not None:
        ax.scatter(start[0], start[1], marker="o", s=70, color="black", label="start")
    ax.scatter(
        TRUTH[0],
        TRUTH[1],
        marker="*",
        s=260,
        color="white",
        edgecolor="black",
        linewidth=0.8,
        label="true",
    )
    ax.set_xlim(mu0_grid[0], mu0_grid[-1])
    ax.set_ylim(sigma0_grid[0], sigma0_grid[-1])
    ax.set_xlabel("mu0")
    ax.set_ylabel("sigma0 (unconstrained)")
    ax.set_title(title)
    ax.legend(fontsize="small", loc="lower right")
    fig.colorbar(drawn, ax=ax, label="log10 loss")
    plt.show()


def candidates_of(label: str) -> tuple[Any, Any]:
    """
    Read the candidates of a fit, and the one it picked

    Parameters
    ----------
    label
        name used in [`fit_and_report`][fit_and_report]

    Returns
    -------
    :
        the candidates and the picked start value
    """
    init = fits[label].results.initialization.sel(replication=0)
    names = [str(n) for n in init.hyperparameters.coords["hyperparameter"].values]
    drawn = np.asarray(init.hyperparameters.values)[
        :, [names.index("mu0"), names.index("sigma0")]
    ]
    losses = np.ravel(np.asarray(init.loss.values))
    return drawn, drawn[int(np.nanargmin(losses))]


def training_path(label: str) -> Any:
    """
    Read the training path of a fit, on the unconstrained scale

    Parameters
    ----------
    label
        name used in [`fit_and_report`][fit_and_report]

    Returns
    -------
    :
        the path, shape (epochs, 2)
    """
    history = fits[label].results.history_stats.hyperparameter.sel(replication=0)
    return np.column_stack(
        [
            np.ravel(history["mu0"].values),
            np.asarray(forward(np.ravel(history["sigma0"].values))),
        ]
    )


# %% [markdown]
# The surface has one minimum, and it sits at the true values. Two features of
# it explain everything below.
#
# + The valley in $\mu_0$ is narrow. A start value that is wrong in $\mu_0$
#   still has a gradient that points to the valley.
# + Below $\sigma_0 \approx 0$ the surface is flat. There
#   $\text{softplus}(\sigma_0)$ is almost zero, so the prior of $\mu$ is almost
#   a point, and the noise of 1 hides the difference. A start value in that
#   region still has a gradient in $\sigma_0$, but a small one. The next
#   sections measure it against the error of the loss estimate.

# %% tags=["remove_input"]
landscape("the loss surface")

# %% [markdown]
# ### Why the start value matters
#
# A box centred at $-4$ starts the training on the flat floor. The training
# then repairs $\mu_0$, which has a gradient, and leaves $\sigma_0$ where it
# was. This box gets 600 epochs, twice the number every method below gets, and
# the loss is still 0.467, against 0.079 for the others.
#
# The model has no local minimum, so this is not a trap in the usual sense.

# %%
loss = fit_and_report(
    "far box",
    el.initializer(
        method="sobol",
        iterations=32,
        distribution=el.initialization.uniform(radius=1, mean=-4),
    ),
    epochs=600,
)

# %% tags=["remove_input"]
drawn, start = candidates_of("far box")
landscape(
    f"far box: final loss {loss:.3f}",
    candidates=drawn,
    start=start,
    path=training_path("far box"),
)

# %% [markdown]
# ### The gradient is not absent, it is outvoted
#
# The floor is often called a region without a gradient. That is not what
# happens here. The gradient in $\sigma_0$ exists, and it is smaller than the
# error of the loss estimate.
#
# + Move $\sigma_0$ a full unit on the floor, from $-4.5$ to $-3.5$. The loss
#   changes by about $4 \cdot 10^{-4}$.
# + One sample of `B=128` and `num_samples=200` estimates that loss with an
#   error of about $2.5 \cdot 10^{-3}$.
#
# The error is six times the signal, so the estimate says almost nothing about
# where $\sigma_0$ should go. Two effects make the signal small. The predictive
# standard deviation is $\sqrt{\text{scale}^2 + 1}$, so a scale of $0.011$
# disappears behind the noise of 1. And softplus saturates, so
# $d\,\text{scale} / d\sigma_0 = \text{sigmoid}(-4.5) = 0.011$ as well.
#
# The error does not average out over the epochs. Every epoch draws with the
# same seed, so the sample is fixed. It defines one slightly wrong surface, and
# the training descends that surface faithfully. Only a new seed, or more
# draws, changes it.
#
# `B` and `num_samples` set the size of that sample, and the error falls as
# $1 / \sqrt{B \cdot \text{num\_samples}}$. Four times the batch and four
# times the draws halve the error. Here that is enough.

# %%
loss = fit_and_report(
    "far box, more draws",
    el.initializer(
        method="sobol",
        iterations=32,
        distribution=el.initialization.uniform(radius=1, mean=-4),
    ),
    epochs=600,
    B=512,
    num_samples=800,
)

# %% tags=["remove_input"]
drawn, start = candidates_of("far box, more draws")
landscape(
    f"far box, B=512 and num_samples=800: final loss {loss:.3f}",
    candidates=drawn,
    start=start,
    path=training_path("far box, more draws"),
)

# %% [markdown]
# The same box, the same 600 epochs, and the training now leaves the floor and
# reaches the true $\sigma_0$.
#
# Read the two losses with care. The MMD estimate carries a sample-size bias,
# so a larger sample lowers the loss on its own. Compare each run with a run of
# the same `B` and `num_samples`, never across the two.
#
# Note also what this costs. The larger sample needs 226 ms per epoch against
# 96 ms, and it does not remove the need for a start value: every method below
# reaches the minimum with the small sample, and in a fifth of the epochs.

# %% [markdown]
# ### Option 1: exact values
#
# Use this when you know the values. Provide them on the unconstrained scale,
# with the `forward` method of [`LowerBound`][elicito.utils.LowerBound].
#
# The start value is the star, so the training only has to stay there.

# %%
loss = fit_and_report(
    "exact values",
    el.initializer(hyperparams=dict(mu0=1.0, sigma0=forward(2.0))),
)

# %% tags=["remove_input"]
landscape(
    f"exact values: final loss {loss:.3f}",
    start=TRUTH,
    path=training_path("exact values"),
)

# %% [markdown]
# ### Option 2: sample a box
#
# `iterations` candidates are drawn from the box, and the candidate with the
# lowest loss starts the training. The sampler is `"sobol"`, `"lhs"` or
# `"random"`.
#
# The box must be centred by you. The default centre, `mean=0` with
# `radius=1`, does not contain the true values. The figure shows what the
# scoring buys: the picked candidate is the corner of the box that is nearest
# to the minimum, so the training starts on the slope that leads there.

# %%
loss = fit_and_report(
    "uniform box",
    el.initializer(
        method="sobol",
        iterations=32,
        distribution=el.initialization.uniform(radius=1, mean=0),
    ),
)

# %% tags=["remove_input"]
drawn, start = candidates_of("uniform box")
landscape(
    f"uniform box: final loss {loss:.3f}",
    candidates=drawn,
    start=start,
    path=training_path("uniform box"),
)

# %% [markdown]
# A candidate is scored at epoch 0 by default, which does not show whether its
# trajectory is stable. `el.initializer(warmup_epochs=10, ...)` trains every
# candidate for ten epochs first, and rejects one that diverges early. It costs
# `iterations * warmup_epochs` extra epochs.

# %% [markdown]
# ### Option 3: derive the box from the expert data
#
# [`from_elicits`][elicito.initialization.from_elicits] needs no `mean` and no
# `radius`. The box is built during `fit`, from the pooled median, spread and
# 95% quantile of the elicited statistics. Each hyperparameter then gets the
# box of its role:
#
# + a **location** is centred at the pooled median;
# + a **shape**, such as a Weibull concentration, covers the natural range
#   1 to 5. A shape has no relation to the scale of the data;
# + any other **magnitude** spans from `spread / 100` up to the pooled 95%
#   quantile. That covers a small prior scale and a large data scale.
#
# Here the box is far wider than the one you would centre yourself, and that is
# the point: it needs no number from you, and it still contains the minimum.

# %%
loss = fit_and_report(
    "from_elicits",
    el.initializer(
        method="sobol",
        iterations=32,
        distribution=el.initialization.from_elicits(),
    ),
)

# %% tags=["remove_input"]
drawn, start = candidates_of("from_elicits")
landscape(
    f"from_elicits: final loss {loss:.3f}",
    candidates=drawn,
    start=start,
    path=training_path("from_elicits"),
)

# %% [markdown]
# The derived box can be inspected with
# [`build_box`][elicito.initialization.build_box], which is the function `fit`
# uses.

# %%
el.initialization.build_box(
    el.initialization.from_elicits(), expert_elicits, parameters
)

# %% [markdown]
# ### Option 4: search the start value
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
#
# The search is not stored, so the figure below records it: wrap
# [`score`][elicito.warmstart.score], the function the search calls, for the
# time of the fit. The white path is the search, and the training after it is
# so short that the red path is a dot.

# %% tags=["remove_input"]
visited: list[Any] = []
scored: list[float] = []
original_score = el.warmstart.score


def recording_score(**kwargs: Any) -> float:
    """Record one evaluation of the search, then score it as usual"""
    value = original_score(**kwargs)
    visited.append(list(kwargs["hyperparams"].values()))
    scored.append(value)
    return value


el.warmstart.score = recording_score

# %%
loss = fit_and_report(
    "warmstart",
    el.initializer(
        method="warmstart",
        iterations=50,
        distribution=el.initialization.from_elicits(),
    ),
)

# %% tags=["remove_input"]
el.warmstart.score = original_score

path = training_path("warmstart")
landscape(
    f"warmstart: final loss {loss:.3f}",
    start=path[0],
    path=path,
    search=np.asarray(visited),
)

# %% [markdown]
# ### What part 1 showed
#
# Every method except the far box reaches the same loss. On a surface with one
# minimum, and with a signal that stands above the sampling error where the box
# sits, the start value only decides how many epochs the training needs.
#
# The two far-box rows are not comparable with the rest, and not with each
# other: the second one uses a larger sample, which lowers the loss on its own.

# %%
print(f"{'method':21s} {'final loss':>10s} {'seconds':>9s}")
for label, final_loss, seconds in results:
    print(f"{label:21s} {final_loss:10.3f} {seconds:9.1f}")

# %% [markdown]
# ### When a hyperparameter has a small effect
#
# The far box is one case of a rule that holds beyond initialization. A
# hyperparameter is learned only while its effect on the loss is larger than
# the error of the loss estimate. `B` and `num_samples` shrink that error as
# $1/\sqrt{B \cdot \text{num\_samples}}$, so an effect half as large costs
# four times the sample.
#
# Before you pay that, ask why the effect is small.
#
# + **The parameterization hides it.** At $\sigma_0 = -4.5$ the softplus
#   contributes a factor $0.011$ for no other reason than saturation. A start
#   value is far cheaper than a larger sample: the warm start above reaches
#   0.079 in about 30 seconds.
# + **The elicited statistics do not respond to it.** Then a larger sample
#   sharpens a direction the data cannot pin down. Part 2 shows this: $k_2$
#   ends near 13 against a true 2, and that run still has the lowest loss of
#   the four. The fix is a query that responds, for example on the parameter
#   itself.
#
# Raise `B` and `num_samples` when the effect is real but buried. Change the
# start value, or the query, when the effect is not there to begin with.

# %% [markdown]
# The next part uses a model where that is no longer true. A bad start value
# there does not merely cost epochs. It produces draws that overflow, and the
# training cannot start at all.

# %% [markdown]
# ## Part 2: a model that can overflow
#
# The surface of part 1 had a gradient everywhere, so a poor start value
# only cost epochs. This part uses a model where a start value can be
# unusable: the draws overflow, the loss is NAN, and there is nothing to
# descend. Six hyperparameters are learned, so no surface can be drawn. The
# figures hold four of them at their true values, and draw the loss of the
# other two.
#
# ### The model
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
def build(initializer: Any, epochs: int = 100) -> el.Elicit:
    """
    Build the eliobj; only the initializer changes between the sections

    Parameters
    ----------
    initializer
        initialization method, as returned by
        [`initializer`][elicito.elicit.initializer]

    epochs
        number of training epochs

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
        trainer=el.trainer(
            method="parametric_prior", seed=0, epochs=epochs, progress=0
        ),
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
    fits[label] = eliobj
    return label, loss, seconds


results = []
fits = {}

# The true values. The training records a hyperparameter on its natural scale,
# so the convergence figure uses these. The two search figures work on the
# unconstrained scale, so they use `truth_unconstrained` below.
truth = dict(mu0=1.0, sigma0=0.5, mu1=0.3, sigma1=0.2, k2=2.0, lambda2=5.0)

forward = el.utils.LowerBound(lower=0.0).forward
truth_unconstrained = dict(
    mu0=1.0,
    sigma0=float(forward(0.5)),
    mu1=0.3,
    sigma1=float(forward(0.2)),
    k2=float(forward(2.0)),
    lambda2=float(forward(5.0)),
)

# the figures cut the loss along two of the fifteen pairs of hyperparameters
projections = [("mu0", "sigma0"), ("k2", "lambda2")]

# %% [markdown]
# ### A box that stops the training
#
# The true values on the unconstrained scale are $\mu_0=1, \sigma_0=-0.43,
# \mu_1=0.3, \sigma_1=-1.51, k_2=1.85, \lambda_2=4.99$.
#
# A box centred at $-20$ contains none of them. Every scale collapses to
# $\text{softplus}(-20) \approx 0$, the shape $k$ goes to zero, and
# $\Gamma(1 + 1/k)$ overflows. No candidate of the box is usable, so `fit`
# raises instead of starting.

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
# ### Option 1: exact values
#
# Six values now, instead of two.

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
# ### Option 2: sample a box
#
# The default centre, `mean=0` with `radius=1`, already raises here: the
# Weibull concentration is then $\text{softplus}(k_2) \in [0.31, 1.31]$, many
# draws of $k$ fall near zero, and $\Gamma(1 + 1/k)$ overflows. The box below
# is centred at 2, which covers the six true values.

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
# ### Option 3: derive the box from the expert data
#
# The roles of part 1 decide the result here. The shape $k_2$ gets the natural
# range 1 to 5, and the scale $\lambda_2$ gets the range of the data. One box
# for both would put candidates where the draws overflow.

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
# ### Option 4: search the start value
#
# This is the model the search is for. It needs no gradient, and it never
# returns a point whose draws overflow. The budget is 100 evaluations here,
# against 50 in part 1, because six hyperparameters are searched instead of
# two. A budget that does not grow with the number of hyperparameters leaves
# the search short: at 50 evaluations it stops at a start value of loss 9.9,
# and the training then needs 140 epochs to recover.
#
# The `score` wrapper of part 1 records the path again.

# %% tags=["remove_input"]
visited = []
scored = []
el.warmstart.score = recording_score

# %%
results.append(
    fit_and_report(
        "warmstart",
        el.initializer(
            method="warmstart",
            iterations=100,
            distribution=el.initialization.from_elicits(),
        ),
    )
)

# %% tags=["remove_input"]
el.warmstart.score = original_score

# %% [markdown]
# ### Comparison
#
# Read this table together with the log above. Two of the four runs stopped
# early: `exact values` after 5 steps, and `from_elicits` after 8. The loss of
# those two is the last finite loss, not the loss after 100 epochs. Even the
# true hyperparameters do not give a stable trajectory for this model.

# %%
print(f"{'method':21s} {'final loss':>10s} {'seconds':>9s}")
for label, loss, seconds in results:
    print(f"{label:21s} {loss:10.3f} {seconds:9.1f}")

# %% [markdown]
# ### Watch the search
#
# Part 1 drew every method on the loss surface. Six hyperparameters have no
# surface, so hold four of them at their true values. The loss of the other two
# is then a plane again, and it is drawn exactly as in part 1. The two slices
# are $(\mu_0, \sigma_0)$ and $(k_2, \lambda_2)$.
#
# A slice is a cut through the surface, not the surface itself. A point that
# lies in a dark region of the cut can still be poor in the four held
# directions. The cut does show what part 1 had no example of: the grey region,
# where the draws overflow and there is no loss to descend.
#
# Each slice costs 900 evaluations, which is about 80 seconds.

# %% tags=["remove_input"]
init = fits["from_elicits"].results.initialization.sel(replication=0)
names = [str(n) for n in init.hyperparameters.coords["hyperparameter"].values]
candidates = np.asarray(init.hyperparameters.values)
losses = np.ravel(np.asarray(init.loss.values))
failed = ~np.isfinite(losses)

search_path = np.asarray(visited)
scores = np.asarray(scored)
overflowed = scores >= el.warmstart.PENALTY

# `score` needs the expert data of this model
eliobj = build(el.initializer(hyperparams=truth_unconstrained))
expert_elicits, _ = el.utils.get_expert_data(
    eliobj.trainer,
    eliobj.model,
    eliobj.targets,
    eliobj.expert,
    eliobj.parameters,
    eliobj.network,
    eliobj.trainer["seed"],
)


def axis_of(name: str, points: int = 30, pad: float = 0.1) -> Any:
    """
    Build the axis of one hyperparameter, wide enough for every drawn point

    Parameters
    ----------
    name
        name of the hyperparameter

    points
        number of grid points

    pad
        margin added on both sides, as a share of the range

    Returns
    -------
    :
        the grid points of the axis
    """
    drawn = np.concatenate(
        [
            candidates[:, names.index(name)],
            search_path[:, names.index(name)],
            [truth_unconstrained[name]],
        ]
    )
    low, high = float(np.min(drawn)), float(np.max(drawn))
    margin = pad * (high - low)
    return np.linspace(low - margin, high + margin, points)


def slice_of(pair: tuple[str, str]) -> Any:
    """
    Compute the loss on the plane of two hyperparameters

    The other four hyperparameters are held at their true values. A point whose
    draws overflow gets no loss, and is returned as NAN.

    Parameters
    ----------
    pair
        names of the two hyperparameters that vary

    Returns
    -------
    :
        the loss, shape (points of y, points of x)
    """
    x_name, y_name = pair
    values = dict(truth_unconstrained)
    grid = np.empty((len(axes_of[y_name]), len(axes_of[x_name])))
    for i, y_value in enumerate(axes_of[y_name]):
        for j, x_value in enumerate(axes_of[x_name]):
            values[x_name] = float(x_value)
            values[y_name] = float(y_value)
            grid[i, j] = el.warmstart.score(
                hyperparams=values,
                expert_elicited_statistics=expert_elicits,
                parameters=eliobj.parameters,
                trainer=eliobj.trainer,
                model=eliobj.model,
                targets=eliobj.targets,
                expert=eliobj.expert,
                seed=0,
            )
    return np.where(grid >= el.warmstart.PENALTY, np.nan, grid)


axes_of = {name: axis_of(name) for pair in projections for name in pair}
slices = {pair: slice_of(pair) for pair in projections}


# %% tags=["remove_input"]
def draw_slice(ax: Any, pair: tuple[str, str]) -> Any:
    """
    Draw one slice of the loss, and the true values on it

    Parameters
    ----------
    ax
        axes to draw on

    pair
        names of the two hyperparameters that vary

    Returns
    -------
    :
        the filled contours, for the colour bar
    """
    x_name, y_name = pair
    # grey is left where the draws overflow
    ax.set_facecolor("0.85")
    drawn = ax.contourf(
        axes_of[x_name],
        axes_of[y_name],
        np.log10(slices[pair]),
        levels=30,
        cmap="viridis",
    )
    ax.scatter(
        truth_unconstrained[x_name],
        truth_unconstrained[y_name],
        marker="*",
        s=260,
        color="white",
        edgecolor="black",
        linewidth=0.8,
        label="true",
    )
    ax.set_xlim(axes_of[x_name][0], axes_of[x_name][-1])
    ax.set_ylim(axes_of[y_name][0], axes_of[y_name][-1])
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    return drawn


# %% [markdown]
# The candidates of `from_elicits` come first. They are stored in
# `eliobj.results.initialization`. A candidate whose draws overflow gets no
# usable loss, and is marked with a cross. A cross can lie in a dark region of
# the cut: what overflows is then one of the four hyperparameters the cut
# holds, not the two it draws.

# %% tags=["remove_input"]
best = int(np.nanargmin(losses))

fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout="tight")
for ax, pair in zip(axs, projections):
    drawn = draw_slice(ax, pair)
    x = candidates[:, names.index(pair[0])]
    y = candidates[:, names.index(pair[1])]
    ax.scatter(
        x[~failed],
        y[~failed],
        s=25,
        color="white",
        edgecolor="0.3",
        linewidth=0.5,
        label="candidates",
    )
    ax.scatter(
        x[failed], y[failed], marker="x", color="crimson", s=45, label="overflows"
    )
    ax.scatter(x[best], y[best], marker="o", s=70, color="black", label="start")
    fig.colorbar(drawn, ax=ax, label="log10 loss")
axs[0].legend(fontsize="small", loc="lower right")
fig.suptitle(f"sobol: 32 candidates, {int(failed.sum())} overflow")
plt.show()

# %% [markdown]
# The search recorded above is next. It starts from the centre of the same box,
# and none of its 200 evaluations overflows: the box of `from_elicits` already
# keeps it out of the grey region. The penalty is what holds it there, because
# a point that overflows is never returned.

# %% tags=["remove_input"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout="tight")
for ax, pair in zip(axs, projections):
    drawn = draw_slice(ax, pair)
    x = search_path[:, names.index(pair[0])]
    y = search_path[:, names.index(pair[1])]
    ax.plot(
        x, y, color="white", linewidth=1.0, marker="o", markersize=3, label="search"
    )
    ax.scatter(
        x[overflowed],
        y[overflowed],
        marker="x",
        color="crimson",
        s=30,
        label="overflows",
    )
    ax.scatter(x[0], y[0], marker="o", s=70, color="black", label="start")
    fig.colorbar(drawn, ax=ax, label="log10 loss")
axs[0].legend(fontsize="small", loc="lower right")
fig.suptitle(
    f"warmstart: {len(scores)} evaluations, {int(overflowed.sum())} overflow, "
    f"best {scores.min():.3g}"
)
plt.show()

# %% [markdown]
# ### Did the training work?
#
# The criterion is not the distance to a true hyperparameter. A real
# elicitation has no true value. The criterion is whether the model reproduces
# the expert data.
#
# The loss over the epochs shows what the table hides. A start value that is
# merely poor gives a higher curve. A start value that cannot train gives a
# curve that stops, and two of the four curves stop.

# %% tags=["remove_input"]
fig, ax = plt.subplots(figsize=(7, 4), layout="tight")
for label, fitted in fits.items():
    trace = np.ravel(
        np.asarray(fitted.results.history_stats.loss.total_loss.sel(replication=0))
    )
    ax.plot(trace, label=label)
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("total loss")
ax.legend(fontsize="small")
ax.set_title("loss per initialization method, 100 epochs")
plt.show()

# %% [markdown]
# The runs that stopped cannot be compared any further, so only the warm start
# is followed from here. One hundred epochs are not enough to converge, so
# train it for 600, and then ask whether the model reproduces the expert data.
# [`elicits`][elicito.plots.elicits] draws the expert-elicited value against
# the model-simulated one, for each of the three targets. A point on the
# diagonal is a statistic the model reproduces.

# %%
long_run = build(
    el.initializer(
        method="warmstart",
        iterations=100,
        distribution=el.initialization.from_elicits(),
    ),
    epochs=600,
)
long_run.fit()

# %% tags=["remove_input"]
el.plots.elicits(long_run, cols=3, figsize=(9, 2.6))
plt.show()

# %% [markdown]
# ### Watch the hyperparameters
#
# This model has an oracle, so the true hyperparameters are known here. Compare
# each one with its true value, and see what the expert data can pin down.
#
# $\mu_0$, $\mu_1$ and $\sigma_0$ arrive close to their true values. The other
# three do not: $k_2$ ends near 13 against a true 2, and $\lambda_2$ near 2.5
# against a true 5.
#
# This is not a failure of the initialization, and not a lack of epochs. The
# fitted point reaches 0.228, below the 0.234 that the run from the true values
# reached before it stopped. The 15 elicited quantiles, at three values of
# the predictor, do not identify six hyperparameters: a smaller Weibull scale
# with a much larger shape produces the same predictive quantiles. Query the
# parameters themselves if you need every hyperparameter back.

# %% tags=["remove_input"]
history = long_run.results.history_stats.hyperparameter.sel(replication=0)

fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="tight", sharex=True)
for ax, name in zip(np.ravel(axes), truth):
    trace = np.ravel(np.asarray(history[name].values))
    ax.plot(trace, color="steelblue")
    ax.axhline(truth[name], color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{name}: {trace[-1]:.2f}, true {truth[name]:.2f}")
    ax.set_xlabel("epoch")
fig.suptitle("hyperparameters over 600 epochs, dashed line is the true value")
plt.show()

# %% tags=["remove_input"]
loss_trace = np.ravel(
    np.asarray(long_run.results.history_stats.loss.total_loss.sel(replication=0))
)
fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
ax.plot(loss_trace, color="darkorange")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("total loss")
ax.set_title(f"final loss {loss_trace[-1]:.3f}")
plt.show()

# %% [markdown]
# ## Which one to choose
#
# Part 1, two hyperparameters, 300 epochs, and 600 for the far box:
#
# | method       | final loss | seconds |
# | :----------- | ---------: | ------: |
# | far box      |      0.467 |    57.6 |
# | exact values |      0.079 |    28.1 |
# | uniform box  |      0.079 |    30.5 |
# | from_elicits |      0.079 |    30.8 |
# | warmstart    |      0.079 |    31.4 |
#
# Part 2, six hyperparameters and a likelihood that overflows, 100 epochs:
#
# | method       | final loss | seconds |
# | :----------- | ---------: | ------: |
# | exact values |      0.234 |     1.0 |
# | uniform box  |      0.228 |    22.4 |
# | from_elicits |      1.637 |     5.5 |
# | warmstart    |      0.226 |    35.4 |
#
# + Use **exact values** when you know them. Nothing is cheaper.
# + Use **`from_elicits`** when you have no numbers. It asks nothing of you,
#   and it puts the box in the right region. In part 2 the default box
#   `mean=0, radius=1` cannot even start, while `from_elicits` reaches 1.64.
# + Use **`warmstart`** on top of that box when the model can overflow, as the
#   Weibull likelihood of part 2 does. It is the best of the four there, at
#   0.226, and the slowest, because it costs one forward simulation per
#   evaluation.
# + Use a **`uniform` box** that you centre yourself when you know the order of
#   magnitude of the hyperparameters. Add `warmup_epochs` to reject a candidate
#   that diverges in the first epochs.
#
# Part 1 also shows the limit of all of this. On a surface with one minimum,
# and a gradient everywhere the box sits, every method reaches the same loss.
# The start value earns its cost only where the training cannot repair it: in a
# region without a gradient, as on the flat floor of part 1, or where the draws
# overflow, as in part 2.
