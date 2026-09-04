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
#     \sigma &\sim \text{HalfNormal}(\sigma_2) \\
#     y &\sim \text{Normal}(\beta_0 + \beta_1 X, \sigma)
# \end{align*}
# $$
#
# The five hyperparameters $\mu_0, \sigma_0, \mu_1, \sigma_1, \sigma_2$ are
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

        epred = tf.add(
            prior_samples[:, :, 0][:, :, None],
            tf.multiply(prior_samples[:, :, 1][:, :, None], X),
        )
        likelihood = tfd.Normal(
            loc=epred, scale=tf.expand_dims(prior_samples[:, :, -1], -1)
        )
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
        name="sigma",
        family=tfd.HalfNormal,
        hyperparams=dict(scale=el.hyper("sigma2", lower=0)),
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
    "beta0": tfd.Normal(loc=5.0, scale=1.0),
    "beta1": tfd.Normal(loc=2.0, scale=1.0),
    "sigma": tfd.HalfNormal(scale=10.0),
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
# $1$ is the value $\text{softplus}^{-1}(1) = 0.54$, and the true values of
# this model are $\mu_0=5, \sigma_0=0.54, \mu_1=2, \sigma_1=0.54,
# \sigma_2=10.0$.
#
# A box that sits far from these values starts the training in a bad place.

# %%
results.append(
    fit_and_report(
        "bad box",
        el.initializer(
            method="sobol",
            iterations=32,
            distribution=el.initialization.uniform(radius=2, mean=-20),
        ),
    )
)

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
                mu0=5.0,
                sigma0=el.utils.LowerBound(lower=0.0).forward(1.0),
                mu1=2.0,
                sigma1=el.utils.LowerBound(lower=0.0).forward(1.0),
                sigma2=el.utils.LowerBound(lower=0.0).forward(10.0),
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

# %%
results.append(
    fit_and_report(
        "uniform box",
        el.initializer(
            method="sobol",
            iterations=32,
            distribution=el.initialization.uniform(radius=1, mean=0),
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
# `radius`. The box is built during `fit`, from the pooled median and spread of
# the elicited statistics. A location hyperparameter is centred at the median.
# A scale hyperparameter spans from near zero up to the spread.

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
# exploding gradient. Here `iterations` is the number of objective evaluations.
# The centre of `distribution` is the start point of the search.

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
# + Use **exact values** when you know them. Nothing is cheaper.
# + Use **`from_elicits`** as the default. It needs no number from you, and it
#   is as cheap as any other box.
# + Use **`warmstart`** for a box you do not trust, for example when the scale
#   of the hyperparameters is unknown, or when a prior family overflows. On
#   this toy model it wins. On a harder model it rescued the worst box and lost
#   to sampling on a well-placed one. It costs one forward simulation per
#   evaluation.
# + Use a **`uniform` box** with `warmup_epochs` when you want several
#   candidates and want to reject the ones that diverge early.
