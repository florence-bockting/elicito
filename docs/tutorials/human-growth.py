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
# # The human growth model
#
# We apply `elicito` to the Preece-Baines growth model of
# Hartmann et al. (2020, PMLR 124).

# %% [markdown]
# ## Imports

# %%
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

# %% [markdown]
# ## The oracle table
#
# The raw elicited thresholds are not published. Each table gives only the
# resulting prior, as $E[\cdot]$ and $V(\cdot)$ per parameter. We treat the
# printed *predictive* prior as the oracle.
#
# Three variances are censored, all of them for $s_0$: `< 0.1`, `< 0.01` and
# `< 0.001`. Half the bound is the obvious substitute, but it fails here. The
# bound of user 1 equals the mean of $s_0$. Half of it gives a standard
# deviation of 0.22, more than twice the mean, and such a prior puts a large
# part of its mass on a curve with a **negative** stature at birth. The 10%
# quantile of the simulated stature at age 0 is then $-311$ cm.
#
# So we set the standard deviation to a quarter of the mean instead. The
# result stays below every printed bound, and every simulated stature is
# positive.
#
# | user | printed | half the bound | our substitute |
# | :-- | --: | --: | --: |
# | 1 | < 0.1 | 0.05 | 0.000625 |
# | 2 | < 0.01 | 0.005 | 0.0001 |
# | 3 | < 0.001 | 0.0005 | 0.00030625 |
#
# The table is stated in the parameters $\delta_h$ and $\delta_s$ of the next
# section, not in $h_1$ and $s_1$. Under independence, $E[\delta] = E[h_1] -
# E[h_0]$ and $V(\delta) = V(h_1) + V(h_0)$.

# %%
# user 1: Table 1 of Hartmann et al. (2020)
# users 2, 3: Tables 1 and 2 of the supplement
PREDICTIVE_PRIOR = {
    # parameter: (mean, variance) per user
    # the order must match the unpacking in GrowthModel.__call__
    "hts": [(162.8, 4.2), (153.73, 1.6), (148.8, 1.86)],
    "dh": [(11.7, 5.0), (38.01, 5.92), (28.34, 5.54)],
    "ts": [(13.4, 0.01), (15.9, 0.7), (11.31, 0.21)],
    # the printed variance of s0 is censored; see the text above
    "s0": [(0.1, 0.000625), (0.04, 0.0001), (0.07, 0.00030625)],
    "ds": [(3.2, 0.26), (1.96, 4.305), (4.47, 37.8305)],
    "b": [(15.79, 12.9), (61.4, 111.4), (18.4, 12.5)],
}

# %% [markdown]
# ## The model
#
# The growth curve is equation 25 of the supplement:
#
# $$
# h(t; \theta) = h_1 - \frac{2 (h_1 - h_0)}
# {\exp[s_0 (t - t_s)] + \exp[s_1 (t - t_s)]}
# $$
#
# The model needs $0 < h_0 < h_1$ and $0 < s_0 < s_1$, to identify it and to
# keep it physically plausible. We follow Manderson and Goudie (2026) and
# learn the differences
#
# $$
# \delta_h = h_1 - h_0, \qquad \delta_s = s_1 - s_0
# $$
#
# instead. Then $h_0$, $\delta_h$, $s_0$ and $\delta_s$ share one positivity
# constraint, and a LogNormal prior satisfies it.
#
# The model learns six parameters, in this order:
#
# - $h_{t_s}$ (`hts`), the stature at $t_s$, in cm. The curve passes through
#   this point, so it fixes the height at the turn.
# - $\delta_h$ (`dh`), the growth that is left after $t_s$, in cm. The adult
#   height is $h_1 = h_{t_s} + \delta_h$.
# - $t_s$ (`ts`), the age at peak growth velocity, in years. It shifts the
#   curve along the age axis.
# - $s_0$ (`s0`), the slow rate constant, in 1/year. Its term dominates the
#   denominator before $t_s$, so $s_0$ sets the childhood growth rate.
# - $\delta_s$ (`ds`), the extra rate of the second constant, in 1/year, with
#   $s_1 = s_0 + \delta_s$. The $s_1$ term dominates after $t_s$. A large
#   $\delta_s$ gives a short and sharp adolescent spurt.
# - $b$ (`b`), the shape of the Weibull likelihood. It sets the spread of one
#   stature around its mean, and a large $b$ gives a small spread. The
#   coefficient of variation is 7.8% at the oracle mean $b = 15.8$ of user 1.
#
# The remedy is not a proof. Manderson and Goudie note that the denominator
# can still be very small, which gives a negative height.
#
# They also constrain $t_s$ to the range of the observed ages. We follow them.
# $t_s$ has a Normal prior truncated to $[0, T]$, where $T$ is the oldest
# queried age. Its location is bounded to the same range, so the unconstrained
# hyperparameter is a logit and $T \cdot \mathrm{sigmoid}$ maps it back.
#
# The bound is still not enough. Measured on 51200 draws from the true prior
# of user 1, the mean height at age 0 was negative for 4.5% of them. So the
# model passes the predicted mean through a softplus. Above 20 cm a softplus
# is the identity to within $10^{-9}$, and a height is positive by
# construction.
#
# The expert sees the stature at four ages.

# %%
AGES = tf.constant([0.0, 2.5, 10.0, 17.5])
T_MAX = float(AGES[-1])


def bounded_ts(loc: Any, scale: Any) -> Any:
    """
    Build the prior of the age at peak velocity

    A Normal truncated to the range of the queried ages.

    Parameters
    ----------
    loc, scale
        location and scale of the untruncated Normal

    Returns
    -------
    :
        truncated Normal on ``[0, T_MAX]``
    """
    return tfd.TruncatedNormal(loc=loc, scale=scale, low=0.0, high=T_MAX)


class GrowthModel:
    """
    Preece-Baines generative model
    """

    # `elicito` calls this inside a compiled forward pass. AutoGraph would
    # then rewrite the body, and it finds the body by file and line. A cell
    # of a notebook has no source file in a worker process of
    # `fit(parallel=...)`, so AutoGraph rewrites the wrong function there.
    # This model has no Python branch to rewrite, so we switch it off.
    @tf.autograph.experimental.do_not_convert
    def __call__(
        self, prior_samples: Any, ages: Any, seed: Any = None
    ) -> dict[str, Any]:
        """
        Compute target quantities from generative model

        Parameters
        ----------
        prior_samples
            prior samples

        ages
            ages at which the expert is queried

        seed
            stateless seed for the likelihood draw. `elicito` passes it
            because this signature names it. The draw then repeats without
            `tf.random.set_seed`, which the compiled forward pass cannot use.

        Returns
        -------
        :
            dictionary with target quantities
        """
        # shape (B, S, 1) each, to broadcast over the ages
        hts, dh, ts, s0, ds, b = (
            tf.expand_dims(prior_samples[:, :, i], -1) for i in range(6)
        )

        # back to the parameters of the curve; both differences are positive
        h1 = hts + dh
        s1 = s0 + ds

        # growth curve, supplement equation 25
        epred = h1 - 2 * (h1 - hts) / (
            tf.exp(s0 * (ages - ts)) + tf.exp(s1 * (ages - ts))
        )

        # The curve dips below zero for an extreme draw, and a height cannot.
        # softplus equals the identity to within 1e-9 above 20 cm, so it
        # changes nothing in the plausible range.
        epred = tf.nn.softplus(epred)

        # Weibull in the mean-variance form, supplement equation 24
        likelihood = tfd.Weibull(
            concentration=b,
            scale=epred / tf.exp(tf.math.lgamma(1.0 + 1.0 / b)),
        )
        ypred = likelihood.sample(seed=seed)

        y_t0, y_t1, y_t2, y_t3 = (ypred[:, :, i] for i in range(4))

        return dict(
            y_t0=y_t0, y_t1=y_t1, y_t2=y_t2, y_t3=y_t3, ypred=ypred, epred=epred
        )

# %% [markdown]
# ## The oracle
#
# We match a LogNormal to each growth parameter by moments, and a Gamma to
# the Weibull shape $b$.


# %%
def oracle(user: int) -> dict[str, Any]:
    """
    Build the ground-truth prior of one user

    Parameters
    ----------
    user
        user number, 1, 2 or 3

    Returns
    -------
    :
        dictionary with one distribution per parameter
    """
    truth = dict()
    for name, table in PREDICTIVE_PRIOR.items():
        m, v = table[user - 1]
        if name == "b":
            truth[name] = tfd.Gamma(concentration=m**2 / v, rate=m / v)
        elif name == "ts":
            # the bounds are far from the mass, so the moments are the
            # untruncated ones to machine precision for user 1
            truth[name] = bounded_ts(loc=float(m), scale=float(np.sqrt(v)))
        else:
            sigma2 = np.log(1.0 + v / m**2)
            truth[name] = tfd.LogNormal(
                loc=float(np.log(m) - sigma2 / 2.0),
                scale=float(np.sqrt(sigma2))
            )
    return truth


oracle(user=1)

# %% [markdown]
# ## Model parameters
#
# The hyperparameter vector is $\lambda = \{a_m, b_m\}$, $m = 0, \ldots, 5$.

# %%
# median bounds, in the units of each parameter. The true medians of the
# three users are 149-163 cm, 11-38 cm, 0.039-0.097 1/y and 1.3-3.2 1/y.
MEDIAN_BOUNDS = {
    "hts": (50.0, 250.0),
    "dh": (1.0, 100.0),
    "s0": (0.01, 1.0),
    "ds": (0.1, 10.0),
}

# every true scale is below 1.04, so a LogNormal mean stays below
# exp(2**2 / 2) = 7.4 times its median
MAX_SCALE = 2.0


def growth_parameter(name: str) -> Any:
    """Build a positive growth parameter with a LogNormal prior"""
    lower, upper = MEDIAN_BOUNDS[name]
    return el.parameter(
        name=name,
        family=tfd.LogNormal,
        hyperparams=dict(
            loc=el.hyper(f"a_{name}", lower=np.log(lower), upper=np.log(upper)),
            scale=el.hyper(f"b_{name}", lower=0, upper=MAX_SCALE),
        ),
    )


# the order must match the order of PREDICTIVE_PRIOR
parameters = [
    growth_parameter("hts"),
    growth_parameter("dh"),
    el.parameter(
        name="ts",
        family=bounded_ts,
        hyperparams=dict(
            # bounded to the same range as the prior, so the unconstrained
            # value is a logit and `T_MAX * sigmoid` maps it back
            loc=el.hyper("a_ts", lower=0, upper=T_MAX),
            scale=el.hyper("b_ts", lower=0),
        ),
    ),
    growth_parameter("s0"),
    growth_parameter("ds"),
    el.parameter(
        name="b",
        family=tfd.Gamma,
        hyperparams=dict(
            concentration=el.hyper("a_b", lower=0, upper=100),
            # the true rates are 0.55 to 1.47
            rate=el.hyper("b_b", lower=0, upper=10.0),
        ),
    ),
]

# %% [markdown]
# ## Elicitation design
#
# The expert reports the stature at each of the four ages, as the quantiles
# 0.10, 0.25, 0.50, 0.75 and 0.90.

# %%
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

targets = [
    el.target(
        name=f"y_t{i}",
        query=el.queries.quantiles(QUANTILES),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    )
    for i in range(4)
]

# %% [markdown]
# ## The generative model and the oracle

# %%
model = el.model(obj=GrowthModel, ages=AGES)

expert = el.expert.simulator(
    ground_truth=oracle(user=1),
    num_samples=10_000
)

# %% [markdown]
# ## The initialization box
#
# The box says where the search begins. It lives in the unconstrained space.
# Every hyperparameter above has two bounds, so every unconstrained value is
# a logit, and `lower + (upper - lower) * sigmoid` maps it back. `b_ts` is
# the one exception. It has a lower bound only, so it is a value before the
# softplus transform.
#
# One logit is then like the next, whatever the parameter. A value of 0 is
# the midpoint of the interval. At $|y| = 6$ the value is within 0.25% of a
# bound, the prediction stops changing, and the search wastes its
# evaluations there.
#
# We state no prior knowledge. The box is centred on zero, with a radius of
# 6 in every direction. The centre is the midpoint of every interval: a
# median stature of 112 cm, a median $s_0$ of 0.1 per year, and the spurt at
# 8.75 years. The true values of the three users sit at $|y| \leq 5.5$, so
# the box holds them all.
#
# Only the Adam run below reads this box. With `optimizer="cmaes"`, `elicito`
# ignores the initializer, with a warning. The CMA-ES training then starts at
# 0, the midpoint of every interval, which is also the centre of this box.
# The training is itself a search. A search before it would only throw away
# the covariance matrix that the training adapts.
#
# For the Adam run, `initializer(method="cmaes")` searches the **whole** box
# for a start value. It starts at the centre, and it takes the step size of
# each coordinate from the radius of that coordinate. `warmstart` runs one
# local Nelder-Mead search from the centre, and `random`, `lhs` and `sobol`
# rank independent draws. All three take the first usable basin, which is not
# the best one.
#
# `cmaes` needs the optional dependency `cma`: `pip install "elicito[cma]"`.

# %%
BOX = el.initializers.uniform(mean=0.0, radius=6.0)

# %% [markdown]
# ## Training
#
# We fit with CMA-ES, not with a gradient. `el.optimizer(optimizer="cmaes")`
# replaces the gradient descent by one search over the whole budget, on the
# full 400 prior draws.
#
# The search has no gradient to explode, and it adapts its own step size.
# `sigma0` is the first step size, on the unconstrained scale. A number sets
# one step size for every coordinate. A dictionary
# `{hyperparameter name: step size}` sets one per hyperparameter.
#
# Every coordinate is a logit now, on the same scale, so one number fits all
# twelve. A step of 3.0 moves a logit from the midpoint to a bound, where
# the loss is flat. We give 1.0. The true values sit at $|y| \leq 5.5$, so a
# few generations reach them, and CMA-ES adapts the step size afterwards.
#
# `epochs` then counts forward simulations, not gradient steps. The search
# spends the budget in generations of `4 + 3 ln(12)` = 11 candidates, so 3000
# evaluations are about 272 generations, and the loss curve has one point per
# generation.
#
# CMA-ES adapts a full covariance matrix of 12 hyperparameters, which needs of
# the order of 12² = 144 generations. 3000 evaluations pass that number. The
# page is therefore slow to render.
#
# This fit is one chain. The section on the three experts below runs five
# replications of one fit with `fit(parallel=...)`.

# %%
eliobj = el.Elicit(
    model=model,
    parameters=parameters,
    targets=targets,
    expert=expert,
    optimizer=el.optimizer(
        optimizer="cmaes",
        sigma0=1.0,
    ),
    trainer=el.trainer(
        method="parametric_prior",
        seed=2025,
        epochs=3000,
        progress=0,
        num_samples=400,
    ),
)

# %%
import time

start = time.time()
eliobj.fit()

# %% [markdown]
# The results hold the learned hyperparameters and the loss. The prior
# samples, the model simulations, the target quantities and the elicited
# summaries follow from them, and `sample` computes them.

# %%
samples = eliobj.sample()
cma_seconds = time.time() - start

# %% [markdown]
# ## Did the training succeed?
#
# The loss curve says whether the optimizer settled. The elicits plot says
# whether the simulated quantiles reach the expert values at each age.

# %%
el.plots.loss(eliobj, figsize=(7, 3));

# %%
el.plots.elicits(eliobj, cols=4, figsize=(9, 3), samples=samples);

# %% [markdown]
# The parameters of interest are $h_{t_s}$, $h_1$, $s_0$, $s_1$ and $t_s$.
# The model does not sample $h_1$ and $s_1$; it computes them from the two
# differences. `el.utils.add_derived` writes both back into the prior samples,
# so the plot can select them by name.

# %%
PARAMS_OF_INTEREST = ["hts", "h1", "s0", "s1", "ts"]


def add_h1_s1(draws):
    """Add the derived parameters h1 and s1 to the prior samples."""
    el.utils.add_derived(
        draws,
        h1=lambda prior: prior["hts"] + prior["dh"],
        s1=lambda prior: prior["s0"] + prior["ds"],
    )


add_h1_s1(samples)
el.plots.prior_marginals(
    eliobj, params=PARAMS_OF_INTEREST, cols=5, figsize=(11, 2.4), samples=samples
);

# %% tags=["remove_input"]
import matplotlib.pyplot as plt

TARGETS = ["y_t0", "y_t1", "y_t2", "y_t3"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]


def predictive(name):
    """Return the prior predictive draws of one target."""
    return (
        samples["target_quantity"][name]
        .to_dataset()
        .stack(stacked=("batch", "draw"))
        .to_array()
        .values[0, 0, :]
    )


fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
for name, age, color in zip(TARGETS, AGES.numpy(), COLORS):
    counts, edges, _ = ax.hist(
        predictive(name), bins="auto", density=True, histtype="step",
        linewidth=2, color=color, label=f"age {age:g}",
    )
    mid = 0.5 * (edges[:-1] + edges[1:])
    ax.annotate(
        f"age {age:g}", (mid[counts.argmax()], counts.max()),
        textcoords="offset points", xytext=(0, 5),
        ha="center", fontsize="x-small", color="#1a1a19",
    )

ax.set_xlabel("stature (cm)", fontsize="small")
ax.set_ylabel("density", fontsize="small")
ax.set_title("prior predictive stature by age", fontsize="small")
ax.legend(fontsize="small", frameon=False)
ax.grid(color="lightgrey", linestyle="dotted", linewidth=1)
ax.set_axisbelow(True)
ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(labelsize="x-small")

# %% tags=["remove_input"]
from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
for ax, name, age, color in zip(axes, TARGETS, AGES.numpy(), COLORS):
    ax.hist(
        predictive(name), bins="auto", density=True,
        color=color, alpha=0.5, edgecolor="none",
    )
    expert_quantiles = (
        eliobj.results.oracle.sel(replication=0)[f"quantiles_{name}"].values[0]
    )
    for q, x in zip(QUANTILES, expert_quantiles):
        ax.axvline(
            x, color="#1a1a19", linewidth=2 if q == 0.5 else 1,
            linestyle="solid" if q == 0.5 else "dashed",
        )
    ax.set_title(f"age {age:g}", fontsize="small")
    ax.set_xlabel("stature (cm)", fontsize="small")
    ax.grid(color="lightgrey", linestyle="dotted", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(labelsize="x-small")

axes[0].set_ylabel("density", fontsize="small")
axes[0].legend(
    handles=[
        Line2D([0], [0], color="#1a1a19", lw=2, label="expert median"),
        Line2D([0], [0], color="#1a1a19", lw=1, ls="dashed",
               label="expert 10/25/75/90%"),
    ],
    fontsize="x-small", frameon=False,
)

# %% [markdown]
# ## The same fit, with a gradient
#
# Adam is the alternative. It uses the same seed and the same 400 prior
# draws. Adam needs a start value, so a CMA-ES search of 300 evaluations over
# the box runs first. The CMA-ES run has no search before it.
#
# The budget is not the same work, and the two numbers are not comparable.
# For Adam, 800 epochs are 800 gradient steps, and each step is one forward
# pass **and** one backward pass. For CMA-ES, 3000 is the number of forward
# simulations.
#
# `decay_steps` matches the number of epochs. The schedule then spans the
# whole run, as it did at the shorter budget.
#
# Adam needs two settings that the search does not need: a learning rate
# schedule, and `clipnorm`. Without them the run dies with a loss that is not
# finite.

# %%
eliobj_adam = el.Elicit(
    model=eliobj.model,
    parameters=eliobj.parameters,
    targets=eliobj.targets,
    expert=eliobj.expert,
    optimizer=el.optimizer(
        optimizer=tf.keras.optimizers.Adam,
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=800,
            decay_rate=0.02,
        ),
        clipnorm=1.0,
    ),
    trainer=el.trainer(
        method="parametric_prior",
        seed=2025,
        epochs=800,
        progress=0,
        num_samples=400,
    ),
    initializer=el.initializer(
        method="cmaes",
        distribution=BOX,
        iterations=300,
    ),
)

start = time.time()
eliobj_adam.fit()
adam_seconds = time.time() - start


# %%
def report(obj, label, seconds):
    """Print the final loss, the learned growth spurt, and the run time."""
    loss = obj.results["history_stats/loss"]["total_loss"].values[0]
    # the history stores the constrained value, so a_ts is already in years
    ts = obj.results["history_stats/hyperparameter"]["a_ts"].values[0]
    print(
        f"{label:5s}: {len(loss):3d} points, final loss {loss[-1]:8.2f},"
        f" best loss {loss.min():8.2f}, ts {ts[-1]:5.2f} years,"
        f" {seconds:5.0f} s"
    )


report(eliobj, "cmaes", cma_seconds)
report(eliobj_adam, "adam", adam_seconds)

# %%
el.plots.loss(eliobj_adam, figsize=(7, 3));

# %%
samples_adam = eliobj_adam.sample()
el.plots.elicits(eliobj_adam, cols=4, figsize=(9, 3), samples=samples_adam);

# %%
add_h1_s1(samples_adam)
el.plots.prior_marginals(
    eliobj_adam,
    params=PARAMS_OF_INTEREST,
    cols=5,
    figsize=(11, 2.4),
    samples=samples_adam,
);

# %% [markdown]
# ## Three experts
#
# The paper reports three users. Table 1 of the paper gives user 1, and
# tables 1 and 2 of the supplement give users 2 and 3. `PREDICTIVE_PRIOR`
# above already holds the three columns, and `oracle(user)` turns one column
# into a ground truth. So this section needs no new data.
#
# The three users do not agree:
#
# - user 1 expects an adult height of 174.5 cm, and the spurt at 13.4 years.
# - user 2 expects the tallest adult, 191.7 cm, and the latest spurt, at 15.9
#   years. Its $b = 61.4$ is the largest of the three, so user 2 states the
#   smallest spread around one stature, 2.1%.
# - user 3 expects 177.1 cm, and the earliest spurt, at 11.3 years.
#
# Every fit uses the same design, the same start point and the same budget.
# Only the expert changes.
#
# One fit is one sample of one search. We repeat it five times, with a
# different seed each time. The five runs then separate two effects. A
# difference between two experts is the expert. A difference between two runs
# of the same expert is the search.
#
# `fit(parallel=...)` runs the five replications, and `elicito` stores them
# in the `replication` dimension of `eliobj.results`.

# %%
USERS = (1, 2, 3)
USER_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
REPLICATIONS = 5
SEEDS = [2025, 2026, 2027, 2028, 2029]


def fit_user(user):
    """Fit the growth model to the prior of one user, five times."""
    obj = el.Elicit(
        model=model,
        parameters=parameters,
        targets=targets,
        expert=el.expert.simulator(
            ground_truth=oracle(user),
            num_samples=10_000,
        ),
        optimizer=el.optimizer(
            optimizer="cmaes",
            sigma0=1.0,
        ),
        trainer=el.trainer(
            method="parametric_prior",
            seed=2025,
            epochs=3000,
            progress=0,
            num_samples=400,
        ),
    )
    obj.fit(parallel=el.utils.parallel(runs=REPLICATIONS, cores=5, seeds=SEEDS))
    return obj


fits = {u: fit_user(u) for u in USERS}

# %% [markdown]
# ## Did the search converge?
#
# The best loss of each run is the first check. A run with a large loss did
# not fit, and it says nothing about its expert. A spread over the five runs
# of one expert is the search, not the expert.


# %%
samples_user = {u: fits[u].sample() for u in USERS}


def successful(draws):
    """Return the replications whose simulated elicits are finite."""
    summary = draws["elicited_summary"]
    return [
        i
        for i in range(summary.sizes["replication"])
        if all(
            np.isfinite(summary.sel(replication=i)[name].values).all()
            for name in summary.data_vars
        )
    ]


print(f"{'user':>5s}  best loss of each of the {REPLICATIONS} runs")
for u in USERS:
    loss = fits[u].results["history_stats/loss"]["total_loss"].values
    print(
        f"{u:5d} "
        + " ".join(f"{np.nanmin(loss[i]):8.2f}" for i in successful(samples_user[u]))
    )

# %% [markdown]
# ## Read the fit in the elicit space
#
# We do **not** compare the learned prior with the prior of the oracle. The
# six parameters are not identified. The section on the model above names two
# directions in which the prediction does not change: $s_0$ and $s_1$ near
# zero, where $\delta_h$ drops out of the curve, and $s_0$ above the overflow
# of `exp`, where every value gives the same curve. A distance in the
# parameter space therefore says nothing about the quality of a fit.
#
# The fit is judged where the expert spoke. The expert states five quantiles
# at each of the four ages, so twenty numbers. The model simulates the same
# twenty numbers. A point on the diagonal is a match.
#
# One row is one expert, and one column is one age. Each row holds the five
# runs of that expert.

# %% tags=["remove_input"]
fig, axes = plt.subplots(
    len(USERS), len(TARGETS), figsize=(9, 6.6), constrained_layout=True
)
for row, u in zip(axes, USERS):
    res = fits[u].results
    summary = samples_user[u]["elicited_summary"]
    for ax, name, age in zip(row, TARGETS, AGES.numpy()):
        for i in successful(samples_user[u]):
            expert_q = res.oracle.sel(replication=i)[f"quantiles_{name}"].values[0]
            model_q = summary.sel(replication=i)[f"quantiles_{name}"].values.mean(
                axis=0
            )
            ax.plot(expert_q, model_q, "o", ms=5, alpha=0.5, color=USER_COLORS[u])
        ax.axline((0, 0), slope=1, color="darkgrey", linestyle="dashed", lw=1)
        ax.set_title(f"user {u}, age {age:g}", fontsize="small")
        ax.grid(color="lightgrey", linestyle="dotted", linewidth=1)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(labelsize="x-small")
    row[0].set_ylabel("model-simulated", fontsize="small")
for ax in axes[-1]:
    ax.set_xlabel("expert", fontsize="small")
fig.suptitle(
    "expert-elicited and model-simulated quantiles (cm)", fontsize="medium"
)

# %% [markdown]
# ## The learned priors
#
# The prior is the answer that the expert asked for, so we show it. We do not
# score it against a truth, for the reason above.
#
# One row is one expert, and one column is one parameter. Each cell holds the
# five runs. Five lines on top of each other say that the search finds one
# answer for that expert. Five different lines say that it finds many, and
# every one of them fits the elicits of the figure above.
#
# The column is shared over the three rows, so a row compares with a row.

# %%
for u in USERS:
    add_h1_s1(samples_user[u])


def prior_draws(obj_samples, replication):
    """Return the learned draws of one replication, per parameter."""
    prior = obj_samples["prior"].sel(replication=replication).to_dataset()
    return {name: prior[name].values.ravel() for name in PARAMS_OF_INTEREST}


draws = {
    u: {i: prior_draws(samples_user[u], i) for i in successful(samples_user[u])}
    for u in USERS
}

# one range per column, from the pooled draws, so that the rows compare
ranges = {
    name: np.percentile(
        np.concatenate([d[name] for u in USERS for d in draws[u].values()]), [1, 99]
    )
    for name in PARAMS_OF_INTEREST
}

# %% tags=["remove_input"]
fig, axes = plt.subplots(
    len(USERS), len(PARAMS_OF_INTEREST), figsize=(12, 6.6), constrained_layout=True
)
for row, u in zip(axes, USERS):
    for ax, name in zip(row, PARAMS_OF_INTEREST):
        bins = np.linspace(*ranges[name], 60)
        for d in draws[u].values():
            ax.hist(
                d[name], bins=bins, density=True, histtype="step",
                linewidth=1.5, alpha=0.8, color=USER_COLORS[u],
            )
        ax.set_xlim(*ranges[name])
        ax.set_yticks([])
        ax.grid(color="lightgrey", linestyle="dotted", linewidth=1)
        ax.set_axisbelow(True)
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.tick_params(labelsize="x-small")
    row[0].set_ylabel(f"user {u}", fontsize="small")
for ax, name in zip(axes[0], PARAMS_OF_INTEREST):
    ax.set_title(name, fontsize="small")
fig.suptitle("learned prior, five runs per expert", fontsize="medium")

# %% [markdown]
# Read the figure in two steps. First read one row: the spread of the five
# lines is the spread of the search. Then compare the rows: that distance is
# the disagreement between the experts, as the fit recovers it. A row whose
# five lines disagree does not support a statement about its expert.

# %% [markdown]
# ## One answer instead of many
#
# The rows above disagree because the twenty elicited numbers do not fix the
# six hyperparameters. The literature calls this non-identification. It has
# one failure mode that is easy to see in the figure: a prior collapses onto
# a point. The search then reports a hyperparameter without any uncertainty,
# and a replication with another seed reports a different point.
#
# The collapse is not a bug of the search. It costs nothing. A parameter that
# the elicits do not constrain can push all of its spread into another
# parameter, and the twenty numbers stay the same.
#
# The literature offers four remedies:
#
# 1. **Ask for more.** A new query changes the map from prior to elicits. It
#    is the only remedy that removes the cause. More quantiles per age, or a
#    query on a quantity that depends on one parameter alone.
# 2. **Use a joint prior.** Fewer hyperparameters identify better than many
#    independent ones. Use `method="deep_prior"` for this.
# 3. **Penalize the collapse.** Add a term to the loss that grows without
#    bound as a prior standard deviation goes to zero.
# 4. **Report the spread, do not hide it.** The figure above already does
#    this. Five lines are five answers, and all five fit the expert.
#
# `elicito` implements the third remedy. `el.trainer()` takes a weight
# `kappa`. It multiplies the negative mean log standard deviation of the
# priors, which is the term of Manderson and Goudie (2026):
#
# $$
# \mathcal{L}(\lambda) = D(\lambda) - \frac{\kappa}{Q}
# \sum_{q=1}^{Q} \log \mathrm{SD}[\theta_q]
# $$
#
# The term goes to infinity as one standard deviation goes to zero, so the
# search cannot reach a point mass.
#
# ```python
# trainer=el.trainer(
#     method="parametric_prior",
#     seed=2025,
#     epochs=3000,
#     progress=0,
#     num_samples=400,
#     kappa=0.1,
# )
# ```
#
# Warning: `kappa` buys spread with fit. Start at a small value. Then read
# the two parts of the history against each other:
#
# ```python
# res = fits[0].results["history_stats/loss"]
# res["total_loss"].values[i][-1]
# res["penalty"].values[i][-1]
# ```
#
# The penalty is recorded at every epoch, also at `kappa=0`. A run at
# `kappa=0` therefore costs nothing and still tells you whether a prior is
# collapsing: the penalty climbs while the loss falls.
