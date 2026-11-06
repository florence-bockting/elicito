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
# # Check whether the expert information is consistent
#
# ## Motivation
#
# An expert gives several statements, for example quantiles of an outcome at
# different conditions. `elicito` learns the prior hyperparameters that
# reproduce all statements together. After the fit, `el.plots.elicits` compares
# the expert statements with the statistics that the learned prior simulates.
# Sometimes a point is far from the diagonal. There are two possible causes:
#
# + **The optimization failed.** A better prior exists, but the training did
#   not find it.
# + **The statements conflict.** No prior of the model reproduces all
#   statements together. The fit is a compromise between them.
#
# The two causes need different actions. A failed optimization needs other
# training settings, for example a different initialization. A conflict needs
# a conversation with the expert, or a different model.
#
# This guide answers two questions:
#
# 1. Is the expert information consistent?
# 2. If not, which statement conflicts with the others?
#
# We use a toy model with five cases of expert information. Two cases are
# consistent by construction. In one of them, the training stops too early.
# The three other cases have a different kind of conflict each. The guide
# ends with the mathematical details and with recommendations.

# %% [markdown]
# ## Imports

# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

# %% [markdown]
# ## The intuition
#
# The loss of `elicito` is a sum. Each expert statistic $k$ adds a loss
# $L_k(\lambda)$, and the training minimizes the total loss
# $L(\lambda) = \sum_k L_k(\lambda)$ over the hyperparameters $\lambda$.
#
# + If the statements are **consistent**, one $\lambda$ minimizes every
#   $L_k$ at the same time. The minimum of the sum is also the minimum of each
#   part.
# + If the statements **conflict**, the minima of the parts are at different
#   places. The minimum of the sum is a compromise, and no part is at its own
#   minimum.
#
# The figure shows this for one hyperparameter, the prior location $\mu_0$.
# Each thin line is the loss of one statistic. The thick line is the total
# loss. On the left, all three statistics have their minimum at $\mu_0 = 0$.
# On the right, statistic `c` has its minimum at $\mu_0 = 3$. The total loss
# then has its minimum at $\mu_0 = 1$, and `a` and `b` pay for it.
#
# Now remove `c` from the right panel. The minimum of the total loss moves
# back to 0, where `a` and `b` agree. The fit then predicts `c` at 0, far from
# its own minimum at 3. Remove `a` instead. The minimum moves to 1.5, and the
# fit predicts `a` at 1.5. In the left panel, the others predict each
# statistic at its own minimum. **The statistic that the others predict worst
# is the one that conflicts.** This is the leave-one-out idea behind the
# diagnostic.

# %% tags=["remove_input"]
grid = np.linspace(-2, 7, 400)
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True, constrained_layout=True)
for ax, minima, title in zip(
    axes, [(0, 0, 0), (0, 0, 3)], ["consistent", "c conflicts"]
):
    parts = [(grid - m) ** 2 for m in minima]
    for part, name in zip(parts, "abc"):
        ax.plot(grid, part, lw=1, label=f"$L_{name}$")
    total = np.sum(parts, axis=0)
    ax.plot(grid, total, color="black", lw=2, label="total")
    ax.axvline(grid[np.argmin(total)], color="black", ls="dotted", lw=1)
    ax.set_title(title, fontsize="small")
    ax.set_xlabel(r"$\mu_0$")
    ax.set_ylim(0, 40)
    ax.spines[["right", "top"]].set_visible(False)
axes[0].set_ylabel("loss")
axes[0].legend(fontsize="x-small", frameon=False)

# %% [markdown]
# ## The toy model
#
# One parameter $\theta$ has a Normal prior with two hyperparameters:
# $$
# \begin{align*}
#     \lambda &= (\mu_0, \sigma_0) \\
#     \theta &\sim \text{Normal}(\mu_0, \sigma_0)
# \end{align*}
# $$
# The expert answers three questions `a`, `b` and `c` about the same quantity
# $\theta$. For each question, the expert gives the 25 %, 50 % and 75 %
# quantiles. The model returns $\theta$ three times, once for each question,
# so the three statements must agree.
#
# | case | `a` | `b` | `c` | what happens |
# | :-- | :-- | :-- | :-- | :-- |
# | consistent | N(0, 1) | N(0, 1) | N(0, 1) | no conflict |
# | location | N(0, 1) | N(0, 1) | N(0.5, 1) | `c` moves the location |
# | spread | N(0, 1) | N(0, 1) | N(0, 2) | `c` changes the spread only |
# | all differ | N(-0.25, 1) | N(0, 1) | N(0.25, 1) | no single statement is the cause |
# | not converged | N(3, 2) | N(3, 2) | N(3, 2) | no conflict, but only 30 epochs |
#
# The table gives the Normal distribution from which each statement takes its
# quantiles. In the case "not converged", the training starts at
# $\mu_0 = 0$ and stops after 30 epochs instead of 100, before the
# optimum.

# %%
QUANTILES = (0.25, 0.50, 0.75)


class Copies:
    """Return the prior samples of theta under three names"""

    def __call__(self, prior_samples: Any) -> dict[str, Any]:
        """Return theta for each of the questions a, b and c"""
        theta = prior_samples[:, :, 0]
        return dict(a=theta, b=theta, c=theta)


def normal_quantiles(loc: float, scale: float) -> list[float]:
    """Return the quantiles of Normal(loc, scale), rounded to 2 decimals"""
    values = tfd.Normal(loc, scale).quantile(QUANTILES).numpy()
    return [round(float(v), 2) for v in values]


CASES = {
    "consistent": [(0, 1), (0, 1), (0, 1)],
    "location": [(0, 1), (0, 1), (0.5, 1)],
    "spread": [(0, 1), (0, 1), (0, 2)],
    "all differ": [(-0.25, 1), (0, 1), (0.25, 1)],
    "not converged": [(3, 2), (3, 2), (3, 2)],
}
# the training of this case stops too early
EPOCHS = {"not converged": 30}


def elicit(normals: list[tuple[float, float]], epochs: int = 100) -> Any:
    """Fit the prior to the quantiles of the three Normal distributions"""
    data = {
        f"quantiles_{name}": normal_quantiles(*normal)
        for name, normal in zip("abc", normals)
    }
    eliobj = el.Elicit(
        model=el.model(obj=Copies),
        parameters=[
            el.parameter(
                name="theta",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)
                ),
            )
        ],
        targets=[
            el.target(
                name=name,
                query=el.queries.quantiles(QUANTILES),
                loss=el.losses.MMD2(kernel="energy"),
                weight=1.0,
            )
            for name in "abc"
        ],
        expert=el.expert.data(dat=data),
        optimizer=el.optimizer(optimizer=tf.keras.optimizers.Adam, learning_rate=0.1),
        initializer=el.initializer(method=None, hyperparams=dict(mu0=0.0, sigma0=0.0)),
        trainer=el.trainer(
            method="parametric_prior",
            seed=1,
            epochs=epochs,
            B=32,
            num_samples=100,
            progress=0,
        ),
    )
    eliobj.fit()
    return eliobj


eliobjs = {
    case: elicit(normals, EPOCHS.get(case, 100)) for case, normals in CASES.items()
}

# %% [markdown]
# ## Step 1: check the optimization
#
# The diagnostics compare fits with each other, so they need a converged
# fit. `el.plots.loss` plots the loss over the epochs. Look at it first.

# %%
for case, eliobj in eliobjs.items():
    fig, _ = el.plots.loss(eliobj, figsize=(6, 2))
    fig.suptitle(case)

# %% [markdown]
# In four cases, the loss is flat at the end: the training has converged. In
# the case "not converged", the loss still falls at the last epoch. More
# epochs, a larger learning rate, or a better initialization can help (see
# "How to choose an initialization method"). In practice, fix the
# optimization before the next step. This guide keeps the case to show what
# the next steps do with it.

# %% [markdown]
# ## Step 2: look at the fit
#
# `el.plots.elicits` plots the expert quantiles against the quantiles that the
# learned prior simulates. A point on the diagonal is a statement that the
# prior reproduces.

# %%
for case, eliobj in eliobjs.items():
    fig, _ = el.plots.elicits(eliobj, figsize=(6, 2))
    fig.suptitle(case)

# %% [markdown]
# `el.diagnostics.baseline` refits the model with three seeds. It returns the
# loss of each statistic for each seed. It also returns the `distance` of
# each statistic: the largest
# difference between a simulated and an expert quantile, in units of the
# expert interval (the largest minus the smallest expert quantile).

# %%
references = {case: el.diagnostics.baseline(eliobj) for case, eliobj in eliobjs.items()}

print(f"{'case':<14}" + "".join(f"{name:>8}" for name in "abc"))
for case, reference in references.items():
    distances = reference["distance"].values
    print(f"{case:<14}" + "".join(f"{d:>8.2f}" for d in distances))

# %% [markdown]
# The distances tell where the fit is poor, but not why:
#
# + **consistent:** all distances are small.
# + **location:** `c` has the largest distance. The compromise moves the
#   location between 0 and 0.5, so `a` and `b` also miss their statements
#   a little.
# + **spread:** the compromise widens the prior. `a` and `b` get too wide,
#   `c` too narrow.
# + **all differ:** `a` and `c` miss, `b` is close to its statements.
# + **not converged:** all three distances are as large as the distance of
#   `c` in the case "spread". The plot looks like a conflict, but the
#   statements agree.
#
# A distance does not show which statement is the cause. In the case "all
# differ", `a` and `c` have large distances, but neither of them alone is the
# cause. The distance also does not tell a failed optimization from a
# conflict.

# %% [markdown]
# ## Step 3: leave one statistic out
#
# `el.diagnostics.leave_one_out` refits the model without one statistic at a
# time. It uses the seeds of `baseline`. Each refit predicts the left-out
# statistic from the other statistics. The `miss` is the largest distance of
# this prediction to the expert values, in units of `tolerance`:
#
# + **A miss below 1:** the other statistics predict the statement within
#   the precision of the expert. The statement does not conflict with them.
# + **A miss above 1:** the other statistics predict something that the
#   expert did not say. The statement conflicts with them.
#
# `tolerance` gives, for each statistic, the largest distance that the
# precision of the expert explains, in the units of the statistic. You choose
# it, for example from the resolution of the answers. Here the expert states
# each quantile to within 0.1. The miss of a consistent statement is not 0:
# the Monte Carlo sample and the fit add about 0.05 in this model. A tolerance
# below this error flags consistent statements.
#
# `el.plots.consistency` plots the mean miss. The whisker shows its 95 %
# interval, and the dashed line marks the tolerance. The plot gives two
# signals:
#
# + **The title** says "inconsistent" if an interval is above 1.
# + **A red bar** marks a statement that conflicts most with the others. The
#   largest miss is red. Another miss is also red if its interval is above 1
#   and it does not differ clearly from the largest miss.
#
# All plots use the same x-axis, so you can compare them.

# %%
# the expert states each quantile to within 0.1
TOLERANCE = {f"quantiles_{name}": 0.1 for name in "abc"}
diagnostics = {
    case: el.diagnostics.leave_one_out(eliobjs[case], references[case], TOLERANCE)
    for case in CASES
}

left = min(0.0, *(d["lower"].values.min() for d in diagnostics.values()))
right = max(d["upper"].values.max() for d in diagnostics.values())
margin = 0.1 * (right - left)
for case, diagnostic in diagnostics.items():
    fig, _ = el.plots.consistency(diagnostic, xlim=(left - margin, right + margin))
    fig.suptitle(case)

# %%
print(f"{'case':<14}" + "".join(f"{name:>22}" for name in "abc"))
for case, diagnostic in diagnostics.items():
    # a star marks a leading statement
    cells = [
        f"{'*' if lead else ''}{g:.1f} [{low:.1f}, {up:.1f}]"
        for g, low, up, lead in zip(
            diagnostic["mean_miss"].values,
            diagnostic["lower"].values,
            diagnostic["upper"].values,
            diagnostic["leading"].values,
        )
    ]
    print(f"{case:<14}" + "".join(f"{cell:>22}" for cell in cells))

# %% [markdown]
# ### How to read the results
#
# Read the plots in two steps:
#
# 1. **The title:** is the information inconsistent?
# 2. **The red bars:** which statement conflicts most with the others?
#
# A grey bar with an interval above 1 is not a conflict of its own. Without
# `a`, the fit still contains `c`, so the compromise misses `a`. The bar of
# `a` is then a signal from `c`.
#
# + **consistent:** all misses are about 0.5, below 1. There is no evidence
#   of a conflict.
# + **location:** `c` is red. Its miss is about twice the miss of `a` and
#   `b`. Ask the expert about `c` first.
# + **spread:** the same pattern. `c` is red.
# + **all differ:** `a` and `c` are red, with about equal misses. The miss of
#   `b` is below 1, because the compromise of `a` and `c` is at `b`. The
#   diagnostic cannot tell whether `a` or `c` is "wrong". It shows only that
#   they conflict with each other.
# + **not converged:** all bars are red, and the title says "inconsistent".
#   But the statements agree. The refits also stop early, so their
#   predictions miss. The loss in step 1 showed that the fit has not
#   converged. The verdict is therefore not valid. Fix the optimization
#   first.
#
# The diagnostic answers "which statement conflicts with the rest?". It does
# not answer "which statement is wrong?". Only the expert can answer that.
# The model can also be the cause: a statement that the model cannot
# reproduce together with the others can be correct.


# %% [markdown]
# ## Mathematical details
#
# This section is for advanced users and developers.
#
# ### The held-out miss
#
# Let $K$ be the number of statistics, and $L_k(\lambda)$ the loss of
# statistic $k$ at the hyperparameters $\lambda$. The training minimizes the
# weighted total loss
# $$
# L(\lambda) = \sum_{k=1}^{K} w_k L_k(\lambda), \qquad
# \lambda^* = \arg\min_\lambda L(\lambda).
# $$
# The refit without statistic $k$ sets $w_k = 0$:
# $$
# \lambda^*_{(-k)} = \arg\min_\lambda \sum_{j \neq k} w_j L_j(\lambda).
# $$
# Let $y_{k,q}$ be the expert values of statistic $k$, for example its
# quantiles $q$. Let $\hat y_{k,q}(\lambda)$ be the mean of the simulated
# statistic over the batches. The refit with seed $s$ gives the miss
# $$
# m_k^{(s)} = \frac{1}{\tau_k} \max_q
# \left| \hat y_{k,q}\big(\lambda^{*(s)}_{(-k)}\big) - y_{k,q} \right|,
# $$
# where $\tau_k$ is the tolerance of statistic $k$.
#
# + **Consistent statements.** A $\lambda^\circ$ exists that reproduces every
#   statistic. The refit without $k$ also finds $\lambda^\circ$. It predicts
#   $y_k$ up to the Monte Carlo error, so $m_k < 1$ if $\tau_k$ is above
#   this error.
# + **Statement $k$ conflicts.** No $\lambda$ reproduces $y_k$ together with
#   the others. The refit fits the others, so its prediction of $y_k$ misses.
#   Then $m_k > 1$ if the conflict is larger than $\tau_k$.
#
# ### Seed noise
#
# `baseline` fits with the seeds $s = 1, \dots, n$. `leave_one_out` refits
# with the same seeds. The mean miss $M_k = \bar m_k$ has the 95 % interval
# $$
# M_k \pm \frac{t_{0.975,\, n-1}\; \text{sd}(m_k)}{\sqrt{n}}.
# $$
# With $n = 3$ seeds, $t_{0.975,\, 2} = 4.30$. With $n = 5$, it is $2.78$.
# More seeds give a narrower interval. The title says "inconsistent" if the
# lower bound of an interval is above 1. The tolerance sets this threshold,
# not the noise. More seeds therefore do not make a small miss
# "inconsistent".
#
# ### The leading statements
#
# The statement $k^* = \arg\max_k M_k$ with the largest miss leads. The seeds
# are paired, so the difference $m_{k^*}^{(s)} - m_k^{(s)}$ of each seed also
# has a t-interval. Statement $k$ also leads if its own interval is above 1
# and the interval of the difference contains 0. The red bars are the
# leading statements.
#
# ### Why not the loss gain
#
# An earlier version compared the loss of the other statistics with and
# without statistic $k$:
# $$
# D_k = \sum_{j \neq k} \left[ L_j(\lambda^*) - L_j(\lambda^*_{(-k)}) \right].
# $$
# The refit without $k$ minimizes a sum with fewer terms. At the optimum,
# $D_k \geq 0$ for every expert, also for a consistent one. The loss also
# averages over batches, so it rewards a prior whose simulated statistics
# vary less. A refit without $k$ can use this freedom, so $D_k$ grows
# further. The test compared $D_k$ with 0, and its interval contained only
# the seed noise. More seeds therefore made every drop "significant". In a
# case study, a consistent expert got the verdict "inconsistent" with 10
# seeds, but not with 3. The miss compares with the tolerance instead.
# `leave_one_out` still returns the drop as `drop` and `gain`, as a
# description only.
#
# ### Relation to other methods
#
# Node-splitting (Presanis et al., 2013) checks the conflict between two
# groups of evidence. It compares the posterior of a parameter from each
# group. The held-out miss has the same idea: it predicts one statement from
# the others. Node-splitting takes the scale of the conflict from the
# posterior uncertainty. `elicito` has no posterior of $\lambda$, so the
# tolerance gives the scale.
#
# Lindley, Tversky and Brown (1979) treat stated judgements as coherent
# values plus an error. Hartmann et al. (2020) model the error of the expert
# with a Dirichlet distribution. Its concentration shows how well the prior
# reproduces the expert. The tolerance is the simplest form of such an error
# model: a fixed bound for each statistic.
#
# ### Cost
#
# `baseline` fits $n$ times. `leave_one_out` fits $K \cdot n$ times and
# simulates once from each refit. With the default $n = 3$ and $K = 3$
# statistics, this is 12 fits.

# %% [markdown]
# ## Limitations
#
# + **The tolerance is a choice.** The verdict depends on it. A tolerance
#   below the Monte Carlo error flags consistent statements. Increase `B`
#   and `num_samples` to make this error smaller.
# + **Conflict, not error.** A miss above 1 shows that a statement conflicts
#   with the others under this model. It does not show that the statement
#   is wrong.
# + **Two statistics.** With $K = 2$, each refit predicts one statistic from
#   the other. If they conflict, both misses are large. Both bars are then
#   red and do not show which statement is the cause.
# + **Converge first.** The refits use the settings of the fit. If the fit
#   has not converged, the refits stop early too, and their predictions
#   miss. The title then says "inconsistent" for consistent statements.
#   Check the loss before you trust the result.
# + **Parametric priors only.** The diagnostics support
#   `method="parametric_prior"`.

# %% [markdown]
# ## Recommendations
#
# 1. Fit the model. Look at `el.plots.loss`. If the loss still falls at the
#    last epoch, fix the optimization first, for example with more epochs or
#    a better initialization (see "How to choose an initialization method").
#    The diagnostics need a converged fit.
# 2. Look at `el.plots.elicits`. If a point is far from the diagonal, choose
#    a `tolerance` for each statistic. Then run `el.diagnostics.baseline` and
#    `el.diagnostics.leave_one_out`.
# 3. If the title says "no evidence of a conflict", check the optimization
#    again. A fit can stop at a poor point with a flat loss.
# 4. If the title says "inconsistent", discuss the red statements with the
#    expert. If two bars are red, these statements conflict with each
#    other.
# 5. For a narrower interval, give `baseline` more seeds. More seeds do not
#    change the threshold.
#
# ## References
#
# Hartmann, M., Agiashvili, G., Bürkner, P., & Klami, A. (2020). Flexible
# prior elicitation via the prior predictive distribution. In *Proceedings of
# the 36th Conference on Uncertainty in Artificial Intelligence*, PMLR 124,
# 1129-1138.
#
# Lindley, D. V., Tversky, A., & Brown, R. V. (1979). On the reconciliation
# of probability assessments. *Journal of the Royal Statistical Society:
# Series A*, 142, 146-180.
#
# Presanis, A. M., Ohlssen, D., Spiegelhalter, D. J., & De Angelis, D.
# (2013). Conflict diagnostics in directed acyclic graphs, with applications
# in Bayesian evidence synthesis. *Statistical Science*, 28(3), 376-397.
