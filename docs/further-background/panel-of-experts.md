# A panel of experts

This page explains how `elicito` learns one prior from several experts.
The tutorial [A prior for a panel of experts][a-prior-for-a-panel-of-experts] shows the steps.

## Three ways to pool

`pool=` of [`Elicit`][elicito.elicit.Elicit] sets how the answers of the experts combine.

- `pool=None` fits each expert alone.
  The experts share nothing, so there is no population prior.
- `pool="complete"` learns one prior for all experts.
  The disagreement in the panel becomes uncertainty of this one prior.
- `pool=el.partial(over=...)` gives each expert a prior of their own.
  The priors share a panel value, and each expert has an offset from it.

Partial pooling is the middle way.
It keeps the disagreement as a separate quantity, the spread of the offsets.

## The hierarchical prior

Take the linear regression of the tutorials.
With `pool=el.partial(over=["mu0", "mu1"])`, expert $e = 1, \dots, E$ has the prior

$$
\begin{align*}
    \beta_{j,e} &\sim \text{Normal}(\mu_j + \delta_{j,e}, \sigma_j)
        \qquad j = 0, 1 \\
    \delta_{j,e} &\sim \text{Normal}(0, \tau_j)
\end{align*}
$$

- $\mu_j$ is the panel value. All experts share it.
- $\sigma_j$ is the uncertainty of one expert. All experts share it.
- $\delta_{j,e}$ is the offset of expert $e$ from the panel value.
- $\tau_j$ is the spread of the offsets: the disagreement in the panel.

A new expert from the same panel has the **population prior**

$$
\beta_j \sim \text{Normal}\left(\mu_j, \sqrt{\sigma_j^2 + \tau_j^2}\right).
$$

The disagreement $\tau_j$ makes it wider than the prior of one expert.

`elicito` adds the offsets on the unconstrained scale of a hyperparameter.
A hyperparameter with bounds then stays inside its bounds for every expert.

## Why the loss needs a term for the offsets

The answers of expert $e$ depend on $\mu_j + \delta_{j,e}$ only.
So the answers cannot separate the panel value from the offsets.
Add 1 to $\mu_j$ and subtract 1 from every offset: the answers do not change.
A term in the loss must hold the offsets near 0.

## Why one fit cannot learn the spread

The obvious term is the negative log density of the offsets,
$\sum_{j,e} \left[\log \tau_j + \delta_{j,e}^2 / (2 \tau_j^2)\right]$.

Set every offset to 0.
The term is then $E \log \tau_j$ for each coefficient.
It goes to $-\infty$ as $\tau_j$ goes to 0, but the discrepancy at $\delta = 0$ stays finite.
So the search makes all experts equal and $\tau_j$ zero.
This is the known collapse of a joint mode in a hierarchical model.

## Two stages

`elicito` keeps $\tau_j$ fixed in each fit.
The term is then a ridge, $\sum_{j,e} \delta_{j,e}^2 / (2 \tau_j^2)$, and it has a minimum.

1. Fit with a weak ridge, $\tau_j = 10$. The offsets follow the experts.
2. Estimate $\hat\tau_j$ as the standard deviation of the learned offsets.
   A floor of $10^{-3}$ keeps the ridge finite.
3. Fit again with $\tau_j = \hat\tau_j$.
   The ridge now pulls each offset toward 0, as far as the disagreement in the panel allows.

`el.partial(over=..., tau=...)` with a given spread skips the first two stages.

The loss is a discrepancy, not a likelihood.
So the weight 1 of the ridge is a choice, not a result.

## How large is the panel?

Five offsets are a small sample.
Their standard deviation is the best estimate of $\tau_j$ that any method can reach from this panel.
It is not the true $\tau_j$.
A larger panel gives a better estimate of the disagreement.
