# Choose an initialization method

This guide shows how to choose the start values of the hyperparameters.

> **Note:** CMA-ES, the default optimizer, ignores the initializer, with a warning.
> It starts at 0 on the unconstrained scale.
> A deep prior starts from the random weights of its flow.
> The initializer matters only for a parametric prior with a gradient optimizer, such as Adam.

Give the initializer to `el.Elicit`:

```python
eliobj = el.Elicit(model, targets, parameters, optimizer="adam", initializer=init)
```

All values are on the unconstrained scale.
A hyperparameter without bounds, such as a mean, keeps its value.
A hyperparameter with bounds, such as a scale with `lower=0`, gets its value through the inverse of its constraint.

## If you know the start values

Give them by name.
Nothing is cheaper.

```python
sigma0 = el.parameters.LowerBound(lower=0.0).forward(2.0)  # sigma0 = 2.0, unconstrained
init = el.initializer(hyperparams=dict(mu0=1.0, sigma0=sigma0))
```

`el.parameters.UpperBound` and `el.parameters.DoubleBound` transform the other bounds.

## If you know the region of the values

Sample a box around the region, and start from the best candidate.
`method` sets how the candidates are drawn: `"sobol"`, `"lhs"` or `"random"`.
`iterations` is the number of candidates (default 32).

```python
init = el.initializer(
    method="sobol",
    iterations=32,
    distribution=el.initializers.uniform(mean=0.0, radius=1.0),
)
```

If some candidates diverge in the first epochs, add `warmup_epochs`.
Each candidate then trains for `warmup_epochs` epochs before the ranking.

## If you do not know the region

Search for a start value before the first gradient step.

- `method="warmstart"` runs one local Nelder-Mead search from the centre of the box.
  It needs no gradient, so it cannot diverge.
  `el.initializer()` with no argument uses it, with 100 evaluations.
- `method="cmaes"` searches the whole box with CMA-ES (default: 500 evaluations).
  Use it for a wide box.
  `warmstart` and the samplers take the first usable basin, which is not always the best one.

```python
init = el.initializer(
    method="cmaes",
    distribution=el.initializers.uniform(mean=0.0, radius=6.0),
    iterations=300,
)
```

## Why the choice matters

On a loss surface with one minimum, every method reaches the same loss.
The start value matters where the training cannot repair it:
where a hyperparameter has an effect below the sampling error of the loss,
or where the draws of the model overflow.
[Initialization of the hyperparameters][initialization-of-the-hyperparameters] shows both cases.
