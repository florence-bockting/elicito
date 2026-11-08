# Supported PyMC functions

The PyMC adapter translates a PyMC model into a TensorFlow model. It
translates each PyTensor operation (op) of the model graph with one rule.
This page lists the functions that have a rule. The column *Op* gives the
name of the op in the graph.

A function that is not in the list raises a `NotImplementedError` with the
name of the op, for example `The adapter does not support the op Log1p.`

See [Use a PyMC model][use-a-pymc-model] for the rules of the model.

## Priors

`elicito` learns the hyperparameters of these priors. Each input of a prior
is a named `pm.Data` node.

| Function | Hyperparameters | Op |
| :-- | :-- | :-- |
| `pm.Normal` | `mu`, `sigma` | `NormalRV` |
| `pm.HalfNormal` | `sigma` | `HalfNormalRV` |
| `pm.Uniform` | `lower`, `upper` | `UniformRV` |
| `pm.Beta` | `alpha`, `beta` | `BetaRV` |
| `pm.Gamma` | `alpha`, `beta` | `GammaRV` |

## Other random variables

The translated model samples these random variables. Use them for the
likelihood, or for a random variable without a `pm.Data` input, such as a
hierarchical effect.

| Function | Op |
| :-- | :-- |
| `pm.Normal`, `pm.HalfNormal`, `pm.Uniform`, `pm.Beta`, `pm.Gamma` | `NormalRV`, `HalfNormalRV`, `UniformRV`, `BetaRV`, `GammaRV` |
| `pm.TruncatedNormal` | `TruncatedNormalRV` |
| `pm.Exponential` | `ExponentialRV` |
| `pm.Bernoulli` | `BernoulliRV` |
| `pm.Binomial` | `BinomialRV` |
| `pm.Categorical` | `CategoricalRV` |
| `pm.Dirichlet` | `DirichletRV` |
| `pm.Multinomial` | `MultinomialRV` |
| `pm.MvNormal` | `MvNormalRV` |
| `pm.Wishart` | `WishartRV` |
| `pm.LKJCholeskyCov` | `_LKJCholeskyCovRV` |

## Arithmetic and math functions

| Function | Op |
| :-- | :-- |
| `a + b` | `Add` |
| `a * b` | `Mul` |
| `a - b` | `Sub` |
| `-a` | `Neg` |
| `a / b` | `TrueDiv` |
| `pt.reciprocal(a)` | `Reciprocal` |
| `a ** b` | `Pow` |
| `pt.exp(a)` | `Exp` |
| `pt.log(a)` | `Log` |
| `pt.log1p(a)` | `Log1p` |
| `pt.sqrt(a)` | `Sqrt` |
| `pt.abs(a)` | `Abs` |
| `pt.sign(a)` | `Sign` |
| `pt.erf(a)` | `Erf` |
| `pt.sigmoid(a)`, `pm.math.invlogit(a)` | `Sigmoid` |
| `pt.minimum(a, b)` | `Minimum` |
| `pt.maximum(a, b)` | `Maximum` |
| `pt.clip(a, lower, upper)` | `Clip` |

## Comparison and selection

| Function | Op |
| :-- | :-- |
| `pt.eq(a, b)` | `EQ` |
| `a < b` | `LT` |
| `a > b` | `GT` |
| `pt.switch(condition, a, b)` | `Switch` |
| `pt.cast(a, dtype)` | `Cast` |

## Reductions

| Function | Op |
| :-- | :-- |
| `pt.sum(a, axis=...)` | `Sum` |
| `pt.prod(a, axis=...)` | `Prod` |
| `pt.argsort(a, axis=...)` | `ArgSortOp` |

## Shape and indexing

| Function | Op |
| :-- | :-- |
| `a[1:]`, `a[::-1]`, `a[0]` | `Subtensor` |
| `a[index_array]`, `a[:, index_array]` | `AdvancedSubtensor` |
| `pt.set_subtensor(a[index_array], b)`, `pt.inc_subtensor(a[index_array], b)` | `AdvancedIncSubtensor` |
| `pt.concatenate([a, b], axis=...)` | `Join` |
| `pt.stack([scalar_a, scalar_b])` | `MakeVector` |
| `pt.reshape(a, shape)` | `Reshape` |
| `pt.repeat(a, n)`, `pt.tile(a, reps)` | `Alloc` |
| `pt.fill(a, value)` | `Second` |
| `a.shape` | `Shape` |
| `a.T`, `a[None]` | `DimShuffle` |

## Linear algebra

| Function | Op |
| :-- | :-- |
| `pt.dot(a, b)` | `Dot` |
| `pt.linalg.inv(a)` | `MatrixInverse` |

`pt.linalg.inv` of a singular matrix, or of a matrix with NaN, gives NaN in
the translated model. PyTensor raises an error instead. During a CMA-ES
fit, the search gives this candidate a penalty, and the fit goes on.

## Ops that PyTensor adds

PyTensor adds these ops to the graph by itself. You do not call them.

| Op |
| :-- |
| `ScalarFromTensor` |
| `ViewOp` |

## Numerical precision

PyMC computes in float64. The translated model computes in float32. A value
near 0 or 1 can therefore round to exactly 0 or 1 in the translated model.

!!! warning
    A probability `p` can be exactly 1 in float32. `pt.log(1 - p)` is then
    `-inf`, and `0 * -inf` gives `NaN`. Clip the argument of the logarithm,
    for example `pt.log(pt.clip(1 - p, 1e-12, 1.0))`.
