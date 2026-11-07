# Specify the generative model

This guide shows how to write the generative model that you give to
[`Elicit`][elicito.elicit.Elicit].
The model takes samples from the prior and returns the quantities that the expert judges.

If your model is a PyMC model, give it to `el.Elicit` directly.
[Use a PyMC model][use-a-pymc-model] shows how.

## Write the model as a callable

The model is a function, or an instance of a class with a `__call__` method.
It must follow these rules:

- It takes the argument `prior_samples`.
  The shape is `(B, num_samples, number of parameters)`.
  The columns follow the order of the `parameters` list.
- It returns a dictionary.
  Each key is the name of a model output.
  Each value has the shape `(B, num_samples)`, or `(B, num_samples, k)`.
- A target reads a model output by its name, or by `output=`.
  For an output with a last axis of `k` values, `select=` picks one of them.

```python
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class Model:
    """Linear regression with three design points"""

    def __call__(self, prior_samples, design_matrix):
        # shape (B, num_samples, 3)
        epred = tf.matmul(prior_samples, design_matrix, transpose_b=True)
        ypred = tfd.Normal(epred, 1.0).sample()
        return dict(ypred=ypred, epred=epred)
```

## Bind the other arguments

`elicito` passes `prior_samples` only.
Bind every other argument with `functools.partial`.
Give the class instance, not the class.

```python
from functools import partial

X = tf.constant([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
model = partial(Model(), design_matrix=X)
```

## Make the samples reproducible

If the model takes an argument `seed`, `elicito` passes a stateless seed to it.
Give it to every `sample()` call in the model.
The outputs then depend on the `seeds` of `Elicit` only.

```python
class SeededModel:
    def __call__(self, prior_samples, design_matrix, seed=None):
        epred = tf.matmul(prior_samples, design_matrix, transpose_b=True)
        ypred = tfd.Normal(epred, 1.0).sample(seed=seed)
        return dict(ypred=ypred)
```

## Test the model before the fit

Call the model with a small sample from known priors.
Check the shape of each output.

```python
prior_samples = tfd.Normal([1.0, 2.0], [0.8, 1.5]).sample((1, 10))
outputs = model(prior_samples)
{name: value.shape for name, value in outputs.items()}
# {'ypred': TensorShape([1, 10, 3]), 'epred': TensorShape([1, 10, 3])}
```

## Give the model to `Elicit`

```python
import elicito as el

parameters = [
    el.parameter(
        name=f"beta{i}",
        family=tfd.Normal,
        hyperparams=dict(loc=el.hyper(f"mu{i}"), scale=el.hyper(f"sigma{i}", lower=0)),
    )
    for i in range(2)
]
targets = [
    el.quantiles(f"y_X{i}", output="ypred", select=i, probs=(0.05, 0.25, 0.50, 0.75, 0.95))
    for i in range(3)
]
eliobj = el.Elicit(model, targets, parameters)
```
