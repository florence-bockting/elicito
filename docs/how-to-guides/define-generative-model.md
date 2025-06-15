```python
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions
```

# Specify the generative model object

Here we introduce how to specify the generative model that
is passed as input argument to the `el.model()` function.

## General workflow
The geneartive model must be a `Python class` with the
following *input-output* structure:

**Inputs:**

+ necessary input: `prior_samples`
+ optional inputs: any argument required by the user-defined Python class

The *optional* arguments have to be specified as keyword
arguments in the `el.model()` function.

**Outputs:**

+ necessary format: dictionary with *keys* referring
  to the name of the respective target quantity and *values* to the
  corresponding `tf.tensor` object

```
# pseudo code
class GenerativeModel:
  def __call__(self, prior_samples, **kwargs):
      # specify here your generative process: which takes the samples
      # from the prior as input and computes the target quantities
        # (e.g., observations from the outcome variable)
        target = ...

        # return the computed target quantities which
        # should be queried from the domain expert
        return dict(target=target)
```
## Example: Probabilistic model notation
$$
\begin{align*}
    (\beta_0, \beta_1) &\sim \text{Normal}(\mu_i, \sigma_i) \quad \text{ for } i=0,1 \\
    \boldsymbol{\mu} &= \boldsymbol{\beta}\textbf{X}^\top \\
    \textbf{y}_{pred} &\sim \text{Normal}(\boldsymbol{\mu}, \textbf{1.})
\end{align*}
$$

## Target quantities
We assume, we want to query the domain expert regarding the outcome variable.
Specifically, we ask the expert regarding three values of the predictor variable:

+ $y_{pred} \mid X_{1}$
+ $y_{pred} \mid X_{2}$
+ $y_{pred} \mid X_{3}$

Let the corresponding design matrix be

$$
\textbf{X}=\begin{bmatrix} 1. & -1 \\ 1. & 0. \\ 1. & 1. \end{bmatrix}
$$

## Computational model implementation
### Create a Python class

```python
class ExampleModel1:
    def __call__(self, prior_samples, X):

        # shape=(B,num_samples,num_obs)
        mu=tf.matmul(prior_samples, X, transpose_b=True)

        # shape=(B,num_samples,num_obs)
        ypred=tfd.Normal(mu, 1.).sample()

        return dict(y_X1=ypred[:,:,0], # shape=(B,num_samples)
                    y_X2=ypred[:,:,1], # shape=(B,num_samples)
                    y_X3=ypred[:,:,2]) # shape=(B,num_samples)
```

### Look behind the scenes
Let us use the `ExampleModel1` and generate some predictions based on artificial prior distributions.

1. Draw prior samples from the following "ground truth": $\beta_0 \sim \text{Normal}(1., 0.8)$
   and $\beta_1 \sim \text{Normal}(2., 1.5)$
2. Define the design matrix $\textbf{X}$
3. Instantiate the generative model
4. Simulate from the generative model

```python
# define number of batches and draws from prior distributions
B, num_samples = (1,10)

# sample from priors
prior_samples = tfd.Normal([1,2], [0.8, 1.5]).sample((B,num_samples))
print("(Step 1) prior samples:\n", prior_samples)

# define the design matrix
X = tf.constant([[1.,-1.],[1.,0], [1.,1.]])
print("\n(Step 2) design matrix:\n", X)

# create an instance of the generative model class
model_instance = ExampleModel1()
print("\n(Step 3) instantiated model:\n", model_instance)

# simulate from the generative model
ypred = model_instance(prior_samples, X)
print("\n(Step 4) samples from outcome variable:\n", ypred)
```

### Implementation of the `eliobj`
The corresponding implementation of the `eliobj` would then look
as follows:

```
eliobj = el.Elicit(
    model=el.model(
        obj=ExampleModel1,    # model class
        X=X                   # additional input argument for model class
        ),
    parameters=[
        el.parameter(
            name=f"beta{i}",
            family=tfd.Normal,
            hyperparams=dict(
                loc=el.hyper(f"mu{i}"),
                scale=el.hyper(f"sigma{i}", lower=0)
                )
        ) for i in range(2)
    ],
    targets=[
        el.target(
            name=f"y_X{i}",
            query=el.queries.quantiles((.05, .25, .50, .75, .95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0
        ) for i in range(1,4)
    ],
    ...
)
```
