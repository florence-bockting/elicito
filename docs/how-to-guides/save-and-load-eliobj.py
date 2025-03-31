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
# # Save and Load the eliobj
#
# Here we introduce a method for harmonising two timeseries.
# This part may be more unusual or unfamiliar
# to people used to working with arrays,
# so it serves as an introduction
# into some of the concepts used in this package.

# %% [markdown]
# ## Imports

# %%
# Imports
import os

from silence_tensorflow import silence_tensorflow

silence_tensorflow()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import copy
from typing import Any, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

# %% [markdown]
# ## Save an unfitted `eliobj` object
# ### Step 0: Load necessary libraries and functions/classes


# %%
# numeric, standardized predictor
def std_predictor(N: int, quantiles: list[Union[float, int]]) -> Any:
    """
    Compute design matrix

    Parameters
    ----------
    N
        number of observations

    quantiles
        quantiles to use in percentage

    Returns
    -------
    :
        design matrix
    """
    X = tf.cast(np.arange(N), tf.float32)
    X_std = (X - tf.reduce_mean(X)) / tf.math.reduce_std(X)
    X_sel = tfp.stats.percentile(X_std, quantiles)
    return X_sel


# implemented, generative model
class ToyModel:
    """
    Generative model
    """

    def __call__(self, prior_samples: Any, design_matrix: Any) -> dict[str, Any]:
        """
        Run the generative model

        Parameters
        ----------
        prior_samples
            samples from the prior distribution

        design_matrix
            design matrix

        Returns
        -------
        :
            dictionary with stored target quantities
        """
        B = prior_samples.shape[0]
        S = prior_samples.shape[1]

        # preprocess shape of design matrix
        X = tf.broadcast_to(design_matrix[None, None, :], (B, S, len(design_matrix)))
        # linear predictor (= mu)
        epred = tf.add(
            prior_samples[:, :, 0][:, :, None],
            tf.multiply(prior_samples[:, :, 1][:, :, None], X),
        )
        # data-generating model
        likelihood = tfd.Normal(
            loc=epred, scale=tf.expand_dims(prior_samples[:, :, -1], -1)
        )
        # prior predictive distribution (=height)
        ypred = likelihood.sample()

        # selected observations
        y_X0, y_X1, y_X2 = (ypred[:, :, 0], ypred[:, :, 1], ypred[:, :, 2])

        return dict(y_X0=y_X0, y_X1=y_X1, y_X2=y_X2)


# %% [markdown]
# ### Step 1: Create the eliobj

# %%
# define the generative model
model = el.model(
    obj=ToyModel, design_matrix=std_predictor(N=200, quantiles=[25, 50, 75])
)


# specify the model parameters and their prior distribution families
parameters = [
    el.parameter(
        name="beta0",
        family=tfd.Normal,
        hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)),
    ),
    el.parameter(
        name="beta1",
        family=tfd.Normal,
        hyperparams=dict(
            loc=el.hyper("mu1"),
            scale=el.hyper("sigma1", lower=0),  # TODO specify error message
        ),
    ),
    el.parameter(
        name="sigma",
        family=tfd.HalfNormal,
        hyperparams=dict(scale=el.hyper("sigma2", lower=0)),
    ),
]


# specify the target quantities and corresponding elicitation technique
targets = [
    el.target(
        name=f"y_X{i}",
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    )
    for i in range(3)
]


# use an oracle to simulate a ground truth for the expert data
expert = el.expert.simulator(
    ground_truth={
        "beta0": tfd.Normal(loc=5, scale=1),
        "beta1": tfd.Normal(loc=2, scale=1),
        "sigma": tfd.HalfNormal(scale=10.0),
    },
    num_samples=10_000,
)


# specify the optimizer for gradient descent
optimizer = el.optimizer(
    optimizer=tf.keras.optimizers.Adam, learning_rate=0.1, clipnorm=1.0
)


# define the trainer model with used approach, seed, etc.
trainer = el.trainer(method="parametric_prior", seed=0, epochs=4, progress=0)


# specify the initialization distribution, used to draw the initial values
# for the hyperparameters
initializer = el.initializer(
    method="sobol",
    loss_quantile=0,
    iterations=32,
    distribution=el.initialization.uniform(radius=1, mean=0),
)

eliobj = el.Elicit(
    model=model,
    parameters=parameters,
    targets=targets,
    expert=expert,
    optimizer=optimizer,
    trainer=trainer,
    initializer=initializer,
)

# %% [markdown]
# ### Step 2: Save the unfitted `eliobj` object
# Two approaches are possible:
# + automatic saving: `name` has to be specified.
#   + The results are then saved according to the
#   + following rule: `res/{method}/{name}_{seed}.pkl`
# + user-specific path: `file` has to be specified.
#   + The path can be freely specified by the user.

# %%
# use automatic saving approach
eliobj.save(name="m1")

# use user-specific file location
eliobj.save(file="results/m1_1")

# %% [markdown]
# ## Load and fit the *unfitted* eliobj

# %%
# load the eliobj
eliobj_m1 = el.utils.load("results/m1_1.pkl")

# fit the eliobj
eliobj_m1.fit()

# %% [markdown]
# ## Inspect the *fitted* eliobj
# + results for each epoch are stored in `history`
# + results saved only for the last epoch are stored in `results`

# %%
# information saved in the history object
print(eliobj_m1.history[0].keys())

# information saved in the results object
print(eliobj_m1.results[0].keys())

# %% [markdown]
# ### Save only subset of results
# Sometimes you don't want to save all possible results
# but only a relevant subset.
# You can control this by the arguments `save_configs_history` and
# `save_configs_results` in the `el.trainer` callable.
#
# **Example**
#
# I don't want to save information about the hyperparameter gradients and
# the single loss components.
# This can be done as follows:

# %%
eliobj_m2 = copy.deepcopy(eliobj_m1)

# update eliobj_m1 by changing the saving settings
trainer_new = el.trainer(method="parametric_prior", seed=0, epochs=4)
eliobj_m2.update(trainer=trainer_new)

# fit updated eliobj
eliobj_m2.fit(
    save_history=el.utils.save_history(
        hyperparameter_gradient=False, loss_component=False
    )
)

# inspect saved results
# note that loss_component and hyperparameter_gradient are not saved
eliobj_m2.history[0].keys()

# %% [markdown]
# ## Save and reload the *fitted* eliobj

# %%
# save the fitted object
eliobj_m2.save(name="m2")

# load the fitted object
eliobj_m2_reload = el.utils.load("./results/parametric_prior/m2_0.pkl")
eliobj_m2_reload.history[0]["loss"]

# %% [markdown]
# ## Q & A
# ### Can I fit an already fitted eliobj?

# %%
# eliobj_m2_reload.fit()

# prompt:
# elicit object is already fitted.
# Do you want to fit it again and overwrite the results?
# Press 'n' to stop process and 'y' to continue fitting.

# user input: n


# %% [markdown]
# ### Can I force re-fitting?
# Sometimes, especially when we only want to test something, it can be
# inconvenient to repeatedly confirm whether
# results should be overwritten. To address this issue, you can set
# `overwrite=True` to enable re-fitting without any prompts.

# %%
eliobj_m2_reload.fit(overwrite=True)

# %% [markdown]
# ### Can I overwrite an eliobj that already exits as file on disk?

# %%
# eliobj_m2_reload.save(name="m2")

# prompt:
# In provided directory exists already a file with identical name.
# Do you want to overwrite it?
# Press 'y' for overwriting and 'n' for abording.

# user input: n

# %% [markdown]
# ### Can I force overwriting an existing eliobj file?

# %%
eliobj_m2_reload.save(name="m2", overwrite=True)

# remove results folder
import shutil

shutil.rmtree("results")
