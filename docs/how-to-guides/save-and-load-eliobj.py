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
# # Save and load the eliobj
#
# This guide shows how to save an `eliobj` to disk and load it again.
# It works before and after the fit.

# %% tags=["hide"]
import os
import shutil

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# remove the files of an earlier run
shutil.rmtree("results", ignore_errors=True)

# %% [markdown]
# ## Imports

# %%
from typing import Any

import tensorflow_probability as tfp

import elicito as el
from elicito.fit import Fit

tfd = tfp.distributions

# %% [markdown]
# ## Set up the eliobj
#
# Any `eliobj` works.
# This guide uses a normal outcome with an unknown mean.


# %%
class Model:
    """Normal outcome with an unknown mean"""

    def __call__(self, prior_samples: Any) -> dict[str, Any]:
        """
        Simulate the outcome

        Parameters
        ----------
        prior_samples
            samples from the prior distributions

        Returns
        -------
        :
            dictionary with the model outputs
        """
        ypred = tfd.Normal(loc=prior_samples[:, :, 0], scale=1.0).sample()
        return dict(ypred=ypred)


parameters = [
    el.parameter(
        name="mu",
        family=tfd.Normal,
        hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)),
    )
]
targets = [el.quantiles("ypred", probs=(0.25, 0.50, 0.75))]
expert = {"ypred": [6.5, 8.0, 9.5]}

eliobj = el.Elicit(Model(), targets, parameters, epochs=100, verbose=False)

# %% [markdown]
# ## Save the eliobj
#
# Give a `name` or a `file`, not both.
#
# - `name="m1"` saves to `./results/{method}/{name}_{seed}.pkl`.
#   `method` is `parametric_prior` or `deep_prior`.
#   `seed` is the first of the `seeds`.
# - `file="results/m1"` saves to `results/m1.pkl`.

# %%
eliobj.save(name="m1")
eliobj.save(file="results/m1")

# %% [markdown]
# ## Load and fit the eliobj
#
# `el.Elicit.load` returns the saved `eliobj`.
# An unfitted `eliobj` needs the answers of the expert to fit.

# %%
eliobj_m1 = el.Elicit.load("results/m1.pkl")
fit = eliobj_m1.fit(expert)
fit

# %% [markdown]
# ## Save and load a fitted eliobj
#
# A fitted `eliobj` keeps its results and the answers of the expert.
# To inspect the loaded results, give the `eliobj` to `Fit`.

# %%
eliobj_m1.save(name="m2")

eliobj_m2 = el.Elicit.load("results/parametric_prior/m2_0.pkl")
Fit(eliobj_m2)

# %% [markdown]
# ## Fit a fitted eliobj again
#
# `fit()` raises a `ValueError` for a fitted `eliobj`.
# To replace the results, set `overwrite=True`.

# %%
fit = eliobj_m2.fit(expert, overwrite=True)

# %% [markdown]
# ## Replace a saved file
#
# `save()` raises a `FileExistsError` if the file exists.
# To replace the file, set `overwrite=True`.

# %%
eliobj_m2.save(name="m1", overwrite=True)
