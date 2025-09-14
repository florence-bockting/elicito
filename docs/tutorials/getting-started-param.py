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
# # Getting started with parameteric priors
#
# Here we introduce how to specify the elicitation method for
# a parametric prior.
#

# %% [markdown]
# ## Imports

# %%
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import copy
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions

# %% [markdown]
# ## The Model
# ### Probabilistic model
# $$
# \begin{align*}
#     \beta_0 &\sim \text{Normal}(\mu_0, \sigma_0) \\
#     \beta_1 &\sim \text{Normal}(\mu_1, \sigma_1) \\
#     \sigma &\sim \text{HalfNormal}(\sigma_2) \\
#     \mu &= \beta_0 + \beta_1X \\
#     y_{pred} &\sim \text{Normal}(\mu, \sigma)
# \end{align*}
# $$
#
# ### Implementation
# #### Predictor


# %%
# create a predictor ranging from 1 to 200
# standardize predictor
# select the 25th (X0), 50th (X1), and 75th (X2) quantile of the std.
# predictor for querying the expert
def X_design(N: int, quantiles: list[float]) -> np.ndarray:
    """
    Compute design matrix

    Parameters
    ----------
    N
        number of observations

    quantiles
        list of quantiles

    Returns
    -------
    :
        design matrix
    """
    X = tf.cast(np.arange(N), tf.float32)
    X_std = X / tf.math.reduce_std(X)
    X_sel = tfp.stats.percentile(X_std, quantiles)
    X_design = tf.stack([tf.ones(X_sel.shape), X_sel], -1)
    return X_design


X_design(N=30, quantiles=[25, 50, 75])

# %% [markdown]
# #### Generative model


# %%
class ToyModel:
    """
    generative model
    """

    def __call__(self, prior_samples: Any, design_matrix: Any) -> dict[str, Any]:
        """
        Compute target quantities from generative model

        Parameters
        ----------
        prior_samples
            prior samples

        design_matrix
            design matrix

        Returns
        -------
        :
            dictionary with target quantities
        """
        # linear predictor
        epred = tf.matmul(prior_samples[:, :, :-1], design_matrix, transpose_b=True)

        # data-generating model
        likelihood = tfd.Normal(
            loc=epred, scale=tf.expand_dims(prior_samples[:, :, -1], -1)
        )
        # prior predictive distribution
        ypred = likelihood.sample()

        # selected observations
        y_X0, y_X1, y_X2 = (ypred[:, :, 0], ypred[:, :, 1], ypred[:, :, 2])

        return dict(y_X0=y_X0, y_X1=y_X1, y_X2=y_X2, ypred=ypred, epred=epred)


# %% [markdown]
# ### Generative model

# %%
# specify the model
model = el.model(obj=ToyModel, design_matrix=X_design(N=30, quantiles=[25, 50, 75]))

# %% [markdown]
# ### Model parameters
# + intercept with normal prior $\beta_0$
# + slope with normal prior $\beta_1$
# + random noise with halfnormal prior $\sigma$
#
# **To be learned hyperparameters**
# $
# \lambda = (\mu_0, \sigma_0, \mu_1, \sigma_1, \sigma_2)
# $
#
# + scale parameters ($\sigma_0, \sigma_1, \sigma_2$) are constrained to be positive

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


# %% [markdown]
# ### Target quantities and elicitation techniques
# **Target quantities**
# + query expert regarding **prior predictions** $y \mid X_{i}$ with $i$
# being the 25th, 50th, and 75th quantile of the predictor.
#
# **Elicitation technique**
# + query each observation using **quantile-based elicitation** using
# $Q_p(y \mid X)$ for $p=25, 50, 75$
#
# **Specifying discrepancy measure and weight for single loss components**
# + prior predictions $y \mid X_i$:
#     + discrepancy measure: Maximum Mean Discrepancy with energy kernel
#     + weight of loss component: 1.0
# + $R^2$:
#     + L2 loss
#     + weight of loss component: 0.5


# %%
def custom_r2(ypred, epred):
    """Compute coefficient of determination"""
    var_epred = tf.math.reduce_variance(epred, -1)
    # variance of difference between ypred and epred
    var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
    var_total = var_epred + var_diff
    # variance of linear predictor divided by total variance
    return tf.divide(var_epred, var_total)


targets = [
    el.target(
        name="y_X0",
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    ),
    el.target(
        name="y_X1",
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    ),
    el.target(
        name="y_X2",
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    ),
    el.target(
        name="R2",
        query=el.queries.quantiles((0.05, 0.50, 0.95)),
        loss=el.losses.L2,
        weight=0.5,
        target_method=custom_r2,
    ),
]

# %% [markdown]
# ### Expert elicitation
#
# Instead of querying a "real" expert, we define a ground truth (i.e., oracle)
# and simulate the oracle-elicited statistics

# %%
# specify ground truth
ground_truth = {
    "beta0": tfd.Normal(loc=5, scale=1),
    "beta1": tfd.Normal(loc=2, scale=1),
    "sigma": tfd.HalfNormal(scale=7.0),
}

# define oracle
expert = el.expert.simulator(ground_truth=ground_truth, num_samples=10_000)
# %% [markdown]
# ### Training
# Learn prior distributions based on expert data

# %%
eliobj = el.Elicit(
    model=model,
    parameters=parameters,
    targets=targets,
    expert=expert,
    optimizer=el.optimizer(
        optimizer=tf.keras.optimizers.Adam, learning_rate=0.1, clipnorm=1.0
    ),
    trainer=el.trainer(method="parametric_prior", seed=2025, epochs=100, progress=0),
    initializer=el.initializer(
        method="sobol",
        iterations=32,
        distribution=el.initialization.uniform(radius=2.0, mean=0.0),
    ),
)

# %% [markdown]
# **Fit eliobj**

# %%
eliobj.fit(parallel=el.utils.parallel())

# %%
tf.stack([eliobj.history[i]["loss_component"] for i in range(len(eliobj.history))],0)

# %%
tf.stack(eliobj.history[0]["loss_tensor_model"],1)

# %%
eliobj.results[0].keys()

# %%
["_".join(k.split("_")[:-2]) for k in eliobj.results[0]["loss_tensor_model"].keys()]

# %%
import itertools

eliobj.results[0].keys()

# %%
eliobj.results[0]["prior_samples"]

# %%
da_list = []
for k in eliobj.results[0]["elicited_statistics"].keys():
    elicits = tf.stack([eliobj.results[i]["elicited_statistics"][k] for i in range(len(eliobj.results))],0)
    da_elicits = xr.DataArray(
        data=elicits,
        dims=["replications", "batch", "summary_dim"],
        coords=dict(
            replications=tf.range(elicits.shape[0]),
            batch=tf.range(elicits.shape[1]),
            summary_dim=tf.range(elicits.shape[-1])
        ),
        name=k
    )
    da_list.append(da_elicits)


# %%
tf.range(elicits.shape[1])

# %% jupyter={"outputs_hidden": true}
[eliobj.results[0]["elicited_statistics"][k] for k in list(eliobj.results[0]["elicited_statistics"].keys())]

# %%
attr_keys = dict()
attr_keys["description"] = "Elicited summaries as simulated by the model."
attr_keys["warning"] = "The dimension 'elicited_summary' can not be used for selection of a specific elicited summary! The dimension is only provided to provide a fast summary information. In order to select a specific elicited summary use the corresponding key for the dictionary data object. Keys and shape of items are provided below"
for k in eliobj.results[0]["elicited_statistics"]:
    attr_keys[k] = eliobj.results[0]["elicited_statistics"][k].shape


da_elicits = xr.DataArray(
    [eliobj.results[i]["elicited_statistics"] for i in range(len(eliobj.results))],
    dims=["replication"],
    coords=dict(
        replication=range(len(eliobj.results))
    ),
    attrs=attr_keys
)

da_elicits=da_elicits.expand_dims(elicited_summary=len(eliobj.results[0]["elicited_statistics"]), axis=-1)
da_elicits=da_elicits.assign_coords(elicited_summary=list(eliobj.results[0]["elicited_statistics"].keys()))

# %% jupyter={"outputs_hidden": true}

# %%
targets = tf.stack([[eliobj.results[i]["target_quantities"][k] for i in range(len(eliobj.results))] for k in eliobj.results[0]["target_quantities"].keys()],-1)

da_targets = xr.DataArray(
        data=targets,
        dims=["replication", "batch", "sample", "target_quantity"],
        coords=dict(
            replication=tf.range(targets.shape[0]),
            batch=tf.range(targets.shape[1]),
            sample=tf.range(targets.shape[2]),
            target_quantity=list(eliobj.results[0]["target_quantities"].keys())
        ),
        name="target quantities",
        attrs=dict(
            description="Target quantities"
        )
    )

attr_keys = dict()
attr_keys["description"] = "Elicited summaries as simulated by the model."
attr_keys["warning"] = "The dimension 'elicited_summary' can not be used for selection of a specific elicited summary! The dimension is only provided to provide a fast summary information. In order to select a specific elicited summary use the corresponding key for the dictionary data object. Keys and shape of items are provided below"
for k in eliobj.results[0]["elicited_statistics"]:
    attr_keys[k] = eliobj.results[0]["elicited_statistics"][k].shape


da_elicits = xr.DataArray(
    [eliobj.results[i]["elicited_statistics"] for i in range(len(eliobj.results))],
    dims=["replication"],
    coords=dict(
        replication=range(len(eliobj.results))
    ),
    attrs=attr_keys
)

da_elicits=da_elicits.expand_dims(elicited_summary=len(eliobj.results[0]["elicited_statistics"]), axis=-1)
da_elicits=da_elicits.assign_coords(elicited_summary=list(eliobj.results[0]["elicited_statistics"].keys()))

priors = tf.stack([eliobj.results[i]["prior_samples"] for i in range(len(eliobj.results))],0)

da_priors = xr.DataArray(
        data=priors,
        dims=["replication", "batch", "sample", "parameter"],
        coords=dict(
            replication=tf.range(priors.shape[0]),
            batch=tf.range(priors.shape[1]),
            sample=tf.range(priors.shape[2]),
            parameter=[eliobj.parameters[k]["name"] for k in range(len(eliobj.parameters))]

        ),
        name="prior samples",
        attrs=dict(
            description="Prior samples"
        )
    )

priors_exp = tf.stack([eliobj.results[i]["expert_prior_samples"] for i in range(len(eliobj.results))],0)

da_priors_expert = xr.DataArray(
        data=priors_exp,
        dims=["replication", "batch", "sample_truth", "parameter"],
        coords=dict(
            replication=tf.range(priors_exp.shape[0]),
            batch=tf.range(priors_exp.shape[1]),
            sample_truth=tf.range(priors_exp.shape[2]),
            parameter=[eliobj.parameters[k]["name"] for k in range(len(eliobj.parameters))]

        ),
        name="prior samples from ground truth",
        attrs=dict(
            description="Prior samples from ground truth (used for the construction of expert-elicited summaries for self-consistency validation)"
        )
    )

da_seed = xr.DataArray(
    data=[eliobj.results[i]["seed"] for i in range(len(eliobj.results))],
    dims=["replication"],
    coords=dict(
        replication=tf.range(init_loss.shape[0])
    ),
    name="seed",
    attrs=dict(
        description="seed per replications (for reproducibility)"
    )
)

ds = xr.Dataset(
    data_vars=dict(
        prior_samples=da_priors,
        target_quantities=da_targets,
        elicited_summaries=da_elicits,
        prior_samples_true=da_priors_expert,
        seed=da_seed
    ),
    attrs=dict(
        global_seed = eliobj.trainer["seed"]
    )
)

# %%
xr.DataArray()

# %%
xr.Dataset(
            data_vars = dict(
                quantiles_y_X0=xr.DataArray(eliobj.results[0]["elicited_statistics"]["quantiles_y_X0"], dims=["batch", "summary_dim0"]),
                quantiles_y_X1=xr.DataArray(eliobj.results[0]["elicited_statistics"]["quantiles_y_X1"], dims=["batch", "summary_dim0"]),
                quantiles_y_X2=xr.DataArray(eliobj.results[0]["elicited_statistics"]["quantiles_y_X2"], dims=["batch", "summary_dim0"]),
                quantiles_R2=xr.DataArray(eliobj.results[0]["elicited_statistics"]["quantiles_R2"], dims=["batch", "summary_dim1"])
        )
)

# %%
ds = xr.Dataset(
    dict(
        prior_samples=da_priors,
        target_quantities=da_targets,
        elicited_summaries = xr.Dataset(
            dict(
                quantiles_y_X0=xr.DataArray(eliobj.results[0]["elicited_statistics"]["quantiles_y_X0"], dims=["batch", "summaries_dim0"]),
                quantiles_y_X1=xr.DataArray(eliobj.results[0]["elicited_statistics"]["quantiles_y_X1"], dims=["batch", "summaries_dim0"]),
                quantiles_y_X2=xr.DataArray(eliobj.results[0]["elicited_statistics"]["quantiles_y_X2"], dims=["batch", "summaries_dim0"]),
                quantiles_R2=xr.DataArray(eliobj.results[0]["elicited_statistics"]["quantiles_R2"], dims=["batch", "summaries_dim1"])
        )
).to_dataarray()
    )
)

ds.elicited_summaries.sel(variable="quantiles_y_X0")

# %% [markdown]
# ## Results
# ### Initialization of hyperparameters

# %%
el.plots.initialization(eliobj, cols=5);

# %% [markdown]
# ### Convergence - Loss

# %%
el.plots.loss(eliobj);

# %% [markdown]
# ### Convergence - hyperparameters

# %%
el.plots.hyperparameter(eliobj);

# %% [markdown]
# ### Expert expectations

# %%
el.plots.elicits(eliobj, cols=4);

# %% [markdown]
# ### Learned prior distributions

# %%
el.plots.prior_marginals(eliobj, cols=3);

# %% [markdown]
# ## Add-on: Shared parameters

# %%
# create a copy of eliobj
eliobj_shared = copy.deepcopy(eliobj)

# share sigma hyperparameter of intercept and slope parameter
parameters_shared = [
    el.parameter(
        name="beta0",
        family=tfd.Normal,
        hyperparams=dict(
            loc=el.hyper("mu0"), scale=el.hyper("sigma1", lower=0, shared=True)
        ),
    ),
    el.parameter(
        name="beta1",
        family=tfd.Normal,
        hyperparams=dict(
            loc=el.hyper("mu1"), scale=el.hyper("sigma1", lower=0, shared=True)
        ),
    ),
    el.parameter(
        name="sigma",
        family=tfd.HalfNormal,
        hyperparams=dict(scale=el.hyper("sigma2", lower=0)),
    ),
]

# update parameters in eliobj
eliobj_shared.update(parameters=parameters_shared)

# refit the model
eliobj_shared.fit()

# %% [markdown]
# ### Results
# #### Initialization of hyperparamters

# %%
el.plots.initialization(eliobj_shared, cols=4);

# %% [markdown]
# #### Convergence - Loss

# %%
el.plots.loss(eliobj_shared);

# %% [markdown]
# #### Convergence - Hyperparameters

# %%
el.plots.hyperparameter(eliobj_shared);

# %% [markdown]
# #### Elicited statistics

# %%
el.plots.elicits(eliobj_shared, cols=4);

# %% [markdown]
# #### Learned prior distributions

# %%
el.plots.prior_marginals(eliobj_shared);

# %% [markdown]
# ## Add-on: Use expert data as input

# %%
# create a copy of eliobj
eliobj_dat = copy.deepcopy(eliobj)

# use dictionary of elicited expert data (instead of simulating data)
expert_dat = {
    "quantiles_y_X0": [-12.5, -0.6, 3.3, 7.1, 19.1],
    "quantiles_y_X1": [-11.2, 1.5, 5.0, 8.8, 20.4],
    "quantiles_y_X2": [-9.3, 3.1, 6.8, 10.5, 23.3],
    "quantiles_R2": [0.001, 0.02, 0.09, 0.41, 0.96],
}

# update expert in eliobj
eliobj_dat.update(expert=el.expert.data(dat=expert_dat))

# refit the model
eliobj_dat.fit()

# %% [markdown]
# ### Results
# #### Initialization of hyperparameters

# %%
el.plots.initialization(eliobj_dat, cols=5);

# %% [markdown]
# #### Convergence - Loss

# %%
el.plots.loss(eliobj_dat);

# %% [markdown]
# #### Convergence - Hyperparameters

# %%
el.plots.hyperparameter(eliobj_dat, cols=5);

# %% [markdown]
# #### Elicited statistics

# %%
el.plots.elicits(eliobj_dat, cols=4);

# %% [markdown]
# #### Learned prior distributions

# %%
el.plots.prior_marginals(eliobj_dat);
