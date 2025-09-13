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

# %%
import os
import copy

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import tensorflow_probability as tfp
import elicito as el
import numpy as np
import pandas as pd

tfd = tfp.distributions


# %% [markdown]
# # Case Study: Human Growth Model
#
# + re-implementation according to [Manderson & Goudie (2023)](https://arxiv.org/abs/2303.08528)
# + height measured at age $t_m$ (in years) with corresponding measurement $y_m$ (in cm)
#
# ## Model
# $$
# \begin{align*}
#     (h_0, h_1, s_1, s_0, \gamma) &\sim \text{LogNormal}(\mu_k, \sigma_k) \quad \text{for } k=1,\ldots, 5\\
#     \sigma_y &\sim \text{LogNormal}(0, 0.2)\\
#     \mu &= h_1 - \frac{2(h_1 - h_0)}{\exp(s_0(t_m - \gamma)) + \exp(s_1(t_m - \gamma))} \\
#     y_m &\sim \text{Normal}(\mu, \sigma_y)
# \end{align*}
# $$
# + assuming $0 < h_0 < h_1$ and $0 < s_0 < s_1$
#
# ## Data
#
# + ages: $t = (2, 8, 13, 18)$
#
# Quantiles of
# $$
# \begin{align*}
#     p(y \mid t_1 = 2) &= \text{Normal}(88, 3.5)\\
#     p(y \mid t_2 = 8) &= \text{Normal}(130, 5.5)\\
#     p(y \mid t_3 = 13) &= \text{Normal}(160, 8)\\
#     p(y \mid t_4 = 18) &= \text{Normal}(172, 9.5)
# \end{align*}
# $$

# %% [markdown]
# ## Implementation
# ### Setup the generative model

# %%
class GenerativeModel():
    def __call__(self, prior_samples: tf.Tensor, tm: tuple[int]) -> dict[str, tf.Tensor]:
        """
        simulate from generative model

        Parameters
        ----------
        prior_samples :
            samples from prior distributions

        tm :
            vector of years

        Returns
        -------
        :
            dictionary with results from simulation run
        """
        B, S, _ = prior_samples.shape
        (h0, h1, s0, s1, gamma) = (prior_samples[:,:,i][..., None] for i in range(5))

        sigma_y = tfd.LogNormal(0., 0.2).sample((B, S, 1))

        mu_list = []
        for t in tm:
            mu = h1 - ( (2*(h1 - h0)) / (tf.exp(s0 * (t - gamma)) + tf.exp(s1 * (t - gamma))) )
            mu_list.append(mu)

        y = tfd.Normal(tf.concat(mu_list, -1), sigma_y).sample()

        # save results
        res = {f"y{t}": y[:,:,i] for i,t in enumerate(tm)}

        for par, par_name in zip([h0, h1, s0, s1, gamma], ["h0", "h1", "s0", "s1", "gamma"]):
            res[par_name] = par

        return res

# define model argument for Elicit
model = el.model(obj=GenerativeModel, tm=(2, 8, 13, 18))

# %% [markdown]
# ### Create expert data
# #### Version 1: Use expert data as provided by Manderson & Goudie (2023)

# %%
# simulate expert data
expert_dat = dict(
    quantiles_y2 = tfd.Normal(88, 3.5).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).numpy(),
    quantiles_y8 = tfd.Normal(130, 5.5).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).numpy(),
    quantiles_y13 = tfd.Normal(160, 8).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).numpy(),
    quantiles_y18 = tfd.Normal(172, 9.5).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).numpy(),
)

# define expert argument for Elicit
expert_data = el.expert.data(dat=expert_dat)

expert_dat

# %% [markdown]
# #### Version 2: Simulate from ground truth (for validation purposes)

# %%
true_hyp = dict(mu0=4.73, sigma0=0.03,
                mu1=5.06, sigma1=0.13,
                mu2=-2.74, sigma2=0.04,
                mu3=-1.73, sigma3=0.09,
                mu4=1.72, sigma4=0.07)

true_prior = dict(
    h0 = tfd.LogNormal(true_hyp["mu0"], true_hyp["sigma0"]),
    h1 = tfd.LogNormal(true_hyp["mu1"], true_hyp["sigma1"]),
    s0 = tfd.LogNormal(true_hyp["mu2"], true_hyp["sigma2"]),
    s1 = tfd.LogNormal(true_hyp["mu3"], true_hyp["sigma3"]),
    gamma = tfd.LogNormal(true_hyp["mu4"], true_hyp["sigma4"])
)

# define expert argument for Elicit
expert_sim = el.expert.simulator(ground_truth=true_prior,
                                 num_samples=10_000)

# %% [markdown]
# ### Setup model parameters

# %%
parameters = [
    el.parameter(
        name="h0",
        family=tfd.LogNormal,
        hyperparams=dict(loc=el.hyper("mu0", lower=4., upper=6.),
                         scale=el.hyper("sigma0", lower=0., upper=1.)),
    ),
    el.parameter(
        name="h1",
        family=tfd.LogNormal,
        hyperparams=dict(loc=el.hyper("mu1", lower=4., upper=6.),
                         scale=el.hyper("sigma1", lower=0., upper=1.)),
    ),
    el.parameter(
        name="s0",
        family=tfd.LogNormal,
        hyperparams=dict(loc=el.hyper("mu2", lower=-3., upper=0.),
                         scale=el.hyper("sigma2", lower=0., upper=1.)),
    ),
    el.parameter(
        name="s1",
        family=tfd.LogNormal,
        hyperparams=dict(loc=el.hyper("mu3", lower=-3., upper=0.),
                         scale=el.hyper("sigma3", lower=0., upper=1.)),
    ),
    el.parameter(
        name="gamma",
        family=tfd.LogNormal,
        hyperparams=dict(loc=el.hyper("mu4", lower=0., upper=2.),
                         scale=el.hyper("sigma4", lower=0., upper=2.)),
    )
]

# %% [markdown]
# ### Setup target quantities and elicitation techniques

# %%
targets = [
    el.target(
        name=tar,
        query=el.queries.quantiles((0.05, 0.25, 0.5, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    ) for tar in ["y2", "y8", "y13", "y18"]
]

# %% [markdown]
# ### Setup of training and initialization

# %%
optimizer=el.optimizer(
        optimizer=tf.keras.optimizers.Adam,
        learning_rate=0.1, clipnorm=1.0
    )

trainer=el.trainer(method="parametric_prior",
                    seed=2024, epochs=700, progress=1)

initializer=el.initializer(
        hyperparams=dict(
            mu0=4., sigma0=0.01,
            mu1=4., sigma1=0.1,
            mu2=1., sigma2=0.01,
            mu3=1., sigma3=0.01,
            mu4=1., sigma4=0.01)
    )

# %% [markdown]
# ### Putting everything together (Setup eliobj)
# First for the validation run using simulated expert data

# %%
# check that ground-truth implies reasonable expert judgements
elicited_stats, *_ = el.utils.dry_run(
    model, parameters, targets, trainer,
    el.initializer(hyperparams=true_hyp),
    network=None)

pd.DataFrame({
        tar: np.round(tf.reduce_mean(elicited_stats[tar], 0), 2).tolist()
        for tar in ["quantiles_y2", "quantiles_y8", "quantiles_y13", "quantiles_y18"]
        }).round(2)


# %%
eliobj = el.Elicit(
    model=model,
    parameters=parameters,
    targets=targets,
    expert=expert_sim,
    optimizer=optimizer,
    trainer=trainer,
    initializer=initializer
)

# %%
# inspect eliobj
eliobj

# %% [markdown]
# ### Run eliobj (Fit prior distributions)

# %%
eliobj.fit()

# %%
{f"{k}": eliobj.history[0]["hyperparameter"][k][-3:] for k in eliobj.history[0]["hyperparameter"]}

# %% [markdown]
# ### Inspect results

# %%
el.plots.loss(eliobj);

# %%
el.plots.hyperparameter(eliobj);

# %%
el.plots.elicits(eliobj, cols=4);

# %%
el.plots.prior_joint(eliobj);

# %%
eliobj_dat = copy.deepcopy(eliobj)

eliobj_dat.update(expert=expert_data)

eliobj_dat.fit()

# %%
el.plots.loss(eliobj_dat);

# %%
el.plots.hyperparameter(eliobj_dat);

# %%
el.plots.elicits(eliobj_dat, cols=4);

# %%
el.plots.prior_joint(eliobj_dat);

# %%
