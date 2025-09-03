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
#     (h_0, \delta_h, s_1, \delta_s, \gamma) &\sim \text{LogNormal}(\mu_k, \sigma_k) \quad \text{for } k=1,\ldots, 5\\
#     \sigma_y &\sim \text{LogNormal}(0, 0.2)\\
#     s_0 &= s_1 - \delta_s \\
#     h_1 &= h_0 + \delta_h \\
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
        (h0, delta_h, s1, delta_s, gamma) = (prior_samples[:,:,i][..., None] for i in range(5))

        sigma0 = tfd.LogNormal(0., 0.2).sample((B, S, 1))

        s0 = s1 - delta_s
        h1 = h0 + delta_h

        mu_list = []
        for t in tm:
            mu = h1 - ( (2*(h1 - h0)) / (tf.exp(s0 * (t - gamma)) + tf.exp(s1 * (t - gamma))) )
            mu_list.append(mu)

        y = tfd.Normal(tf.concat(mu_list, -1), sigma0).sample()

        res = dict()
        for i,t in enumerate(tm):
            res[f"y{t}"] = y[:,:,i]
        res["s0"] = s0
        res["h1"] = h1
        res["s1"] = s1
        res["h0"] = h0
        res["gamma"] = gamma

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
    quantiles_y18 = tfd.Normal(172, 9.5).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).numpy()
)

# define expert argument for Elicit
expert=el.expert.data(dat=expert_dat)

expert_dat

# %% [markdown]
# #### Version 2: Simulate from ground truth (for validation purposes)

# %%
ground_truth = dict(
    h0 = tfd.LogNormal(tf.math.log(152.), 0.001),
    delta_h = tfd.LogNormal(tf.math.log(12.), 0.001),
    s1 = tfd.LogNormal(tf.math.log(3.3), 0.001),
    delta_s = tfd.LogNormal(tf.math.log(3.2), 0.001),
    gamma = tfd.LogNormal(tf.math.log(13.4), 0.001)
)

# define expert argument for Elicit
expert = el.expert.simulator(ground_truth=ground_truth, num_samples=10_000)

# %% [markdown]
# ### Setup model parameters

# %%
parameters = [
    el.parameter(
        name="h0",
        family=tfd.LogNormal,
        hyperparams=dict(loc=el.hyper("mu1", lower=4.8, upper=5.2),
                         scale=el.hyper("sigma1", lower=0., upper=0.1)),
    ),
    el.parameter(
        name="delta_h",
        family=tfd.LogNormal,
        hyperparams=dict(loc=el.hyper("mu2", lower=-3, upper=4.),
                         scale=el.hyper("sigma2", lower=0., upper=0.1)),
    ),
    el.parameter(
        name="s1",
        family=tfd.LogNormal,
        hyperparams=dict(loc=el.hyper("mu3", lower=-3, upper=3.),
                         scale=el.hyper("sigma3", lower=0., upper=1.55)),
    ),
    el.parameter(
        name="delta_s",
        family=tfd.LogNormal,
        hyperparams=dict(loc=el.hyper("mu4", lower=-3, upper=3.),
                         scale=el.hyper("sigma4", lower=0., upper=1.14)),
    ),
    el.parameter(
        name="gamma",
        family=tfd.LogNormal,
        hyperparams=dict(loc=el.hyper("mu5", lower=-3., upper=5.),
                         scale=el.hyper("sigma5", lower=0., upper=0.1)),
    )
]

# %% [markdown]
# ### Setup target quantities and elicitation techniques

# %%
targets = [
    el.target(
        name=tar,
        query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0,
    ) for tar in ["y2", "y8", "y13", "y18"]
]

# %% [markdown]
# ### Putting everything together (Setup eliobj)

# %%
eliobj = el.Elicit(
    model=model,
    parameters=parameters,
    targets=targets,
    expert=expert,
    optimizer=el.optimizer(
        optimizer=tf.keras.optimizers.Adam, learning_rate=0.05, clipnorm=1.0
    ),
    trainer=el.trainer(method="parametric_prior", seed=2025, epochs=400, progress=1),
    initializer=el.initializer(
        hyperparams = {
            "mu1": tf.math.log(150.), "sigma1": el.utils.DoubleBound(0., 0.1).forward(0.01),
            "mu2": tf.math.log(2.), "sigma2": el.utils.DoubleBound(0., 0.1).forward(0.01),
            "mu3": tf.math.log(2.), "sigma3": el.utils.DoubleBound(0., 2.).forward(1.55),
            "mu4": tf.math.log(2.), "sigma4": el.utils.DoubleBound(0., 0.2).forward(0.14),
            "mu5": tf.math.log(2.), "sigma5": el.utils.DoubleBound(0., 0.1).forward(0.01)
        }
    ),
)

# %%
# inspect eliobj
eliobj

# %% [markdown]
# ### Run eliobj (Fit prior distributions)

# %%
eliobj.fit()

# %% [markdown]
# ### Inspect results

# %%
el.plots.loss(eliobj);

# %%
el.plots.hyperparameter(eliobj);

# %%
el.plots.elicits(eliobj);

# %%
df_list = []

for i in range(len(eliobj.history)):
    df = pd.DataFrame({f"{key}": [np.round(
        tf.reduce_mean(eliobj.history[i]["hyperparameter"][key][:-30]),2)]
     for key in eliobj.history[0]["hyperparameter"].keys()})
    df["sim"]=i
    df_list.append(df)

df2 = pd.concat(df_list)
df2

# %%
# only if "true prior distributions" for expert data is used

tf.reduce_mean(eliobj.results[0]["prior_samples"], (0,1))
