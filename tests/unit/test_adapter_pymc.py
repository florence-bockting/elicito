import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el
from elicito.adapter import pymc as adapter_pymc

pm = pytest.importorskip("pymc")
tfd = tfp.distributions

SEED = tf.constant([1, 2], dtype=tf.int32)


def _samples(beta0, beta1, sigma, n=5):
    """prior samples of shape (1, n, 3) with fixed values"""
    return tf.constant([[[beta0, beta1, sigma]] * n], dtype=tf.float32)


@pytest.fixture
def toy_model():
    """the linear regression of the tutorial getting-started-param"""
    x = np.array([0.5, 1.0, 1.5])
    with pm.Model() as m:
        mu0, sigma0 = pm.Data("mu0", 0.0), pm.Data("sigma0", 1.0)
        mu1, sigma1 = pm.Data("mu1", 0.0), pm.Data("sigma1", 1.0)
        sigma2 = pm.Data("sigma2", 1.0)
        beta0 = pm.Normal("beta0", mu0, sigma0)
        beta1 = pm.Normal("beta1", mu1, sigma1)
        sigma = pm.HalfNormal("sigma", sigma2)
        epred = pm.Deterministic("epred", beta0 + beta1 * x)
        pm.Normal("ypred", epred, sigma, observed=np.zeros(3))
    return m


def test_parameters_match_the_tutorial(toy_model):
    params = adapter_pymc.parameters(toy_model)
    assert [p["name"] for p in params] == ["beta0", "beta1", "sigma"]
    assert [p["family"] for p in params] == [tfd.Normal, tfd.Normal, tfd.HalfNormal]
    names = [h["name"] for p in params for h in p["hyperparams"].values()]
    assert names == ["mu0", "sigma0", "mu1", "sigma1", "sigma2"]
    assert params[0]["hyperparams"]["scale"]["constraint_name"] == "softplusL"
    assert params[0]["hyperparams"]["loc"]["constraint_name"] == "identity"


def test_unsupported_family_raises():
    with pm.Model() as m:
        pm.Gamma("g", alpha=pm.Data("a", 2.0), beta=pm.Data("b", 1.0))
    with pytest.raises(NotImplementedError, match="GammaRV"):
        adapter_pymc.parameters(m)


def test_constant_hyperparameter_raises():
    with pm.Model() as m:
        pm.Normal("b", mu=0.0, sigma=pm.Data("s", 1.0))
    with pytest.raises(TypeError, match="pm.Data"):
        adapter_pymc.parameters(m)


def test_shared_hyperparameter():
    with pm.Model() as m:
        s = pm.Data("s", 1.0)
        pm.Normal("b0", pm.Data("mu0", 0.0), s)
        pm.Normal("b1", pm.Data("mu1", 0.0), s)
    params = adapter_pymc.parameters(m)
    assert params[0]["hyperparams"]["scale"]["shared"]
    assert params[1]["hyperparams"]["scale"]["shared"]
    assert not params[0]["hyperparams"]["loc"]["shared"]


def test_model_is_accepted_by_el_model(toy_model):
    el.model(obj=adapter_pymc.model(toy_model))


def test_model_computes_epred(toy_model):
    out = adapter_pymc.model(toy_model)()(_samples(1.0, 2.0, 1.0), seed=SEED)
    assert set(out) == {"ypred", "epred"}
    assert out["epred"].shape == (1, 5, 3)
    np.testing.assert_allclose(out["epred"][0, 0], [2.0, 3.0, 4.0])


def test_model_same_seed_same_ypred(toy_model):
    gen = adapter_pymc.model(toy_model)()
    ps = _samples(1.0, 2.0, 1.0)
    tf.debugging.assert_equal(gen(ps, seed=SEED)["ypred"], gen(ps, seed=SEED)["ypred"])


def test_model_ypred_moments(toy_model):
    out = adapter_pymc.model(toy_model)()(_samples(0.0, 0.0, 2.0, n=100_000), seed=SEED)
    assert abs(float(tf.reduce_mean(out["ypred"]))) < 0.05
    assert abs(float(tf.math.reduce_std(out["ypred"])) - 2.0) < 0.05


def test_model_unsupported_op_raises():
    with pm.Model() as m:
        b = pm.Normal("b", pm.Data("mu", 0.0), pm.Data("s", 1.0))
        pm.Deterministic("e", pm.math.exp(b))
    with pytest.raises(NotImplementedError, match="Exp"):
        adapter_pymc.model(m)


def _slice(i):
    """target function that selects design point i of ypred"""

    def target(ypred):
        return ypred[:, :, i]

    return target


def test_set_hyperparameters_writes_the_learned_values(toy_model):
    eliobj = el.Elicit(
        model=el.model(obj=adapter_pymc.model(toy_model)),
        parameters=adapter_pymc.parameters(toy_model),
        targets=[
            el.target(
                name=f"y_X{i}",
                query=el.queries.quantiles((0.25, 0.5, 0.75)),
                loss=el.losses.MMD2(kernel="energy"),
                target_method=_slice(i),
            )
            for i in range(3)
        ],
        expert=el.expert.simulator(
            ground_truth={
                "beta0": tfd.Normal(5.0, 1.0),
                "beta1": tfd.Normal(2.0, 1.0),
                "sigma": tfd.HalfNormal(7.0),
            },
            num_samples=1_000,
        ),
        optimizer=el.optimizer(optimizer=tf.keras.optimizers.Adam, learning_rate=0.1),
        trainer=el.trainer(method="parametric_prior", seed=0, epochs=2, progress=0),
        initializer=el.initializer(
            method="sobol",
            iterations=1,
            distribution=el.initializers.uniform(radius=1.0, mean=0.0),
        ),
    )
    eliobj.fit()
    adapter_pymc.set_hyperparameters(toy_model, eliobj)

    learned = eliobj.hyperparameters()
    for name, value in learned.items():
        np.testing.assert_allclose(toy_model[name].get_value(), value, rtol=1e-6)
    # the PyMC prior now draws with the learned values
    draws = pm.draw(toy_model["beta0"], draws=20_000, random_seed=1)
    assert abs(draws.mean() - learned["mu0"]) < 5 * learned["sigma0"] / np.sqrt(20_000)
