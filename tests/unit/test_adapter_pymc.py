import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el
from elicito.adapter import pymc as adapter_pymc
from elicito.adapter._pytensor import translate, vectorize

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
        pm.Laplace("l", mu=pm.Data("mu", 0.0), b=pm.Data("b", 1.0))
    with pytest.raises(NotImplementedError, match="LaplaceRV"):
        adapter_pymc.parameters(m)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda a, b: pm.Gamma("x", alpha=a, beta=b), id="Gamma"),
        pytest.param(lambda a, b: pm.Beta("x", alpha=a, beta=b), id="Beta"),
        pytest.param(lambda a, b: pm.Uniform("x", lower=a, upper=b), id="Uniform"),
    ],
)
def test_family_matches_pymc(build):
    """the tfd family with the pm.Data values has the mean of the PyMC prior"""
    with pm.Model() as m:
        build(pm.Data("a", 2.0), pm.Data("b", 5.0))
    [p] = adapter_pymc.parameters(m)
    kwargs = {k: m[h["name"]].get_value() for k, h in p["hyperparams"].items()}
    pm_mean = pm.draw(m["x"], draws=20_000, random_seed=1).mean()
    np.testing.assert_allclose(p["family"](**kwargs).mean(), pm_mean, rtol=0.02)


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


@pytest.fixture
def hierarchical_model():
    """a prior mu, a latent theta of shape 4, and an observed y"""
    with pm.Model() as m:
        mu = pm.Normal("mu", pm.Data("mu0", 0.0), pm.Data("sigma0", 1.0))
        theta = pm.Normal("theta", mu, 1.0, shape=4)
        pm.Normal("y", theta, 0.5, observed=np.zeros(4))
    return m


def test_latent_rv_is_not_a_parameter(hierarchical_model):
    params = adapter_pymc.parameters(hierarchical_model)
    assert [p["name"] for p in params] == ["mu"]


def test_latent_rv_with_a_pm_data_input_raises():
    with pm.Model() as m:
        mu = pm.Normal("mu", pm.Data("mu0", 0.0), pm.Data("sigma0", 1.0))
        pm.Normal("theta", mu, pm.Data("tau", 1.0))
    with pytest.raises(TypeError, match="pm.Data"):
        adapter_pymc.parameters(m)


def test_vector_prior_becomes_scalar_parameters():
    with pm.Model() as m:
        pm.Normal("b", pm.Data("mu", 0.0), pm.Data("s", 1.0), shape=3)
    params = adapter_pymc.parameters(m)
    assert [p["name"] for p in params] == ["b[0]", "b[1]", "b[2]"]
    assert all(h["shared"] for p in params for h in p["hyperparams"].values())


def test_vector_prior_with_a_fixed_input():
    with pm.Model() as m:
        pm.HalfNormal("s", pm.Data("sigma", 1.0), shape=2)
    params = adapter_pymc.parameters(m)
    assert [p["family"] for p in params] == [tfd.HalfNormal, tfd.HalfNormal]


def test_vector_hyperparameter_raises():
    with pm.Model() as m:
        pm.Normal("b", pm.Data("mu", np.zeros(3)), pm.Data("s", 1.0))
    with pytest.raises(TypeError, match="scalar"):
        adapter_pymc.parameters(m)


def test_model_puts_the_columns_back_into_a_vector_prior():
    with pm.Model() as m:
        a = pm.Normal("a", pm.Data("mu_a", 0.0), pm.Data("s_a", 1.0))
        b = pm.Normal("b", pm.Data("mu_b", 0.0), pm.Data("s_b", 1.0), shape=2)
        pm.Deterministic("e", a + b * np.array([1.0, 10.0]))
    samples = tf.constant([[[1.0, 2.0, 3.0]]])
    out = adapter_pymc.model(m)()(samples, seed=SEED)
    np.testing.assert_allclose(out["e"][0, 0], [3.0, 31.0])


def test_model_samples_a_latent_rv(hierarchical_model):
    samples = tf.fill((1, 100_000, 1), 3.0)
    out = adapter_pymc.model(hierarchical_model)()(samples, seed=SEED)
    assert out["y"].shape == (1, 100_000, 4)
    assert abs(float(tf.reduce_mean(out["y"])) - 3.0) < 0.05
    assert abs(float(tf.math.reduce_std(out["y"])) - np.sqrt(1.25)) < 0.05


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
        pm.Deterministic("e", pm.math.cos(b))
    with pytest.raises(NotImplementedError, match="Cos"):
        adapter_pymc.model(m)


def test_vectorize_batches_an_rv_whose_first_input_is_constant():
    """pytensor's vectorize_graph keeps the old size of Binomial(n, p) here"""
    pt = pytest.importorskip("pytensor.tensor")
    p = pt.vector("p", shape=(3,))
    x = pt.random.binomial(n=10, p=p, size=(3,))
    [out] = vectorize([x], {p: pt.matrix("p_batched", shape=(None, 3))})
    assert out.type.ndim == 2


N = 20_000


@pytest.mark.parametrize(
    "dist",
    [
        pytest.param(lambda: pm.Normal.dist(1.0, 2.0, size=N), id="Normal"),
        pytest.param(lambda: pm.HalfNormal.dist(2.0, size=N), id="HalfNormal"),
        pytest.param(
            lambda: pm.TruncatedNormal.dist(0.0, 1.0, lower=0.0, upper=1.0, size=N),
            id="TruncatedNormal",
        ),
        pytest.param(lambda: pm.Uniform.dist(1.0, 3.0, size=N), id="Uniform"),
        pytest.param(lambda: pm.Beta.dist(2.0, 5.0, size=N), id="Beta"),
        pytest.param(lambda: pm.Gamma.dist(2.0, 3.0, size=N), id="Gamma"),
        pytest.param(lambda: pm.Exponential.dist(2.0, size=N), id="Exponential"),
        pytest.param(lambda: pm.Bernoulli.dist(0.3, size=N), id="Bernoulli"),
        pytest.param(lambda: pm.Binomial.dist(10, 0.3, size=N), id="Binomial"),
        pytest.param(
            lambda: pm.Categorical.dist([0.2, 0.5, 0.3], size=N), id="Categorical"
        ),
    ],
)
def test_rv_matches_pymc(dist):
    """the translated RV has the mean and standard deviation of pm.draw"""
    x = dist()
    [value] = translate([x], [])([], SEED)
    tf_draws = np.asarray(value, dtype=float)
    pm_draws = pm.draw(x, random_seed=1).astype(float)
    se = np.sqrt((tf_draws.var() + pm_draws.var()) / N)
    assert abs(tf_draws.mean() - pm_draws.mean()) < 5 * se
    np.testing.assert_allclose(tf_draws.std(), pm_draws.std(), rtol=0.05)


def test_model_returns_counts_as_float():
    with pm.Model() as m:
        p = pm.Beta("p", pm.Data("a", 2.0), pm.Data("b", 2.0))
        pm.Binomial("k", n=10, p=p, observed=np.zeros(3, dtype=int))
    out = adapter_pymc.model(m)()(tf.fill((1, 50, 1), 0.5), seed=SEED)
    assert out["k"].dtype == tf.float32
    assert set(np.unique(out["k"])) <= set(range(11))


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
