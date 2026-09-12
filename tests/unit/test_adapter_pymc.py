import numpy as np
import pytest
import tensorflow_probability as tfp

from elicito.adapter import pymc as adapter_pymc

pm = pytest.importorskip("pymc")
tfd = tfp.distributions


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
