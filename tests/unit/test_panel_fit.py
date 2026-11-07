"""
Tests of the fit of a panel of experts
"""

import pytest
import tensorflow_probability as tfp

import elicito as el
from elicito.elicit import _is_panel
from elicito.fit import Fit, Panel

tfd = tfp.distributions

PARAMETERS = [
    el.parameter(
        name="beta0",
        family=tfd.Normal,
        hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)),
    ),
]
TARGETS = [el.quantiles("y", probs=(0.25, 0.50, 0.75))]
EXPERTS = {
    "expert 1": {"y": [-0.7, 0.0, 0.7]},
    "expert 2": {"y": [1.3, 2.0, 2.7]},
}


def _model(prior_samples):
    return dict(y=prior_samples[..., 0])


def _eliobj(pool=None):
    return el.Elicit(
        _model,
        TARGETS,
        PARAMETERS,
        pool=pool,
        epochs=40,
        B=4,
        num_samples=100,
        verbose=False,
        dry_run=False,
    )


def test_a_dict_of_dicts_is_a_panel():
    oracle = el.expert.simulator(ground_truth={"beta0": tfd.Normal(0, 1)})

    assert _is_panel(EXPERTS)
    assert not _is_panel(EXPERTS["expert 1"])
    assert not _is_panel(oracle)


def test_a_mixed_dict_raises():
    with pytest.raises(ValueError, match="mix answers and experts"):
        _is_panel({"expert 1": {"y": [0.0, 1.0, 2.0]}, "y": [0.0, 1.0, 2.0]})


def test_a_pool_for_one_expert_warns():
    with pytest.warns(UserWarning, match="answers are of one expert"):
        fit = _eliobj(pool="complete").fit(EXPERTS["expert 1"])
    assert isinstance(fit, Fit)


def test_a_pool_for_a_panel_of_one_warns():
    with pytest.warns(UserWarning, match="panel has one expert"):
        panel = _eliobj(pool="complete").fit({"expert 1": EXPERTS["expert 1"]})
    assert panel.pool is None


def test_no_pool_fits_each_expert_alone():
    panel = _eliobj().fit(EXPERTS)

    assert isinstance(panel, Panel)
    assert panel.experts == ["expert 1", "expert 2"]
    assert panel["expert 1"].hyperparameters()["mu0"] < 1.0
    assert panel["expert 2"].hyperparameters()["mu0"] > 1.0


def test_complete_pooling_fits_one_prior():
    panel = _eliobj(pool="complete").fit(EXPERTS)

    assert set(panel.joint.hyperparameters()) == {"mu0", "sigma0"}
    with pytest.raises(KeyError, match="panel.joint"):
        panel["expert 1"]


def test_partial_pooling_with_a_fixed_tau_learns_offsets():
    panel = _eliobj(pool=el.partial(over=["mu0"], tau=1.0)).fit(EXPERTS)

    hyper = panel.joint.hyperparameters()
    assert set(hyper) == {"mu0", "sigma0", "mu0__e0", "mu0__e1"}
    assert hyper["mu0__e0"] < hyper["mu0__e1"]
    assert panel.tau == {"mu0": 1.0}


def test_partial_pooling_estimates_tau_in_two_stages():
    panel = _eliobj(pool=el.partial(over=["mu0"])).fit(EXPERTS)

    assert set(panel.tau) == {"mu0"}
    assert panel.tau["mu0"] > 0.1


def test_a_panel_takes_an_oracle():
    experts = {
        "expert 1": EXPERTS["expert 1"],
        "oracle": el.expert.simulator(
            ground_truth={"beta0": tfd.Normal(2.0, 1.0)}, num_samples=1_000
        ),
    }
    panel = _eliobj(pool=el.partial(over=["mu0"], tau=1.0)).fit(experts)

    hyper = panel.joint.hyperparameters()
    assert hyper["mu0__e0"] < hyper["mu0__e1"]


@pytest.fixture(scope="module")
def partial_panel():
    return _eliobj(pool=el.partial(over=["mu0"], tau=1.0)).fit(EXPERTS)


@pytest.fixture(scope="module")
def complete_panel():
    return _eliobj(pool="complete").fit(EXPERTS)


@pytest.fixture(scope="module")
def unpooled_panel():
    return _eliobj().fit(EXPERTS)


def test_an_expert_gets_the_panel_value_plus_its_offset(partial_panel):
    panel = partial_panel.hyperparameters()
    offsets = partial_panel.offsets()
    expert = partial_panel.hyperparameters("expert 2")

    assert set(panel) == {"mu0", "sigma0"}
    assert expert["mu0"] == pytest.approx(panel["mu0"] + float(offsets["mu0"][1]))
    assert expert["sigma0"] == panel["sigma0"]


def test_offsets_have_one_value_for_each_expert(partial_panel):
    offsets = partial_panel.offsets()

    assert list(offsets.data_vars) == ["mu0"]
    assert list(offsets["expert"].values) == ["expert 1", "expert 2"]


def test_an_unknown_expert_raises(partial_panel):
    with pytest.raises(KeyError, match="Unknown expert"):
        partial_panel.hyperparameters("expert 9")


def test_complete_pooling_gives_every_expert_the_panel_prior(complete_panel):
    assert complete_panel.hyperparameters("expert 1") == (
        complete_panel.hyperparameters()
    )
    with pytest.raises(ValueError, match="Only pool=el.partial"):
        complete_panel.offsets()


def test_no_pooling_has_no_panel_values(unpooled_panel):
    assert set(unpooled_panel.hyperparameters("expert 1")) == {"mu0", "sigma0"}
    with pytest.raises(ValueError, match="no panel values"):
        unpooled_panel.hyperparameters()
    with pytest.raises(ValueError, match="no population prior"):
        unpooled_panel.population_prior()


def test_the_population_prior_holds_the_disagreement(partial_panel):
    sigma0 = partial_panel.hyperparameters()["sigma0"]
    draws = partial_panel.population_prior(num_samples=20_000)["beta0"]

    assert draws.dims == ("draw",)
    expected = (sigma0**2 + 1.0**2) ** 0.5
    assert float(draws.std()) == pytest.approx(expected, rel=0.05)


def test_the_population_prior_of_complete_pooling_is_the_shared_prior(
    complete_panel,
):
    sigma0 = complete_panel.hyperparameters()["sigma0"]
    draws = complete_panel.population_prior(num_samples=20_000)["beta0"]

    assert float(draws.std()) == pytest.approx(sigma0, rel=0.05)


def test_the_population_prior_repeats_with_its_seed(partial_panel):
    first = partial_panel.population_prior(num_samples=10, seed=3)
    second = partial_panel.population_prior(num_samples=10, seed=3)

    assert first.equals(second)


def test_the_panel_prints_a_row_for_the_panel_and_each_expert(partial_panel):
    text = repr(partial_panel)

    assert "pool='partial'" in text
    assert "tau: mu0=1" in text
    for row in ("panel", "expert 1", "expert 2"):
        assert row in text


@pytest.mark.parametrize("name", ["partial_panel", "complete_panel", "unpooled_panel"])
def test_a_saved_panel_loads_with_the_same_values(request, monkeypatch, tmp_path, name):
    panel = request.getfixturevalue(name)
    monkeypatch.chdir(tmp_path)
    panel.eliobj.save(file="panel")

    loaded = el.load("panel.pkl")

    assert isinstance(loaded, Panel)
    assert loaded.eliobj is not None
    assert loaded.experts == panel.experts
    assert loaded.tau == panel.tau
    for expert in panel.experts:
        assert loaded.hyperparameters(expert) == panel.hyperparameters(expert)
    assert repr(loaded) == repr(panel)


def test_a_loaded_partial_panel_keeps_its_penalty(partial_panel, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    partial_panel.eliobj.save(file="panel")

    loaded = el.load("panel.pkl")

    assert loaded.joint.eliobj.trainer["penalty"] is not None


def test_a_fitted_panel_refits_only_with_overwrite():
    eliobj = _eliobj(pool="complete")
    eliobj.update(epochs=2)
    first = eliobj.fit(EXPERTS)

    with pytest.raises(ValueError, match="already fitted"):
        eliobj.fit(EXPERTS)
    second = eliobj.fit(EXPERTS, overwrite=True)

    assert eliobj.panel is second
    assert first is not second
