"""
Unittest for init.py module
"""

import re
from copy import deepcopy

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr
from numpy import testing as npt
from tests.utils import eliobj as base_eliobj

import elicito as el
from elicito import Elicit
from elicito.utils import get_expert_datformat

tfd = tfp.distributions
scipy = pytest.importorskip("scipy")


class TestModel:
    def __call__(self, prior_samples):
        b0 = tfd.Normal(0, 1).sample((3, 3, 1))
        return dict(prior_samples=prior_samples, b0=b0)


@pytest.fixture
def eliobj():
    return Elicit(
        model=el.model(TestModel),
        parameters=[
            el.parameter(
                name=f"b{i}",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper(name=f"mu{i}"), scale=el.hyper(f"sigma{i}")
                ),
            )
            for i in range(2)
        ],
        targets=[
            el.target(name="b0", loss=el.losses.L2, query=el.queries.quantiles((0.5,)))
        ],
        expert=el.expert.data({"quantiles_b0": [0.5]}),
        optimizer=el.optimizer(optimizer=tf.keras.optimizers.Adam, learning_rate=0.1),
        trainer=el.trainer(method="parametric_prior", seed=42, epochs=1),
        initializer=el.initializer(
            "random",
            distribution=el.initialization.uniform(radius=1, mean=0),
            iterations=1,
        ),
        network=None,
    )


@pytest.fixture
def network():
    return el.networks.NF(
        inference_network=el.networks.InvertibleNetwork,
        network_specs=dict(
            num_params=2,
            num_coupling_layers=3,
            coupling_design="affine",
            coupling_settings={
                "dropout": False,
                "dense_args": {
                    "units": 128,
                    "activation": "relu",
                    "kernel_regularizer": None,
                },
                "num_dense": 2,
            },
            permutation="fixed",
        ),
        base_distribution=el.networks.base_normal,
    )


def test_eliobj_attributes(eliobj):
    # check unfitted obj
    assert "temp_results" in dir(eliobj)
    assert "temp_history" in dir(eliobj)

    # check fitted obj
    assert "results" in dir(base_eliobj)
    assert "temp_results" not in dir(base_eliobj)
    assert "temp_history" not in dir(base_eliobj)


def test_expert_input(eliobj):
    # test correct format
    dat_format = eliobj.expert
    expected_format = get_expert_datformat(eliobj.targets)

    assert dat_format["data"].keys() == expected_format.keys()

    # error; wrong format
    msg = (
        "Provided expert data is not in the "
        "correct format. Please use "
        "el.utils.get_expert_datformat to check expected format."
    )

    with pytest.raises(AssertionError, match=msg):
        eliobj.update(expert=el.expert.data({"b0": 0.5}))

    # error; wrong number of parameters
    msg2 = (
        "Dimensionality of ground truth in "  # type: ignore
        "'expert' is not the same  as number of model "
        "parameters. Got num_params=1, expected "
        "2."
    )
    with pytest.raises(AssertionError, match=msg2):
        eliobj.update(expert=el.expert.simulator(ground_truth={"b0": tfd.Normal(0, 1)}))


def test_initializer_network(eliobj, network):
    # parametric method
    # correct: initialized is configured
    assert eliobj.initializer is not None
    assert eliobj.network is None

    # error message if initializer is None
    msg1 = (
        "If method is 'parametric_prior', " " the section 'initializer' can't be None."
    )
    with pytest.raises(ValueError, match=msg1):
        eliobj.update(initializer=None, network=None)

    msg2 = (
        "If method is 'parametric prior' "
        "the 'network' is not used and should be set to None."
    )
    with pytest.raises(ValueError, match=msg2):
        eliobj.update(network=network)

    # correct
    eliobj.update(
        trainer=el.trainer(method="deep_prior", epochs=1, seed=123),
        network=network,
        initializer=None,
    )

    assert eliobj.network == network

    msg3 = "If method is 'deep prior', " " the section 'network' can't be None."
    with pytest.raises(ValueError, match=msg3):
        eliobj.update(network=None)

    msg4 = (
        "For method 'deep_prior' the "
        "'initializer' is not used and should be set to None."
    )
    with pytest.raises(ValueError, match=msg4):
        eliobj.update(
            initializer=el.initializer(
                "random",
                distribution=el.initialization.uniform(radius=1, mean=0),
                iterations=1,
            )
        )


def test_network_parameter(eliobj, network):
    # error due to wrong number of parameters
    msg = (
        "The number of model parameters as "
        "specified in the parameters section, must match the "
        "number of parameters specified in the network."
        "Expected 1 but got 2"
    )
    with pytest.raises(ValueError, match=msg):
        eliobj.update(
            trainer=el.trainer(method="deep_prior", epochs=1, seed=123),
            initializer=None,
            network=network,
            parameters=[
                el.parameter(
                    name="b0",
                    family=tfd.Normal,
                    hyperparams=dict(
                        loc=el.hyper(name="mu0"), scale=el.hyper("sigma0")
                    ),
                )
            ],
        )


def test_network_base(eliobj):
    # error; distribution is not basenormal
    msg = (
        "Currently only the standard normal distribution "
        "is implemented as base distribution. "
        "See GitHub issue #35."
    )
    with pytest.raises(NotImplementedError, match=msg):
        eliobj.update(
            trainer=el.trainer(method="deep_prior", epochs=1, seed=123),
            initializer=None,
            network=el.networks.NF(
                inference_network=el.networks.InvertibleNetwork,
                network_specs=dict(
                    num_params=2,
                    num_coupling_layers=3,
                    coupling_design="affine",
                    coupling_settings={
                        "dropout": False,
                        "dense_args": {
                            "units": 128,
                            "activation": "relu",
                            "kernel_regularizer": None,
                        },
                        "num_dense": 2,
                    },
                    permutation="fixed",
                ),
                base_distribution=tfd.Normal,
            ),
        )


def test_shared_hyperparameter(eliobj):
    # raise error if same name for hyperparameter
    # but shared is False (default)
    msg = (
        "The following hyperparameter have the same "
        "name but are not shared: ['mu0', 'sigma0']. \n"
        "Have you forgot to set shared=True?"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        eliobj.update(
            parameters=[
                el.parameter(
                    name=f"b{i}",
                    family=tfd.Normal,
                    hyperparams=dict(
                        loc=el.hyper(name="mu0"), scale=el.hyper("sigma0")
                    ),
                )
                for i in range(2)
            ]
        )

    # correct; specify shared=True
    eliobj.update(
        parameters=[
            el.parameter(
                name=f"b{i}",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper(name="mu0", shared=True),
                    scale=el.hyper("sigma0", shared=True),
                ),
            )
            for i in range(2)
        ]
    )

    for i in range(2):
        assert eliobj.parameters[i].hyperparams["loc"]["name"] == "mu0"
        assert eliobj.parameters[i].hyperparams["scale"]["name"] == "sigma0"


def test_initializer_setup(eliobj):
    # correct; hyperparameter names match
    eliobj.update(
        initializer=el.initializer(
            hyperparams=dict(mu0=0.1, sigma0=0.2, mu1=0.1, sigma1=0.2)
        )
    )

    assert eliobj.initializer["hyperparams"]["mu0"] == 0.1
    assert eliobj.initializer["hyperparams"]["sigma1"] == 0.2

    # error; hyperparameter names do not match
    msg = (
        "Hyperparameter name 'mu2' doesn't "
        "match any name specified in the parameters "
        "section. Have you misspelled the name?"
    )
    with pytest.raises(ValueError, match=msg):
        eliobj.update(
            initializer=el.initializer(
                hyperparams=dict(mu2=0.1, sigma2=0.2, mu1=0.1, sigma1=0.2)
            )
        )


def test_hyperparameters(eliobj):
    # error; as parametric prior needs family and hyperparams arg.
    msg = (
        "When using method='parametric_prior', the argument "
        "'hyperparams' of el.parameter "
        "cannot be None."
    )
    with pytest.raises(ValueError, match=msg):
        eliobj.update(parameters=[el.parameter(name=f"b{i}") for i in range(2)])


def test_checks_fit():
    # refitting without overwrite is refused
    with pytest.raises(ValueError, match="already fitted"):
        base_eliobj.fit(overwrite=False)

    # overwrite=True refits and replaces the results
    base_eliobj.fit(overwrite=True)

    epochs = base_eliobj.results["history_stats"]["epoch"].shape[0]
    assert base_eliobj.results["history_stats"]["loss"]["total_loss"].values.shape == (
        1,
        epochs,
    )
    assert isinstance(base_eliobj.results, xr.DataTree)


@pytest.mark.parametrize("dry_run", [True, False])
def test_str_method(dry_run):
    summary_data = {
        "quantiles_y_obs": [0, -0.012148, 0.049513, 0.100585, 0.153291, 0.634314]
    }

    class GenerativeModel:
        def __call__(self, prior_samples, N, se_y):
            sigma_j = tf.cast(se_y, tf.float32)
            mu = prior_samples[:, :, 0]
            tau = prior_samples[:, :, 1]

            theta_j = tfd.Normal(loc=mu, scale=tau).sample(N)
            y_pred = tfd.Normal(
                loc=tf.transpose(theta_j, perm=[1, 2, 0]), scale=sigma_j[None, None, :]
            ).sample()

            return dict(y_obs=y_pred)

    eliobj = el.Elicit(
        model=el.model(obj=GenerativeModel, N=50, se_y=tfd.Normal(0, 1).sample(50)),
        parameters=[
            el.parameter(
                name="mu",
                family=tfd.Normal,
                hyperparams=dict(loc=el.hyper("mu_mu"), scale=el.hyper("sigma_mu")),
            ),
            el.parameter(
                name="tau",
                family=tfd.Normal,
                hyperparams=dict(loc=el.hyper("l_tau"), scale=el.hyper("d_tau")),
            ),
        ],
        targets=[
            el.target(
                name="y_obs",
                query=el.queries.quantiles([0.05, 0.25, 0.5, 0.75, 0.95]),
                loss=el.losses.MMD2(kernel="energy"),
            ),
        ],
        expert=el.expert.data(summary_data),
        optimizer=el.optimizer(
            optimizer=tf.keras.optimizers.Adam, learning_rate=0.0, clipnorm=1.0
        ),
        trainer=el.trainer(method="parametric_prior", seed=123, epochs=1, progress=1),
        initializer=el.initializer(
            hyperparams=dict(mu_mu=1.0, sigma_mu=0.5, l_tau=0.1, d_tau=0.4)
        ),
        meta_settings=el.meta_settings(dry_run=dry_run),
    )

    obs_eliobj = eliobj.__str__()

    if dry_run:
        assert "y_obs (128, 200, 50) -> quantiles_y_obs (128, 5)" in obs_eliobj
    else:
        assert "y_obs -> quantiles_y_obs" in obs_eliobj

    eliobj.fit()

    obs_eliobj2 = eliobj.__str__()

    # a fitted eliobj no longer stores the simulated quantities, so the
    # summary shows the names without the shapes
    assert "y_obs -> quantiles_y_obs" in obs_eliobj2


def test_eliobj_fit():
    summary_data = {
        "quantiles_y_obs": [0, -0.012148, 0.049513, 0.100585, 0.153291, 0.634314]
    }

    class GenerativeModel:
        def __call__(self, prior_samples, N, se_y):
            sigma_j = tf.cast(se_y, tf.float32)
            mu = prior_samples[:, :, 0]
            tau = prior_samples[:, :, 1]

            theta_j = tfd.Normal(loc=mu, scale=tau).sample(N)
            y_pred = tfd.Normal(
                loc=tf.transpose(theta_j, perm=[1, 2, 0]), scale=sigma_j[None, None, :]
            ).sample()

            return dict(y_obs=y_pred)

    eliobj = el.Elicit(
        model=el.model(obj=GenerativeModel, N=50, se_y=tfd.Normal(0, 1).sample(50)),
        parameters=[
            el.parameter(
                name="mu",
                family=tfd.Normal,
                hyperparams=dict(loc=el.hyper("mu_mu"), scale=el.hyper("sigma_mu")),
            ),
            el.parameter(
                name="tau",
                family=tfd.Normal,
                hyperparams=dict(loc=el.hyper("l_tau"), scale=el.hyper("d_tau")),
            ),
        ],
        targets=[
            el.target(
                name="y_obs",
                query=el.queries.quantiles([0.05, 0.25, 0.5, 0.75, 0.95]),
                loss=el.losses.MMD2(kernel="energy"),
            ),
        ],
        expert=el.expert.data(summary_data),
        optimizer=el.optimizer(
            optimizer=tf.keras.optimizers.Adam, learning_rate=0.0, clipnorm=1.0
        ),
        trainer=el.trainer(method="parametric_prior", seed=123, epochs=1, progress=1),
        initializer=el.initializer(
            hyperparams=dict(mu_mu=1.0, sigma_mu=0.5, l_tau=0.1, d_tau=0.4)
        ),
    )

    # check parallel fit with specific seeds set by the user
    eliobj1 = deepcopy(eliobj)
    expected_seeds = [2021, 2022, 2023, 2024]
    eliobj1.fit(parallel=el.utils.parallel(runs=4, seeds=expected_seeds))

    assert hasattr(eliobj1, "results")

    npt.assert_array_equal(
        eliobj1.results.history_stats.seed_replication.values, expected_seeds
    )

    with npt.assert_raises(AssertionError):
        assert hasattr(eliobj1, "temp_results")
        assert hasattr(eliobj1, "temp_history")

    # check parallel fit with seeds randomly generated
    eliobj2 = deepcopy(eliobj)
    eliobj2.fit(parallel=el.utils.parallel(runs=4), overwrite=True)

    assert hasattr(eliobj2, "results")

    with npt.assert_raises(AssertionError):
        assert hasattr(eliobj2, "temp_results")
        assert hasattr(eliobj2, "temp_history")
        npt.assert_array_equal(
            eliobj2.results.history_stats.seed_replication.values, expected_seeds
        )

    # check that placeholder for storing results are accurately
    # reset when updating eliobj
    eliobj2.update(
        trainer=el.trainer(method="parametric_prior", seed=125, epochs=1, progress=0)
    )

    assert hasattr(eliobj2, "temp_results")
    assert hasattr(eliobj2, "temp_history")

    with npt.assert_raises(AssertionError):
        assert hasattr(eliobj2, "results")


def test_warmup_epochs_default():
    """a candidate is scored at epoch 0 unless warmup_epochs is set"""
    init = el.initializer(
        method="sobol",
        iterations=2,
        distribution=el.initialization.uniform(radius=1, mean=0),
    )
    assert init["warmup_epochs"] == 0

    init = el.initializer(
        method="sobol",
        iterations=2,
        warmup_epochs=3,
        distribution=el.initialization.uniform(radius=1, mean=0),
    )
    assert init["warmup_epochs"] == 3


def test_warmup_epochs_changes_candidate_loss():
    """warmup_epochs>0 scores a candidate after training, not at epoch 0"""
    summary_data = {"quantiles_y_obs": [-1.5, -0.4, 0.1, 0.7, 1.9]}

    class GenerativeModel:
        def __call__(self, prior_samples, N):
            mu = prior_samples[:, :, 0]
            sigma = tf.exp(prior_samples[:, :, 1])
            y_pred = tfd.Normal(loc=mu, scale=sigma).sample(N)
            return dict(y_obs=tf.transpose(y_pred, perm=[1, 2, 0]))

    def build(warmup_epochs):
        return el.Elicit(
            model=el.model(obj=GenerativeModel, N=10),
            parameters=[
                el.parameter(
                    name="mu",
                    family=tfd.Normal,
                    hyperparams=dict(
                        loc=el.hyper("mu_loc"), scale=el.hyper("mu_scale", lower=0)
                    ),
                ),
                el.parameter(
                    name="log_sigma",
                    family=tfd.Normal,
                    hyperparams=dict(
                        loc=el.hyper("s_loc"), scale=el.hyper("s_scale", lower=0)
                    ),
                ),
            ],
            targets=[
                el.target(
                    name="y_obs",
                    query=el.queries.quantiles([0.05, 0.25, 0.5, 0.75, 0.95]),
                    loss=el.losses.MMD2(kernel="energy"),
                ),
            ],
            expert=el.expert.data(summary_data),
            optimizer=el.optimizer(
                optimizer=tf.keras.optimizers.Adam, learning_rate=0.1, clipnorm=1.0
            ),
            trainer=el.trainer(method="parametric_prior", seed=7, epochs=1, progress=0),
            initializer=el.initializer(
                method="sobol",
                iterations=2,
                warmup_epochs=warmup_epochs,
                distribution=el.initialization.uniform(radius=1, mean=0),
            ),
        )

    cold = build(0)
    cold.fit()
    warm = build(2)
    warm.fit()

    cold_loss = cold.results["initialization"]["loss"].values
    warm_loss = warm.results["initialization"]["loss"].values

    # the warm-up scores the same two candidates, but after two epochs
    assert cold_loss.shape == warm_loss.shape
    assert bool(np.isfinite(warm_loss).all())
    assert (cold_loss != warm_loss).any()

    # the warm-up must not change the epochs of the caller's trainer
    assert warm.trainer["epochs"] == 1


def _one_param():
    return [
        el.parameter(
            name="b0",
            family=tfd.Normal,
            hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)),
        )
    ]


def _one_shape_param():
    return [
        el.parameter(
            name="k",
            family=tfd.Weibull,
            hyperparams=dict(
                concentration=el.hyper("k0", lower=0),
                scale=el.hyper("lambda0", lower=0),
            ),
        )
    ]


def test_from_elicits_box():
    # quantiles 0, 2, 4, 6, 8 -> median 4, IQR 4, spread 4 / 1.35, q95 7.6
    expert = {"quantiles_y": tf.constant([[0.0, 2.0, 4.0, 6.0, 8.0]])}
    box = el.initialization._from_elicits_box(expert, _one_param(), factor=2.0)
    spread = 4.0 / 1.35

    assert box["hyper"] == ["mu0", "sigma0"]
    # a location follows the pooled median
    npt.assert_allclose(box["mean"][0], 4.0, rtol=1e-6)
    npt.assert_allclose(box["radius"][0], 2.0 * spread, rtol=1e-6)
    # a magnitude spans from spread / 100 up to the pooled 95% quantile
    forward = el.utils.LowerBound(0.0).forward
    low = float(forward(spread / 100.0))
    high = float(forward(7.6))
    npt.assert_allclose(box["mean"][1], (low + high) / 2.0, rtol=1e-5)
    npt.assert_allclose(box["radius"][1], (high - low) / 2.0, rtol=1e-5)


def test_from_elicits_box_ignores_the_data_scale_for_a_shape():
    # a shape does not live on the scale of the data, so the box covers the
    # natural range 1 to 5, whatever the elicited values are
    expert = {"quantiles_y": tf.constant([[100.0, 200.0, 400.0, 600.0, 800.0]])}
    box = el.initialization._from_elicits_box(expert, _one_shape_param())

    assert box["hyper"] == ["k0", "lambda0"]
    forward = el.utils.LowerBound(0.0).forward
    low = float(forward(el.initialization.SHAPE_LOW))
    high = float(forward(el.initialization.SHAPE_HIGH))
    npt.assert_allclose(box["mean"][0], (low + high) / 2.0, rtol=1e-5)
    npt.assert_allclose(box["radius"][0], (high - low) / 2.0, rtol=1e-5)
    # the scale of the same family stays a magnitude, and follows the data
    assert box["mean"][1] > box["mean"][0]


def test_from_elicits_uses_spread_floor():
    expert = {"quantiles_y": tf.constant([[3.0, 3.0, 3.0]])}
    box = el.initialization._from_elicits_box(expert, _one_param())

    npt.assert_allclose(box["radius"][0], 2.0e-3, rtol=1e-6)
    npt.assert_allclose(box["mean"][0], 3.0, rtol=1e-6)


def test_from_elicits_marks_the_box_as_deferred():
    box = el.initialization.from_elicits(factor=3.0)

    assert box["from_elicits"] is True
    assert box["factor"] == 3.0
    assert box["hyper"] is None


def test_initializer_rejects_an_unknown_method_name():
    with pytest.raises(ValueError, match="warmstart"):
        el.initializer(
            method="nelder",
            iterations=8,
            distribution=el.initialization.uniform(radius=1, mean=0),
        )


def test_start_vector_reads_the_box():
    names = ["mu0", "sigma0"]

    scalar_box = el.initialization.uniform(radius=1.0, mean=2.0)
    assert el.warmstart._start_vector(scalar_box, names) == [2.0, 2.0]

    listed_box = el.initialization.uniform(
        radius=[1.0, 1.0], mean=[3.0, 4.0], hyper=["sigma0", "mu0"]
    )
    assert el.warmstart._start_vector(listed_box, names) == [4.0, 3.0]


def test_uniform_samples_fills_the_box_in_every_dimension():
    """one design over all hyperparameters, not one sequence per hyperparameter"""
    parameters = [
        el.parameter(
            name=f"b{i}",
            family=tfd.Normal,
            hyperparams=dict(
                loc=el.hyper(f"mu{i}"), scale=el.hyper(f"sigma{i}", lower=0)
            ),
        )
        for i in range(3)
    ]
    names = el.initialization.hyper_names(parameters)

    samples = el.initialization.uniform_samples(
        seed=0,
        hyppar=names,
        n_samples=256,
        method="sobol",
        mean=[0.0] * len(names),
        radius=[1.0] * len(names),
        parameters=parameters,
    )

    drawn = np.stack([samples[name].numpy() for name in names], axis=-1)
    assert drawn.shape == (256, len(names))
    # a separate one-dimensional sequence per hyperparameter left every column
    # correlated above 0.97, so the candidates sat on a diagonal of the box
    correlation = np.corrcoef(drawn.T)
    off_diagonal = correlation[~np.eye(len(names), dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.1


def test_all_finite_and_nonfinite_fraction():
    """the share of overflowing values is reported, not only that one exists"""
    good = {"y": tf.constant([[1.0, 2.0], [3.0, 4.0]])}
    bad = {"y": tf.constant([[1.0, np.inf], [3.0, 4.0]])}

    assert el.utils.all_finite(good) is True
    assert el.utils.all_finite(bad) is False
    npt.assert_allclose(el.utils.nonfinite_fraction(good), 0.0)
    npt.assert_allclose(el.utils.nonfinite_fraction(bad), 0.25)


def _overflow_eliobj(overflow):
    """an eliobj whose draws overflow, while its quantiles stay finite"""

    class GenerativeModel:
        def __call__(self, prior_samples, N):
            mu = prior_samples[:, :, 0]
            sigma = tf.exp(prior_samples[:, :, 1])
            y_pred = tfd.Normal(loc=mu, scale=sigma).sample(N)
            y_obs = tf.transpose(y_pred, perm=[1, 2, 0])
            if overflow:
                # one draw of many; every elicited quantile stays finite
                flat = tf.reshape(y_obs, [-1])
                flat = tf.tensor_scatter_nd_update(flat, [[0]], [np.inf])
                y_obs = tf.reshape(flat, tf.shape(y_obs))
            return dict(y_obs=y_obs)

    return el.Elicit(
        model=el.model(obj=GenerativeModel, N=30),
        parameters=[
            el.parameter(
                name="mu",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper("mu_loc"), scale=el.hyper("mu_scale", lower=0)
                ),
            ),
            el.parameter(
                name="log_sigma",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper("s_loc"), scale=el.hyper("s_scale", lower=0)
                ),
            ),
        ],
        targets=[
            el.target(
                name="y_obs",
                query=el.queries.quantiles((0.25, 0.5, 0.75)),
                loss=el.losses.L2,
                weight=1.0,
            )
        ],
        expert=el.expert.data({"quantiles_y_obs": [-0.7, 0.1, 0.8]}),
        optimizer=el.optimizer(
            optimizer=tf.keras.optimizers.Adam, learning_rate=0.1, clipnorm=1.0
        ),
        trainer=el.trainer(
            method="parametric_prior", seed=0, epochs=1, progress=0, num_samples=50
        ),
        initializer=el.initializer(
            method="sobol",
            iterations=4,
            distribution=el.initialization.uniform(radius=1, mean=0),
        ),
    )


def test_init_runs_rejects_a_candidate_whose_draws_overflow(monkeypatch):
    """a quantile query hides an overflow, so the draws decide, not the loss"""
    # without the overflow the same model initializes and trains
    _overflow_eliobj(overflow=False).fit()

    # the loss alone accepts the candidate: with the check disabled, the model
    # with the overflow initializes as well
    monkeypatch.setattr(el.utils, "all_finite", lambda quantities: True)
    _overflow_eliobj(overflow=True).fit()
    monkeypatch.undo()

    with pytest.raises(ValueError, match="non-finite loss"):
        _overflow_eliobj(overflow=True).fit()


def test_warm_start_spends_the_budget(monkeypatch):
    """Nelder-Mead converges early, so the search restarts until the budget ends"""
    calls = {"n": 0}

    def failing_scorer(**kwargs):
        def scorer(hyperparams):
            calls["n"] += 1
            return el.warmstart.PENALTY * 1.5

        return scorer

    monkeypatch.setattr(el.warmstart, "compile_score", failing_scorer)

    el.warmstart.warm_start(
        expert_elicited_statistics={},
        parameters=base_eliobj.parameters,
        trainer=base_eliobj.trainer,
        model=base_eliobj.model,
        targets=base_eliobj.targets,
        expert=base_eliobj.expert,
        distribution=el.initialization.uniform(radius=1, mean=2.0),
        max_evals=120,
        seed=0,
    )

    assert calls["n"] >= 120


def test_warm_start_falls_back_when_every_point_fails(monkeypatch, caplog):
    """a point that failed is never returned as a start value"""
    monkeypatch.setattr(
        el.warmstart,
        "compile_score",
        lambda **kwargs: lambda hyperparams: el.warmstart.PENALTY * 1.5,
    )

    with caplog.at_level("WARNING"):
        hyperparams = el.warmstart.warm_start(
            expert_elicited_statistics={},
            parameters=base_eliobj.parameters,
            trainer=base_eliobj.trainer,
            model=base_eliobj.model,
            targets=base_eliobj.targets,
            expert=base_eliobj.expert,
            distribution=el.initialization.uniform(radius=1, mean=2.0),
            max_evals=30,
            seed=0,
        )

    assert set(hyperparams.values()) == {2.0}
    assert "every evaluated point failed" in caplog.text


def test_warm_start_returns_one_value_per_hyperparameter():
    expert_elicits, _ = el.utils.get_expert_data(
        base_eliobj.trainer,
        base_eliobj.model,
        base_eliobj.targets,
        base_eliobj.expert,
        base_eliobj.parameters,
        base_eliobj.network,
        base_eliobj.trainer["seed"],
    )

    hyperparams = el.warmstart.warm_start(
        expert_elicited_statistics=expert_elicits,
        parameters=base_eliobj.parameters,
        trainer=base_eliobj.trainer,
        model=base_eliobj.model,
        targets=base_eliobj.targets,
        expert=base_eliobj.expert,
        distribution=el.initialization.from_elicits(),
        max_evals=20,
        seed=0,
    )

    assert list(hyperparams) == el.initialization.hyper_names(base_eliobj.parameters)
    assert all(np.isfinite(v) for v in hyperparams.values())


def test_box_vector_reads_every_entry_of_the_box():
    names = ["mu0", "sigma0"]

    scalar_box = el.initialization.uniform(radius=1.5, mean=2.0)
    assert el.warmstart._box_vector(scalar_box, names, "radius") == [1.5, 1.5]

    listed_box = el.initialization.uniform(
        radius=[1.0, 2.0], mean=[3.0, 4.0], hyper=["sigma0", "mu0"]
    )
    assert el.warmstart._box_vector(listed_box, names, "radius") == [2.0, 1.0]


def test_cmaes_is_a_registered_initialization_method():
    method = el.initialization.get_init_method("cmaes")

    assert isinstance(method, el.initialization.CmaEs)
    assert method.default_iterations == 500


def test_cma_search_returns_one_value_per_hyperparameter():
    pytest.importorskip("cma")

    expert_elicits, _ = el.utils.get_expert_data(
        base_eliobj.trainer,
        base_eliobj.model,
        base_eliobj.targets,
        base_eliobj.expert,
        base_eliobj.parameters,
        base_eliobj.network,
        base_eliobj.trainer["seed"],
    )

    hyperparams = el.cmaes.cma_search(
        expert_elicited_statistics=expert_elicits,
        parameters=base_eliobj.parameters,
        trainer=base_eliobj.trainer,
        model=base_eliobj.model,
        targets=base_eliobj.targets,
        expert=base_eliobj.expert,
        distribution=el.initialization.uniform(radius=1.0, mean=0.0),
        max_evals=12,
        seed=0,
    )

    assert list(hyperparams) == el.initialization.hyper_names(base_eliobj.parameters)
    assert all(np.isfinite(v) for v in hyperparams.values())


def test_cma_search_falls_back_when_every_point_fails(monkeypatch, caplog):
    """a point that failed is never returned as a start value"""
    pytest.importorskip("cma")

    monkeypatch.setattr(
        el.warmstart,
        "compile_score",
        lambda **kwargs: lambda hyperparams: el.warmstart.PENALTY * 1.5,
    )

    with caplog.at_level("WARNING"):
        hyperparams = el.cmaes.cma_search(
            expert_elicited_statistics={},
            parameters=base_eliobj.parameters,
            trainer=base_eliobj.trainer,
            model=base_eliobj.model,
            targets=base_eliobj.targets,
            expert=base_eliobj.expert,
            distribution=el.initialization.uniform(radius=1.0, mean=2.0),
            max_evals=12,
            seed=0,
        )

    assert set(hyperparams.values()) == {2.0}
    assert "every evaluated point failed" in caplog.text


def _cmaes_eliobj(
    epochs,
    sigma0=0.5,
    popsize=4,
    method="parametric_prior",
    init_method="sobol",
):
    """an eliobj that is fitted with the CMA-ES search instead of a gradient"""
    return Elicit(
        model=base_eliobj.model,
        parameters=base_eliobj.parameters,
        targets=base_eliobj.targets,
        expert=base_eliobj.expert,
        optimizer=el.optimizer(
            optimizer=el.cmaes.CMAES, sigma0=sigma0, popsize=popsize
        ),
        trainer=el.trainer(method=method, seed=0, epochs=epochs, progress=0),
        initializer=el.initializer(
            method=init_method,
            iterations=2,
            distribution=el.initialization.uniform(radius=1.0, mean=0.0),
        ),
    )


def test_cma_training_records_one_point_per_generation():
    pytest.importorskip("cma")

    eliobj = _cmaes_eliobj(epochs=12)
    eliobj.fit()

    loss = eliobj.results["history_stats/loss"]["total_loss"]
    hyper = eliobj.results["history_stats/hyperparameter"]
    # 12 evaluations, 4 candidates per generation
    assert loss.sizes["epoch"] == 3
    for name in el.initialization.hyper_names(base_eliobj.parameters):
        assert hyper[name].sizes["epoch"] == 3
    assert np.all(np.isfinite(loss.values))


def test_cma_training_spends_the_budget(monkeypatch):
    pytest.importorskip("cma")

    calls = []
    original = el.warmstart.evaluate

    def counted(**kwargs):
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(el.warmstart, "evaluate", counted)

    eliobj = _cmaes_eliobj(epochs=12)
    eliobj.fit()

    # the budget, plus the final evaluation of the best point
    assert len(calls) == 12 + 1


def test_cma_training_improves_the_loss():
    pytest.importorskip("cma")

    eliobj = _cmaes_eliobj(epochs=40)
    eliobj.fit()

    loss = eliobj.results["history_stats/loss"]["total_loss"].values
    assert loss[0, -1] < loss[0, 0]


def test_step_size_reads_a_number():
    assert el.cmaes._step_size(0.3, ["mu0", "sigma0"]) == (0.3, None)


def test_step_size_orders_a_dict_like_the_variables():
    names = ["mu0", "sigma0", "mu1"]
    sigma0, stds = el.cmaes._step_size(dict(mu1=0.1, mu0=0.2), names)

    # the scalar is 1.0, so `CMA_stds` alone sets the step size
    assert sigma0 == 1.0
    # `sigma0` is not given, so it gets the default
    assert stds == [0.2, el.cmaes.DEFAULT_SIGMA0, 0.1]


def test_step_size_fills_a_missing_name_from_the_default():
    names = ["mu0", "sigma0", "mu1"]
    box = dict(mu0=3.5, sigma0=3.5, mu1=3.5)

    _, stds = el.cmaes._step_size(dict(mu1=0.1), names, box)

    # `mu1` is given, the other two keep the value of the box
    assert stds == [3.5, 3.5, 0.1]


def test_step_size_rejects_an_unknown_hyperparameter():
    with pytest.raises(ValueError, match="mu2"):
        el.cmaes._step_size(dict(mu2=0.1), ["mu0", "sigma0"])


def test_cma_training_accepts_a_step_size_per_hyperparameter():
    pytest.importorskip("cma")

    eliobj = _cmaes_eliobj(epochs=12, sigma0=dict(mu0=0.1, sigma2=1.0))
    eliobj.fit()

    loss = eliobj.results["history_stats/loss"]["total_loss"].values
    assert np.all(np.isfinite(loss))
    assert "sigma0=2 values" in repr(eliobj)


def test_cmaes_init_drops_its_search_for_a_cmaes_training():
    box = el.initialization.uniform(radius=1.0, mean=0.0)
    method = el.initialization.resolve_init_method(
        el.initializer(method="cmaes", distribution=box)
    )

    # the training runs the same search, so one search is enough
    assert method.skips_search(dict(optimizer=el.cmaes.CMAES))
    # a gradient training is a different search, so the start value is needed
    assert not method.skips_search(dict(optimizer=tf.keras.optimizers.Adam))


def test_warmstart_init_keeps_its_search_for_a_cmaes_training():
    box = el.initialization.uniform(radius=1.0, mean=0.0)
    method = el.initialization.resolve_init_method(
        el.initializer(method="warmstart", distribution=box)
    )

    assert not method.skips_search(dict(optimizer=el.cmaes.CMAES))


def test_box_step_size_reads_the_radius_of_the_box():
    expert_elicits, _ = el.utils.get_expert_data(
        base_eliobj.trainer,
        base_eliobj.model,
        base_eliobj.targets,
        base_eliobj.expert,
        base_eliobj.parameters,
        base_eliobj.network,
        base_eliobj.trainer["seed"],
    )
    box = el.initialization.uniform(radius=4.0, mean=0.0)

    sigma0 = el.cmaes.box_step_size(
        el.initializer(method="cmaes", distribution=box),
        expert_elicits,
        base_eliobj.parameters,
    )

    names = el.initialization.hyper_names(base_eliobj.parameters)
    assert sigma0 == {name: 2.0 for name in names}

    # a method that keeps its own search hands no box to the training
    other = el.cmaes.box_step_size(
        el.initializer(method="sobol", distribution=box),
        expert_elicits,
        base_eliobj.parameters,
    )
    assert other == el.cmaes.DEFAULT_SIGMA0


def test_cma_training_runs_no_search_before_it(monkeypatch):
    pytest.importorskip("cma")

    calls = []
    monkeypatch.setattr(el.cmaes, "cma_search", lambda **kwargs: calls.append(1) or {})

    eliobj = _cmaes_eliobj(epochs=12, sigma0=None, init_method="cmaes")
    eliobj.optimizer.pop("sigma0")
    eliobj.fit()

    assert calls == []
    loss = eliobj.results["history_stats/loss"]["total_loss"].values
    assert np.all(np.isfinite(loss))


def test_cmaes_optimizer_rejects_the_deep_prior_method():
    with pytest.raises(ValueError, match="parametric_prior"):
        _cmaes_eliobj(epochs=12, method="deep_prior")


def test_cmaes_optimizer_rejects_a_warmup():
    eliobj = _cmaes_eliobj(epochs=12)
    with pytest.raises(ValueError, match="warmup_epochs"):
        Elicit(
            model=eliobj.model,
            parameters=eliobj.parameters,
            targets=eliobj.targets,
            expert=eliobj.expert,
            optimizer=eliobj.optimizer,
            trainer=eliobj.trainer,
            initializer=el.initializer(
                method="sobol",
                iterations=2,
                warmup_epochs=2,
                distribution=el.initialization.uniform(radius=1.0, mean=0.0),
            ),
        )
