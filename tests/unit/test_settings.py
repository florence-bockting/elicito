"""
Tests of the flat settings of Elicit
"""

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el
from elicito import _settings
from elicito.optimizers.cmaes import CMAES

tfd = tfp.distributions


def _trainer(**kwargs):
    settings = dict(
        network=None,
        seeds=(1,),
        epochs=3,
        B=8,
        num_samples=10,
        kappa=0.0,
        verbose=True,
    )
    return _settings.trainer(**{**settings, **kwargs})


def test_trainer_infers_the_method():
    assert _trainer()["method"] == "parametric_prior"
    assert _trainer(network={})["method"] == "deep_prior"


def test_trainer_rejects_no_epochs():
    with pytest.raises(ValueError, match="Got epochs=0"):
        _trainer(epochs=0)


def test_trainer_rejects_a_negative_kappa():
    with pytest.raises(ValueError, match="Got kappa=-1.0"):
        _trainer(kappa=-1.0)


def test_auto_picks_cmaes_without_a_network():
    assert _settings.optimizer("auto", None, network=None)["optimizer"] == CMAES


def test_auto_picks_adam_with_a_network():
    opt = _settings.optimizer("auto", None, network={})
    assert opt == dict(optimizer=tf.keras.optimizers.Adam, learning_rate=0.1)


def test_a_keras_optimizer_keeps_its_settings():
    given = tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
    opt = _settings.optimizer(given, None, None)
    built = opt.pop("optimizer")(**opt)
    assert isinstance(built, tf.keras.optimizers.Adam)
    assert built is not given
    assert built.clipnorm == 1.0
    assert float(built.learning_rate.numpy()) == pytest.approx(0.01)


def test_cmaes_settings_go_to_the_search():
    opt = _settings.optimizer("cmaes", dict(sigma0=0.5, popsize=4), None)
    assert opt == dict(optimizer=CMAES, sigma0=0.5, popsize=4)


def test_cmaes_settings_without_cmaes_warn():
    with pytest.warns(UserWarning, match="is ignored"):
        _settings.optimizer("adam", dict(sigma0=0.5), None)


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError, match="optimizer must be"):
        _settings.optimizer("sgd", None, None)


def test_a_callable_model_needs_parameters():
    with pytest.raises(ValueError, match="needs parameters"):
        _settings.model(lambda prior_samples: prior_samples, None)


class _DimsModel:
    """a model with the dims of one output, as the PyMC adapter gives"""

    def __init__(self):
        self.dims = {"y": [("obs", (10, 20, 30)), ("rep", None)]}

    def __call__(self, prior_samples):
        return prior_samples


ALL = slice(None)


@pytest.mark.parametrize(
    ("select", "expected"),
    [
        (None, None),
        (2, (Ellipsis, 2)),
        ({"obs": 20}, (ALL, ALL, 1, ALL)),
        ((ALL, ALL, 0), (ALL, ALL, 0)),
    ],
)
def test_select_becomes_an_index(select, expected):
    target = el.identity("y", select=select)
    assert _settings.select(target, _DimsModel()) == expected


def test_targets_copies_the_targets():
    target = el.identity("y", select={"obs": 30})
    (copy,) = _settings.targets([target], _DimsModel())
    assert copy["select"] == (ALL, ALL, 2, ALL)
    assert target["select"] == {"obs": 30}


@pytest.mark.parametrize(
    ("target", "match"),
    [
        (el.identity("y", select={"obs": 40}), "40 is not a label of 'obs'"),
        (el.identity("y", select={"rep": 0}), "0 is not a label of 'rep'"),
        (el.identity("y", select={"time": 1}), r"has no dims \['time'\]"),
        (el.identity("z", select={"obs": 10}), "needs a PyMC output with dims"),
        (
            el.identity("f", output=lambda y: y, select={"obs": 10}),
            "needs a PyMC output with dims",
        ),
    ],
)
def test_select_raises_for_a_wrong_dict(target, match):
    with pytest.raises(ValueError, match=match):
        _settings.select(target, _DimsModel())


PANEL_PARAMETERS = [
    el.parameter(
        name="beta0",
        family=tfd.Normal,
        hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)),
    ),
]


def test_pool_accepts_none_complete_and_partial():
    pool = el.partial(over=["mu0"], tau=0.5)

    assert _settings.pool(None, None, PANEL_PARAMETERS) is None
    assert _settings.pool("complete", None, PANEL_PARAMETERS) == "complete"
    assert _settings.pool(pool, None, PANEL_PARAMETERS) == pool


def test_partial_keeps_over_as_a_tuple():
    assert el.partial(over=["mu0"]) == dict(over=("mu0",), tau=None)


@pytest.mark.parametrize(
    ("pool", "network", "match"),
    [
        ("complete", {}, "needs a parametric prior"),
        ("none", None, "pool must be"),
        (el.partial(over=[]), None, "must name hyperparameters"),
        (el.partial(over=["mu9"]), None, r"Unknown: \['mu9'\]"),
        (el.partial(over=["mu0"], tau={"sigma0": 1.0}), None, "one value for each"),
        (el.partial(over=["mu0"], tau=0.0), None, "must be positive"),
        (el.partial(over=["mu0"], tau={"mu0": -1.0}), None, "must be positive"),
    ],
)
def test_pool_rejects_a_wrong_setting(pool, network, match):
    with pytest.raises(ValueError, match=match):
        _settings.pool(pool, network, PANEL_PARAMETERS)


def test_elicit_stores_the_pool():
    pool = el.partial(over=["mu0"])
    eliobj = el.Elicit(
        lambda prior_samples: dict(y=prior_samples[..., 0]),
        [el.quantiles("y", probs=(0.25, 0.75))],
        PANEL_PARAMETERS,
        pool=pool,
        dry_run=False,
    )
    assert eliobj.settings["pool"] == pool
