"""
Tests of the flat settings of Elicit
"""

import pytest
import tensorflow as tf

from elicito import _settings
from elicito.optimizers.cmaes import CMAES


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
