"""
Tests of the target constructors
"""

import elicito as el


def test_quantiles_defaults():
    tar = el.quantiles("y_X0")
    assert tar["output"] == "y_X0"
    assert tar["query"]["value"] == (5.0, 50.0, 95.0)
    assert isinstance(tar["loss"], el.losses.MMD2)
    assert tar["weight"] == 1.0


def test_quantiles_callable_output():
    def r2(ypred, epred):
        return epred

    tar = el.quantiles("R2", output=r2, loss=el.losses.L2)
    assert tar["output"] is r2
    assert tar["loss"] is el.losses.L2


def test_identity_defaults():
    tar = el.identity("mu")
    assert tar["query"]["name"] == "identity"
    assert tar["output"] == "mu"
    assert isinstance(tar["loss"], el.losses.MMD2)


def test_correlation_defaults():
    tar = el.correlation("cor")
    assert tar["query"]["name"] == "pearson_correlation"
    assert tar["loss"] is el.losses.L2


def test_custom_keeps_func():
    def mean(ypred):
        return ypred

    tar = el.custom("mean", func=mean, loss=el.losses.L2, weight=0.5)
    assert tar["query"]["name"] == "custom"
    assert tar["query"]["value"] is mean
    assert tar["weight"] == 0.5
