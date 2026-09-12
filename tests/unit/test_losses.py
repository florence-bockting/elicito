"""
Unittests for losses.py module
"""

import tensorflow as tf

from elicito.losses import spread_penalty


def draws(sd: float, seed: int = 0) -> tf.Tensor:
    """Return prior draws of shape [B, num_samples, num_params]."""
    return tf.random.stateless_normal((4, 500, 2), seed=(seed, seed), stddev=sd)


def test_spread_penalty_is_a_scalar():
    assert spread_penalty(draws(1.0)).shape == ()


def test_spread_penalty_falls_with_the_spread():
    assert spread_penalty(draws(2.0)) < spread_penalty(draws(0.5))


def test_spread_penalty_is_large_and_finite_at_a_point_mass():
    penalty = spread_penalty(tf.zeros((4, 500, 2)))
    assert tf.math.is_finite(penalty)
    assert penalty > 18.0  # -log(1e-8) = 18.4


def test_spread_penalty_has_a_gradient_that_widens_the_prior():
    scale = tf.Variable(0.5)
    with tf.GradientTape() as tape:
        penalty = spread_penalty(scale * draws(1.0))
    # a larger scale lowers the penalty
    assert tape.gradient(penalty, scale) < 0.0
