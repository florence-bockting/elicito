"""
Generative model, and the forward pass from prior samples to elicited statistics
"""

import inspect
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito.parameters._base import seed_pair
from elicito.parameters.priors import Priors
from elicito.targets import (
    computation_elicited_statistics,
    computation_target_quantities,
)
from elicito.types import Target

tfd = tfp.distributions


def simulate_from_generator(
    prior_samples: tf.Tensor,
    seed: int,
    model: dict[str, Any],  # shape=[B,num_samples,num_params]
) -> Any:
    """
    Simulate data from the specified generative model.

    Parameters
    ----------
    prior_samples
        Samples from prior distributions.

    seed
        Seed used for learning. Specification in :func:`elicit.elicit.trainer`.

    model
        Specification of generative model using :func:`elicit.elicit.model`.

    Returns
    -------
    model_simulations :
        simulated data from generative model.

    """
    # get model and initialize generative model
    GenerativeModel = model["obj"]
    generative_model = GenerativeModel()
    # get model specific arguments (that are not prior samples)
    add_model_args = model.copy()
    add_model_args.pop("obj")
    signature = inspect.signature(generative_model.__call__)
    if "seed" in signature.parameters and "seed" not in add_model_args:
        add_model_args["seed"] = tfp.random.split_seed(
            seed_pair(seed), n=1, salt="model"
        )[0]
    # simulate from generator
    if len(add_model_args) < 1:
        model_simulations = generative_model(prior_samples)
    else:
        model_simulations = generative_model(prior_samples, **add_model_args)

    return model_simulations


def all_finite(quantities: dict[str, Any]) -> bool:
    """
    Report whether every target quantity is finite

    A quantile query hides an overflow: the 95% quantile of a sample with a
    few infinite draws is still finite. Check the target quantities to see
    the overflow.

    Parameters
    ----------
    quantities
        Target quantities of one forward simulation.

    Returns
    -------
    finite :
        ``True`` if no target quantity holds an infinite or NAN value.

    """
    return all(
        bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, tf.float32))))
        for value in quantities.values()
    )


def nonfinite_fraction(quantities: dict[str, Any]) -> float:
    """
    Compute the share of target quantities that are not finite

    A search needs to know how bad a failure is, not only that it failed.

    Parameters
    ----------
    quantities
        Target quantities of one forward simulation.

    Returns
    -------
    fraction :
        Share of infinite or NAN values, over all target quantities.

    """
    bad = 0.0
    total = 0.0
    for value in quantities.values():
        tensor = tf.cast(value, tf.float32)
        bad += float(tf.reduce_sum(tf.cast(~tf.math.is_finite(tensor), tf.float32)))
        total += float(tf.size(tensor, out_type=tf.int64))
    if total == 0.0:
        return 0.0
    return bad / total


def one_forward_simulation(
    prior_model: Priors, model: dict[str, Any], targets: list[Target], seed: int
) -> tuple[dict[Any, Any], tf.Tensor, dict[Any, Any], dict[Any, Any]]:
    """
    Run one forward simulation from prior samples to elicited statistics.

    The seed is set here. The simulation itself is done by
    [`simulate_and_elicit`][elicito.models.simulate_and_elicit].

    Parameters
    ----------
    prior_model
        Initialized prior distributions which can be used for sampling.

    model
        Specification of generative model

    targets
        List of target quantities

    seed
        Random seed.

    Returns
    -------
    elicited_statistics :
        Dictionary containing the elicited statistics that can be used to
        compute the loss components

    prior_samples :
        Samples from prior distributions

    model_simulations :
        Samples from the generative model (likelihood) given the prior samples
        for the model parameters

    target_quantities :
        Target quantities as a function of the model simulations.

    """
    # set seed
    tf.random.set_seed(seed)
    return simulate_and_elicit(prior_model, model, targets, seed)


def simulate_and_elicit(
    prior_model: Priors, model: dict[str, Any], targets: list[Target], seed: int
) -> tuple[dict[Any, Any], tf.Tensor, dict[Any, Any], dict[Any, Any]]:
    """
    Run one forward simulation, without setting the seed

    The seed is set by the caller. A caller that compiles this function with
    ``tf.function`` must set the seed before every call: a
    ``tf.random.set_seed`` inside a graph runs at trace time only.

    Parameters
    ----------
    prior_model
        Initialized prior distributions which can be used for sampling.

    model
        Specification of generative model

    targets
        List of target quantities

    seed
        Random seed.

    Returns
    -------
    elicited_statistics :
        Dictionary containing the elicited statistics that can be used to
        compute the loss components

    prior_samples :
        Samples from prior distributions

    model_simulations :
        Samples from the generative model (likelihood) given the prior samples
        for the model parameters

    target_quantities :
        Target quantities as a function of the model simulations.

    """
    # generate samples from initialized prior
    prior_samples = prior_model()
    # simulate prior predictive distribution based on prior samples
    # and generative model
    model_simulations = simulate_from_generator(prior_samples, seed, model)
    # compute the target quantities
    target_quantities = computation_target_quantities(
        model_simulations, prior_samples, targets
    )
    # compute the elicited statistics by applying a specific elicitation
    # method on the target quantities
    elicited_statistics = computation_elicited_statistics(target_quantities, targets)
    return (elicited_statistics, prior_samples, model_simulations, target_quantities)
