"""
Simulations from prior and model
"""

import inspect
from typing import Any, Callable, Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito.methods import get_method, seed_pair
from elicito.targets import (
    computation_elicited_statistics,
    computation_target_quantities,
)
from elicito.types import ExpertDict, NFDict, Parameter, Target, Trainer

tfd = tfp.distributions


# initalize generator model
class Priors(tf.Module):
    """
    Initialize the hyperparameters (i.e., trainable variables)

    Parameters
    ----------
    ground_truth
        True if expert data are simulated from a given ground truth (oracle)

    init_matrix_slice
        Samples drawn from the initialization distribution to initialize
        the hyperparameter of the parametric prior distributions
        Only required for `method = "parametric_prior"` otherwise None.
    trainer
        Specification of training settings

    parameters
        List of model parameters

    network
        Specification of neural network
        Only required for ``deep_prior`` method.
        For ``parametric_prior`` use ``None``.

    expert
        Provide input data from expert or simulate data from oracle with
        either the ``data`` or ``simulator`` method

    seed
        Seed used for learning.
    """

    def __init__(  # noqa: PLR0913
        self,
        ground_truth: bool,
        init_matrix_slice: Optional[dict[str, tf.Tensor]],
        trainer: Trainer,
        parameters: list[Parameter],
        network: Optional[NFDict],
        expert: ExpertDict,
        seed: int,
    ):
        self.ground_truth = ground_truth
        self.init_matrix_slice = init_matrix_slice
        self.trainer = trainer
        self.parameters = parameters
        self.network = network
        self.expert = expert
        # initialize new attribute
        self.init_priors: Optional[dict[str, tf.Tensor]]
        # set seed
        tf.random.set_seed(seed)
        # initialize hyperparameter for learning (if true hyperparameter
        # are given, no initialization is needed)
        if not self.ground_truth:
            self.init_priors = intialize_priors(
                self.init_matrix_slice,
                self.trainer["method"],
                seed,
                self.parameters,
                self.network,
            )

        else:
            self.init_priors = None

    def __call__(self) -> Any:  # shape=[B,num_samples,num_params]
        """
        Sample from the initialized prior distribution(s).

        Returns
        -------
        prior_samples
            Samples from prior distribution(s).

        """
        prior_samples = sample_from_priors(
            initialized_priors=self.init_priors,
            ground_truth=self.ground_truth,
            num_samples=self.trainer["num_samples"],
            B=self.trainer["B"],
            seed=self.trainer["seed"],
            method=self.trainer["method"],
            parameters=self.parameters,
            network=self.network,
            expert=self.expert,
        )

        return prior_samples


def intialize_priors(
    init_matrix_slice: Optional[dict[str, tf.Tensor]],
    method: str,
    seed: int,
    parameters: list[Parameter],
    network: Optional[NFDict],
) -> Any:
    """
    Initialize prior distributions.

    Parameters
    ----------
    init_matrix_slice
        Samples drawn from the initialization distribution to initialize
        the hyperparameter of the parametric prior distributions
        Only for method="parametric_prior", otherwise None.

    method
        Parametric_prior or deep_prior method

    seed
        Seed of current workflow run

    parameters
        List of model parameter

    network
        specification of neural network
        Only required for ``deep_prior`` method. For ``parametric_prior``
        use ``None``.

    Returns
    -------
    init_prior :
        returns initialized prior distributions ready for prior sampling.

    """
    tf.random.set_seed(seed)
    return get_method(method).build(parameters, network, init_matrix_slice, seed)


@tf.autograph.experimental.do_not_convert  # type: ignore [misc]
def sample_from_priors(  # noqa: PLR0913
    initialized_priors: Union[None, dict[str, tf.Tensor], Callable[[Any], Any]],
    ground_truth: bool,
    num_samples: int,
    B: int,
    seed: int,
    method: str,
    parameters: list[Parameter],
    network: Optional[NFDict],
    expert: ExpertDict,
) -> Any:  # shape=[B,num_samples,num_params]
    """
    Sample from initialized prior distributions.

    Parameters
    ----------
    initialized_priors
        Initialized prior distributions ready for prior sampling.

    ground_truth
        True if expert data is simulated from ground truth.

    num_samples
        Number of samples from the prior(s).

    B
        Batch size.

    seed
        Seed used for learning.

    method
        Parametric_prior or deep_prior method

    parameters
        List of model parameters

    network
        Specification of neural network
        Only required for ``deep_prior`` method. For ``parametric_prior``
        use ``None``.

    expert
        Provide input data from expert or simulate data from oracle with
        either the ``data`` or ``simulator`` method

    Returns
    -------
    prior_samples :
        Samples from prior distributions.

    """
    if ground_truth:
        # number of samples for ground truth
        rep_true = expert["num_samples"]
        priors = []

        truths = list(expert["ground_truth"].values())
        # one stateless seed per distribution, see `ParametricPrior.sample`
        seeds = tfp.random.split_seed(seed_pair(seed), n=len(truths), salt="truth")
        for pr, pr_seed in zip(truths, seeds):
            # sample from the prior distribution
            prior_sample = pr.sample((1, rep_true), seed=pr_seed)
            # ensure that all samples have the same shape
            try:
                prior_sample.shape
            except AttributeError:
                prior = prior_sample
            else:
                if len(prior_sample.shape) < 3:  # noqa: PLR2004
                    prior = tf.expand_dims(prior_sample, -1)
                else:
                    prior = prior_sample

            priors.append(prior)
        # concatenate all prior samples into one tensor
        if type(priors[0]) is list:
            priors = priors[0]
        prior_samples = tf.concat(priors, axis=-1)
        return prior_samples

    return get_method(method).sample(
        initialized_priors, parameters, network, B, num_samples, seed
    )


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
    [`simulate_and_elicit`][elicito.simulations.simulate_and_elicit].

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
