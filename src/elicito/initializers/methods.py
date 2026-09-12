"""
Protocol of the initialization methods, their registry, and the entry point
"""

from typing import Any, Optional, Protocol

import tensorflow as tf

from elicito.initializers._base import InitResult
from elicito.initializers.cmaes import CmaEs
from elicito.initializers.exact import ExactValues
from elicito.initializers.sampling import BoxSample
from elicito.initializers.warmstart import WarmStart
from elicito.parameters.priors import Priors
from elicito.types import (
    ExpertDict,
    Initializer,
    NFDict,
    Parameter,
    Target,
    Trainer,
)


class InitMethod(Protocol):
    """Behaviour that differs between the initialization methods."""

    name: str
    default_iterations: int

    def check(self, initializer: Initializer) -> None:
        """Reject an input this method cannot use."""
        ...

    def skips_search(self, optimizer: dict[str, Any]) -> bool:
        """Whether the training repeats this search, so it can be dropped."""
        ...

    def propose(  # noqa: PLR0913
        self,
        expert_elicited_statistics: dict[str, tf.Tensor],
        initializer: Initializer,
        parameters: list[Parameter],
        trainer: Trainer,
        optimizer: dict[str, Any],
        model: dict[str, Any],
        targets: list[Target],
        network: Optional[NFDict],
        expert: ExpertDict,
        seed: int,
        progress: int,
    ) -> InitResult:
        """Pick the hyperparameters that start the training."""
        ...

    def dry_run_slice(
        self,
        initializer: Initializer,
        parameters: list[Parameter],
        trainer: Trainer,
    ) -> Any:
        """Return a slice of the right shape for ``Elicit.__init__``."""
        ...


_INIT_METHODS: dict[str, type[InitMethod]] = {
    "sobol": BoxSample,
    "lhs": BoxSample,
    "random": BoxSample,
    WarmStart.name: WarmStart,
    CmaEs.name: CmaEs,
}


def get_init_method(name: str) -> InitMethod:
    """Return a new strategy object for an ``initializer["method"]`` string."""
    try:
        method_cls = _INIT_METHODS[name]
    except KeyError:
        msg = (
            "Currently implemented initialization methods are "
            f"{', '.join(repr(key) for key in sorted(_INIT_METHODS))}, but got "
            f"method={name!r} as input."
        )
        raise ValueError(msg) from None
    return method_cls()


def resolve_init_method(initializer: Initializer) -> InitMethod:
    """Return the initialization method that ``initializer`` asks for."""
    # exact values are chosen by their presence, not by a method string
    if initializer["hyperparams"] is not None:
        return ExactValues()

    name = initializer["method"]
    if name is None:
        msg = (
            "Either 'method' or 'hyperparams' has"
            "to be specified. Use method for sampling from an"
            "initialization distribution and 'hyperparams' for"
            "specifying exact initial values per hyperparameter."
        )
        raise ValueError(msg)
    return get_init_method(name)


def init_prior(  # noqa: PLR0913
    expert_elicited_statistics: dict[str, tf.Tensor],
    initializer: Optional[Initializer],
    parameters: list[Parameter],
    trainer: Trainer,
    optimizer: dict[str, Any],
    model: dict[str, Any],
    targets: list[Target],
    network: Optional[NFDict],
    expert: ExpertDict,
    seed: int,
    progress: int,
) -> tuple[Any, Any, Any]:
    """
    Extract target loss and initialize prior model

    Parameters
    ----------
    expert_elicited_statistics
        Expert-elicited statistics

    initializer
        Initialization of hyperparameter values

    parameters
        Specification of model parameters

    trainer
        Specification of trainer settings for the optimization process

    optimizer
        User-input from [`optimizer`][elicito.specs.optimizer]. Used to run
        the warm-up epochs of a candidate.

    model
        Generative model

    targets
        Elicitation techniques and target quantities

    network
        Generative model for learning non-parametric priors

    expert
        Expert specification

    seed
        Internally used seed for reproducible results

    progress
        whether progress should be printed or muted

    Returns
    -------
    init_prior_model :
        initialized priors that will be used for the training phase.

    loss_list :
        list with all losses computed for each initialization run.

    init_matrix :
        dictionary with *keys* being the hyperparameter names and *values*
        being the drawn initial values per run.

    """
    if initializer is None:
        # check() allows no initializer for deep_prior only; it runs no search
        prior_model = Priors(
            ground_truth=False,
            init_matrix_slice=None,
            trainer=trainer,
            parameters=parameters,
            network=network,
            expert=expert,
            seed=seed,
        )
        return prior_model, None, None

    result = resolve_init_method(initializer).propose(
        expert_elicited_statistics=expert_elicited_statistics,
        initializer=initializer,
        parameters=parameters,
        trainer=trainer,
        optimizer=optimizer,
        model=model,
        targets=targets,
        network=None,
        expert=expert,
        seed=seed,
        progress=progress,
    )
    return result.prior_model, result.losses, result.candidates
