"""
Base class of the initialization methods that run a search
"""

import logging
from typing import Any, Optional

import tensorflow as tf

from elicito.initializers._base import InitResult, _check_box
from elicito.initializers.exact import ExactValues
from elicito.initializers.sampling import BoxSample
from elicito.optimizers.search import hyper_names, start_vector
from elicito.types import (
    ExpertDict,
    Initializer,
    NFDict,
    Parameter,
    Target,
    Trainer,
)

logger = logging.getLogger(__name__)


class _SearchStart:
    """A start value that a derivative-free search picks out of the box."""

    name: str
    default_iterations: int

    def check(self, initializer: Initializer) -> None:
        """Reject an input this method cannot use."""
        _check_box(initializer)

    def skips_search(self, optimizer: dict[str, Any]) -> bool:
        """Whether the training repeats this search, so it can be dropped."""
        return False

    def search(  # noqa: PLR0913
        self,
        expert_elicited_statistics: dict[str, tf.Tensor],
        parameters: list[Parameter],
        trainer: Trainer,
        model: dict[str, Any],
        targets: list[Target],
        expert: ExpertDict,
        distribution: dict[str, Any],
        max_evals: int,
        seed: int,
    ) -> dict[str, Any]:
        """Return one value per hyperparameter, on the unconstrained scale."""
        raise NotImplementedError

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
        """Search for the start values, then build the prior model."""
        distribution = initializer["distribution"]
        iterations = initializer["iterations"]
        if distribution is None or iterations is None:
            # check() rejects this earlier; the guard narrows the type
            msg = f"Method {self.name!r} needs 'distribution' and 'iterations'."
            raise ValueError(msg)

        # a derivative-free search needs no gradient, so it cannot diverge.
        # The copy keeps the user's Elicit object unchanged.
        initializer = dict(initializer)  # type: ignore [assignment]
        if self.skips_search(optimizer):
            logger.info(
                f"{self.name}: the training runs the same search, so the "
                "initialization only reads the box. 'iterations' is not used."
            )
            names = hyper_names(parameters)
            centre = start_vector(dict(distribution), names)
            initializer["hyperparams"] = dict(zip(names, centre))
        else:
            initializer["hyperparams"] = self.search(
                expert_elicited_statistics=expert_elicited_statistics,
                parameters=parameters,
                trainer=trainer,
                model=model,
                targets=targets,
                expert=expert,
                # dict() satisfies the signature; a TypedDict is invariant
                distribution=dict(distribution),
                max_evals=iterations,
                seed=seed,
            )
        return ExactValues().propose(
            expert_elicited_statistics=expert_elicited_statistics,
            initializer=initializer,
            parameters=parameters,
            trainer=trainer,
            optimizer=optimizer,
            model=model,
            targets=targets,
            network=network,
            expert=expert,
            seed=seed,
            progress=progress,
        )

    def dry_run_slice(
        self,
        initializer: Initializer,
        parameters: list[Parameter],
        trainer: Trainer,
    ) -> Any:
        """Return a slice of the right shape for ``Elicit.__init__``."""
        # the dry run only needs a slice of the right shape. The search
        # looks for the real values during `fit`.
        initializer = dict(initializer)  # type: ignore [assignment]
        initializer["method"] = "random"
        return BoxSample().dry_run_slice(initializer, parameters, trainer)
