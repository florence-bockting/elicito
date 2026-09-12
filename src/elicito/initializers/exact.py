"""
Initialization with exact start values
"""

from typing import Any, Optional

import tensorflow as tf

from elicito.initializers._base import InitResult
from elicito.parameters.priors import Priors
from elicito.types import (
    ExpertDict,
    Initializer,
    NFDict,
    Parameter,
    Target,
    Trainer,
)


class ExactValues:
    """Start from hyperparameter values the user supplied."""

    name = "exact"
    default_iterations = 0  # nothing is drawn

    def check(self, initializer: Initializer) -> None:
        """Reject an input this method cannot use."""
        if initializer["hyperparams"] is None:
            msg = "Method 'exact' needs 'hyperparams'."
            raise ValueError(msg)

    def skips_search(self, optimizer: dict[str, Any]) -> bool:
        """Whether the training repeats this search, so it can be dropped."""
        return False

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
        """Build the prior model from the given values."""
        prior_model = Priors(
            ground_truth=False,
            init_matrix_slice=initializer["hyperparams"],
            trainer=trainer,
            parameters=parameters,
            network=None,
            expert=expert,
            seed=seed,
        )
        return InitResult(prior_model=prior_model)

    def dry_run_slice(
        self,
        initializer: Initializer,
        parameters: list[Parameter],
        trainer: Trainer,
    ) -> Any:
        """Return a slice of the right shape for ``Elicit.__init__``."""
        return initializer["hyperparams"]
