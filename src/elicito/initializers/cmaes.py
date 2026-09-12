"""
Initialization with a CMA-ES search
"""

from typing import Any

import tensorflow as tf

from elicito.initializers._search import _SearchStart
from elicito.optimizers import cmaes
from elicito.types import (
    ExpertDict,
    Parameter,
    Target,
    Trainer,
)


class CmaEs(_SearchStart):
    """Search for a start point with CMA-ES, over the whole box."""

    name = "cmaes"
    # objective evaluations, not candidates. A global search needs more of
    # them than the local warm start: it spends the first generations on
    # where the good region is, not on the value inside it.
    default_iterations = 500

    def skips_search(self, optimizer: dict[str, Any]) -> bool:
        """Whether the training repeats this search, so it can be dropped."""
        # `optimizer="cmaes"` runs this search again, from the point that
        # this search returns. The second run starts with a new covariance
        # matrix, so it drops what the first one learned. One run over the
        # whole budget is then better, and the box gives it its start point
        # and its step size.
        return bool(optimizer["optimizer"] == cmaes.CMAES)

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
        return cmaes.cma_search(
            expert_elicited_statistics=expert_elicited_statistics,
            parameters=parameters,
            trainer=trainer,
            model=model,
            targets=targets,
            expert=expert,
            distribution=distribution,
            max_evals=max_evals,
            seed=seed,
        )
