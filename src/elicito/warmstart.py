"""
Derivative-free warm start for the hyperparameters of a parametric prior
"""

import logging
from typing import Any

import numpy as np
import tensorflow as tf

from elicito.exceptions import MissingOptionalDependencyError
from elicito.optimizers import search
from elicito.optimizers.search import (
    MIN_SEARCH_SAMPLES,
    PENALTY,
    SEARCH_FRACTION,
    hyper_names,
    start_vector,
)
from elicito.types import ExpertDict, Parameter, Target, Trainer

logger = logging.getLogger(__name__)


def warm_start(  # noqa: PLR0913
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
    """
    Search a start value with Nelder-Mead, before any gradient step

    The search needs no gradient, so it cannot diverge through an exploding
    gradient. It runs on the unconstrained scale, and evaluates the same
    loss the training uses, on fewer prior draws.

    The search minimises the loss at the start, which does not predict the
    loss after training. Measured on the case study with a lognormal noise
    family, it improved the worst box from 98.6 to 12.8, and made a
    well-placed box worse, from 0.68 to 1.55.

    Parameters
    ----------
    expert_elicited_statistics
        Elicited statistics of the expert.

    parameters
        List including dictionary with all information about the
        (hyper-)parameters.

    trainer
        Specification of trainer settings.

    model
        Generative model.

    targets
        Elicitation techniques and target quantities.

    expert
        Expert specification.

    distribution
        Initialization box. Its centre is the start point of the search.

    max_evals
        Budget, in objective evaluations.

    seed
        Seed used for the forward simulation.

    Raises
    ------
    MissingOptionalDependencyError
        ``scipy`` is required for the search.

    Returns
    -------
    hyperparams :
        One value per hyperparameter, on the unconstrained scale.

    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise MissingOptionalDependencyError("warm_start", requirement="scipy") from exc

    names = hyper_names(parameters)
    search_trainer = dict(trainer)
    search_trainer["num_samples"] = max(
        MIN_SEARCH_SAMPLES, trainer["num_samples"] // SEARCH_FRACTION
    )

    # Nelder-Mead can end on a point that failed, because a failure is scored
    # as a finite number. Keep the best usable point of the search instead.
    best: dict[str, Any] = {"value": PENALTY, "values": None}

    scorer = search.compile_score(
        expert_elicited_statistics=expert_elicited_statistics,
        parameters=parameters,
        trainer=search_trainer,  # type: ignore [arg-type]
        model=model,
        targets=targets,
        expert=expert,
        seed=seed,
    )

    def objective(values: Any) -> float:
        value = float(scorer(dict(zip(names, values))))
        if value < best["value"]:
            best["value"] = value
            best["values"] = np.array(values, dtype=np.float64)
        return value

    # the scipy stubs describe the objective as a variadic callable over a
    # float64 array, which no plain function matches
    minimizer: Any = minimize
    start = start_vector(distribution, names)

    # Nelder-Mead converges on its own tolerances, and it can stall inside the
    # failing region long before the budget is spent. Restart the simplex from
    # the best point, so that the whole budget is used. Each restart builds a
    # fresh simplex, which leaves a shallow stall.
    used = 0
    values = np.asarray(start, dtype=np.float64)
    result: Any = None
    while used < max_evals:
        result = minimizer(
            objective,
            values,
            method="Nelder-Mead",
            options={"maxfev": int(max_evals - used)},
        )
        used += int(result.nfev)
        values = np.asarray(result.x, dtype=np.float64)
        if result.fun < PENALTY:
            # the point is usable; more evaluations only refine a start value
            break

    logger.info(f"warm start: loss {result.fun:.4f} after {used} evaluations.")
    if result is None or not np.isfinite(result.fun) or result.fun >= PENALTY:
        if best["values"] is None:
            logger.warning(
                "warm start: every evaluated point failed. The centre of the "
                "initialization box is used as the start value. Re-centre the "
                "box, or reduce its radius."
            )
            values = np.asarray(start, dtype=np.float64)
        else:
            logger.warning(
                "warm start: the search ended on a point that failed. The best "
                f"usable point, with loss {best['value']:.4f}, is used instead."
            )
            values = best["values"]

    return {name: float(v) for name, v in zip(names, values)}
