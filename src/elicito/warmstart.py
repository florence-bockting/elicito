"""
Derivative-free warm start for the hyperparameters of a parametric prior
"""

import logging
from typing import Any

import numpy as np
import tensorflow as tf

import elicito as el
from elicito.exceptions import MissingOptionalDependencyError
from elicito.types import ExpertDict, Parameter, Target, Trainer

logger = logging.getLogger(__name__)

# The search uses a quarter of the training draws. A noisier objective is
# acceptable, because the result is only a start value.
SEARCH_FRACTION = 4
MIN_SEARCH_SAMPLES = 100

# Value reported for a set of hyperparameters that cannot be used. Nelder-Mead
# needs a finite number, and a usable point always scores far below this one.
PENALTY = 1e12


def score(  # noqa: PLR0913
    hyperparams: dict[str, Any],
    expert_elicited_statistics: dict[str, tf.Tensor],
    parameters: list[Parameter],
    trainer: Trainer,
    model: dict[str, Any],
    targets: list[Target],
    expert: ExpertDict,
    seed: int,
) -> float:
    """
    Compute the loss of one set of hyperparameter values

    The values are used as they are, without any training. A non-finite loss
    is reported as the largest float, so that a search can steer away from it.

    Parameters
    ----------
    hyperparams
        One value per hyperparameter, on the unconstrained scale.

    expert_elicited_statistics
        Elicited statistics of the expert.

    parameters
        List including dictionary with all information about the
        (hyper-)parameters.

    trainer
        Specification of trainer settings. Its ``num_samples`` decides how
        many prior draws the loss uses.

    model
        Generative model.

    targets
        Elicitation techniques and target quantities.

    expert
        Expert specification.

    seed
        Seed used for the forward simulation.

    Returns
    -------
    loss :
        Total loss against the expert-elicited statistics.

    """
    prior_model = el.simulations.Priors(
        ground_truth=False,
        init_matrix_slice={
            name: tf.constant(float(value), dtype=tf.float32)
            for name, value in hyperparams.items()
        },
        trainer=trainer,
        parameters=parameters,
        network=None,
        expert=expert,
        seed=seed,
    )
    (elicited, _, _, target_quantities) = el.utils.one_forward_simulation(
        prior_model=prior_model, model=model, targets=targets, seed=seed
    )
    (loss, *_) = el.losses.total_loss(
        elicit_training=elicited,
        elicit_expert=expert_elicited_statistics,
        targets=targets,
    )
    value = float(loss)
    # A quantile query hides an overflow, so the loss alone is not enough.
    # One flat failure value would give Nelder-Mead nothing to follow, so
    # grade the penalty by the share of draws that overflow. The search can
    # then walk out of the bad region.
    bad = el.utils.nonfinite_fraction(target_quantities)
    if bad > 0.0:
        return PENALTY * (1.0 + bad)
    # Nelder-Mead cannot use a non-finite value. Steer it away instead.
    return value if np.isfinite(value) else PENALTY * 2.0


def _start_vector(box: dict[str, Any], names: list[str]) -> list[float]:
    """Read one start value per hyperparameter out of the box."""
    mean = box["mean"]
    if np.isscalar(mean):
        return [float(mean)] * len(names)  # type: ignore [arg-type]
    order = box["hyper"] if box["hyper"] is not None else names
    lookup = dict(zip(order, mean))
    return [float(lookup[name]) for name in names]


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

    names = el.initialization.hyper_names(parameters)
    box = el.initialization.build_box(
        distribution, expert_elicited_statistics, parameters
    )

    search_trainer = dict(trainer)
    search_trainer["num_samples"] = max(
        MIN_SEARCH_SAMPLES, trainer["num_samples"] // SEARCH_FRACTION
    )

    # Nelder-Mead can end on a point that failed, because a failure is scored
    # as a finite number. Keep the best usable point of the search instead.
    best: dict[str, Any] = {"value": PENALTY, "values": None}

    def objective(values: Any) -> float:
        value = score(
            hyperparams=dict(zip(names, values)),
            expert_elicited_statistics=expert_elicited_statistics,
            parameters=parameters,
            trainer=search_trainer,  # type: ignore [arg-type]
            model=model,
            targets=targets,
            expert=expert,
            seed=seed,
        )
        if value < best["value"]:
            best["value"] = value
            best["values"] = np.array(values, dtype=np.float64)
        return value

    # the scipy stubs describe the objective as a variadic callable over a
    # float64 array, which no plain function matches
    search: Any = minimize
    start = _start_vector(box, names)

    # Nelder-Mead converges on its own tolerances, and it can stall inside the
    # failing region long before the budget is spent. Restart the simplex from
    # the best point, so that the whole budget is used. Each restart builds a
    # fresh simplex, which leaves a shallow stall.
    used = 0
    values = np.asarray(start, dtype=np.float64)
    result: Any = None
    while used < max_evals:
        result = search(
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
