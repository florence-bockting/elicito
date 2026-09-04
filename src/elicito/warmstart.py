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

    def objective(values: Any) -> float:
        prior_model = el.simulations.Priors(
            ground_truth=False,
            init_matrix_slice={
                name: tf.constant(float(v), dtype=tf.float32)
                for name, v in zip(names, values)
            },
            trainer=search_trainer,  # type: ignore [arg-type]
            parameters=parameters,
            network=None,
            expert=expert,
            seed=seed,
        )
        (elicited, *_) = el.utils.one_forward_simulation(
            prior_model=prior_model, model=model, targets=targets, seed=seed
        )
        (loss, *_) = el.losses.total_loss(
            elicit_training=elicited,
            elicit_expert=expert_elicited_statistics,
            targets=targets,
        )
        value = float(loss)
        # Nelder-Mead cannot use a non-finite value. Steer it away instead.
        return value if np.isfinite(value) else float(np.finfo(np.float64).max)

    # the scipy stubs describe the objective as a variadic callable over a
    # float64 array, which no plain function matches
    search: Any = minimize
    result = search(
        objective,
        np.asarray(_start_vector(box, names), dtype=np.float64),
        method="Nelder-Mead",
        options={"maxfev": int(max_evals)},
    )
    logger.info(f"warm start: loss {result.fun:.4f} after {result.nfev} evaluations.")

    return {name: float(v) for name, v in zip(names, result.x)}
