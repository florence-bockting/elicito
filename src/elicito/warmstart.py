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
    (value, _) = evaluate(
        prior_model=prior_model,
        expert_elicited_statistics=expert_elicited_statistics,
        model=model,
        targets=targets,
        seed=seed,
    )
    return value


def evaluate(  # noqa: PLR0913
    prior_model: Any,
    expert_elicited_statistics: dict[str, tf.Tensor],
    model: dict[str, Any],
    targets: list[Target],
    seed: int,
    run: Any = None,
) -> tuple[float, dict[str, Any]]:
    """
    Simulate once from a prior model, and score the result

    The prior model is used as it is. The caller decides where its
    hyperparameter values come from: a fresh model built by ``score``, or the
    variables of a model under training.

    Parameters
    ----------
    prior_model
        Initialized prior model, ready to sample from.

    expert_elicited_statistics
        Elicited statistics of the expert.

    model
        Generative model.

    targets
        Elicitation techniques and target quantities.

    seed
        Seed used for the forward simulation.

    run
        Compiled simulation, built by
        [`compile_evaluate`][elicito.warmstart.compile_evaluate]. The eager
        path is used if it is not given.

    Returns
    -------
    value :
        Total loss against the expert-elicited statistics. A set of values
        that cannot be used is reported as a large finite number, so that a
        derivative-free search can steer away from it.

    output :
        Quantities of this simulation, in the keys that ``sgd_training`` uses
        for its results.

    """
    if run is None:
        (elicited, prior_sim, model_sim, target_quantities) = (
            el.utils.one_forward_simulation(
                prior_model=prior_model, model=model, targets=targets, seed=seed
            )
        )
        (loss, indiv_losses, loss_components_expert, loss_components_training) = (
            el.losses.total_loss(
                elicit_training=elicited,
                elicit_expert=expert_elicited_statistics,
                targets=targets,
            )
        )
    else:
        (
            elicited,
            prior_sim,
            model_sim,
            target_quantities,
            loss,
            indiv_losses,
            loss_components_expert,
            loss_components_training,
        ) = run()
    # `loss` has shape (1,). NumPy 2.5 rejects `float()` on an array
    # that is not 0-dimensional, so reduce the shape first.
    value = float(tf.squeeze(loss))
    # A quantile query hides an overflow, so the loss alone is not enough.
    # One flat failure value would give the search nothing to follow, so
    # grade the penalty by the share of draws that overflow. The search can
    # then walk out of the bad region.
    bad = el.utils.nonfinite_fraction(target_quantities)
    if bad > 0.0:
        value = PENALTY * (1.0 + bad)
    # A derivative-free search cannot use a non-finite value. Steer it away.
    elif not np.isfinite(value):
        value = PENALTY * 2.0

    output = {
        "target_quantities": target_quantities,
        "elicited_statistics": elicited,
        "prior_samples": prior_sim,
        "model_samples": model_sim,
        "loss_tensor_expert": loss_components_expert,
        "loss_tensor_model": loss_components_training,
        "loss": loss,
        "loss_component": indiv_losses,
    }
    return value, output


def compile_evaluate(
    prior_model: Any,
    model: dict[str, Any],
    targets: list[Target],
    expert_elicited_statistics: dict[str, tf.Tensor],
    seed: int,
) -> Any:
    """
    Trace the forward simulation and the loss once, and re-use the graph

    Eager execution dispatches every operation from Python. The tensors of one
    simulation are small, so this dispatch, and not the arithmetic, decides the
    runtime. A traced graph pays the dispatch once. Measured on the human
    growth model, one simulation with its loss went from 212 ms to 6.6 ms.

    The graph reads the variables of ``prior_model``, so an assignment before
    the call reaches the next simulation. ``model`` and ``targets`` are read at
    trace time. A change to either needs a new compiled function.

    Warning: the caller must not call ``tf.random.set_seed`` in front of the
    returned function. It clears the kernel caches, and the next call rebuilds
    the whole graph, which costs 90 ms. The draws repeat without it, because
    the simulation seeds every distribution itself.

    Parameters
    ----------
    prior_model
        Initialized prior model, ready to sample from.

    model
        Generative model.

    targets
        Elicitation techniques and target quantities.

    expert_elicited_statistics
        Elicited statistics of the expert.

    seed
        Seed used for the forward simulation.

    Returns
    -------
    run :
        Callable without arguments. It returns the four results of
        [`simulate_and_elicit`][elicito.utils.simulate_and_elicit], followed by
        the four results of [`total_loss`][elicito.losses.total_loss].

    """

    @tf.function(reduce_retracing=True)  # type: ignore [misc]
    def run() -> Any:
        (elicited, prior_sim, model_sim, target_quantities) = (
            el.utils.simulate_and_elicit(prior_model, model, targets, seed)
        )
        (loss, indiv_losses, loss_components_expert, loss_components_training) = (
            el.losses.total_loss(
                elicit_training=elicited,
                elicit_expert=expert_elicited_statistics,
                targets=targets,
            )
        )
        return (
            elicited,
            prior_sim,
            model_sim,
            target_quantities,
            loss,
            indiv_losses,
            loss_components_expert,
            loss_components_training,
        )

    return run


def _variable_names(variables: Any) -> list[str]:
    """
    Read the hyperparameter name of each trainable variable

    The prior model names a variable ``"<constraint>.<hyperparameter>"``.
    The order is the order in which the optimizer reads the variables.

    Parameters
    ----------
    variables
        Trainable variables of the prior model.

    Returns
    -------
    names :
        One hyperparameter name per variable.

    """
    return [str(var.name)[:-2].split(".")[1] for var in variables]


def compile_score(  # noqa: PLR0913
    expert_elicited_statistics: dict[str, tf.Tensor],
    parameters: list[Parameter],
    trainer: Trainer,
    model: dict[str, Any],
    targets: list[Target],
    expert: ExpertDict,
    seed: int,
) -> Any:
    """
    Build one prior model, and trace its simulation once

    [`score`][elicito.warmstart.score] builds a prior model for every set of
    values. That costs 10 ms per evaluation, and it gives the compiler a new
    object each time, which forces a new trace. The scorer built here keeps one
    prior model, and writes the values into its variables.

    Parameters
    ----------
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
    scorer :
        Callable that takes one value per hyperparameter name, and returns the
        loss of that point.

    """
    names = el.initialization.hyper_names(parameters)
    prior_model = el.simulations.Priors(
        ground_truth=False,
        init_matrix_slice=dict.fromkeys(names, tf.constant(0.0, dtype=tf.float32)),
        trainer=trainer,
        parameters=parameters,
        network=None,
        expert=expert,
        seed=seed,
    )
    variables = el.methods.get_method(trainer["method"]).trainable_variables(
        prior_model
    )
    var_names = _variable_names(variables)
    run = compile_evaluate(
        prior_model, model, targets, expert_elicited_statistics, seed
    )

    def scorer(hyperparams: dict[str, Any]) -> float:
        for var, name in zip(variables, var_names):
            var.assign(tf.constant(float(hyperparams[name]), dtype=var.dtype))
        (value, _) = evaluate(
            prior_model=prior_model,
            expert_elicited_statistics=expert_elicited_statistics,
            model=model,
            targets=targets,
            seed=seed,
            run=run,
        )
        return value

    return scorer


def _box_vector(box: dict[str, Any], names: list[str], key: str) -> list[float]:
    """Read one value per hyperparameter out of one entry of the box."""
    entry = box[key]
    if np.isscalar(entry):
        return [float(entry)] * len(names)  # type: ignore [arg-type]
    order = box["hyper"] if box["hyper"] is not None else names
    lookup = dict(zip(order, entry))
    return [float(lookup[name]) for name in names]


def _start_vector(box: dict[str, Any], names: list[str]) -> list[float]:
    """Read one start value per hyperparameter out of the box."""
    return _box_vector(box, names, "mean")


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

    scorer = compile_score(
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
