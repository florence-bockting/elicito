"""
Global search for the hyperparameters of a parametric prior, with CMA-ES
"""

import logging
import time
from typing import Any

import numpy as np
import tensorflow as tf

import elicito as el
from elicito._progress import ProgressTable
from elicito.exceptions import MissingOptionalDependencyError
from elicito.types import ExpertDict, Parameter, Target, Trainer

logger = logging.getLogger(__name__)

# Step size of the first generation, as a share of the box radius. CMA-ES
# adapts the step size afterwards. Half the radius covers the box, and keeps
# most of the first generation inside it.
SIGMA_FRACTION = 2.0

# First step size of `cma_training`, on the unconstrained scale. The fitter
# has no box to read a scale from, so the value is a setting of the optimizer.
DEFAULT_SIGMA0 = 0.5

# Value of `el.optimizer(optimizer=...)` that selects the CMA-ES fitter
CMAES = "cmaes"


def cma_search(  # noqa: PLR0913
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
    Search a start value with CMA-ES, before any gradient step

    The search reads the whole initialization box, not only its centre. It
    starts at the centre, with a step size of half the radius, and it keeps
    every candidate inside the box. It needs no gradient, so it cannot diverge
    through an exploding gradient.

    Use this method when you cannot place the box around the answer. A wide
    box gives the two local methods, ``warmstart`` and the box samplers, the
    first basin they meet, which is not the best one. Measured on the human
    growth model, from a box of radius 5 around zero, the warm start ended at
    a loss of 482.9 and CMA-ES at 27.7, in about the same number of
    evaluations.

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
        Initialization box. Its centre is the start point of the search, and
        its radius sets both the step size and the bounds.

    max_evals
        Budget, in objective evaluations. The search stops at the end of the
        generation that reaches the budget, so it can use a few more.

    seed
        Seed used for the forward simulation.

    Raises
    ------
    MissingOptionalDependencyError
        ``cma`` is required for the search.

    Returns
    -------
    hyperparams :
        One value per hyperparameter, on the unconstrained scale.

    """
    try:
        import cma  # type: ignore [import-untyped]
    except ImportError as exc:
        raise MissingOptionalDependencyError("cma_search", requirement="cma") from exc

    names = el.initialization.hyper_names(parameters)
    box = el.initialization.build_box(
        distribution, expert_elicited_statistics, parameters
    )

    search_trainer = dict(trainer)
    search_trainer["num_samples"] = max(
        el.warmstart.MIN_SEARCH_SAMPLES,
        trainer["num_samples"] // el.warmstart.SEARCH_FRACTION,
    )

    centre = np.asarray(el.warmstart._box_vector(box, names, "mean"), dtype=np.float64)
    radius = np.asarray(
        el.warmstart._box_vector(box, names, "radius"), dtype=np.float64
    )

    # CMA-ES uses one step size for every coordinate. A box with a different
    # radius per hyperparameter is handled by `CMA_stds`, which rescales each
    # coordinate. The step size of coordinate i is then sigma0 * radius[i].
    options = {
        "CMA_stds": radius,
        "bounds": [centre - radius, centre + radius],
        "maxfevals": max_evals,
        # cma reads a seed of 0 as "draw a random seed"
        "seed": int(seed) + 1,
        "verbose": -9,
        "verb_log": 0,
    }

    # A failed point is scored as a large finite number, so the last generation
    # can be worse than a point seen before. Keep the best usable point.
    best: dict[str, Any] = {"value": el.warmstart.PENALTY, "values": None}

    scorer = el.warmstart.compile_score(
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

    strategy = cma.CMAEvolutionStrategy(centre, 1.0 / SIGMA_FRACTION, options)
    used = 0
    while used < max_evals and not strategy.stop():
        candidates = strategy.ask()
        losses = [objective(candidate) for candidate in candidates]
        used += len(candidates)
        strategy.tell(candidates, losses)

    if best["values"] is None:
        logger.warning(
            "CMA-ES: every evaluated point failed. The centre of the "
            "initialization box is used as the start value. Re-centre the "
            "box, or reduce its radius."
        )
        values = centre
    else:
        logger.info(f"CMA-ES: loss {best['value']:.4f} after {used} evaluations.")
        values = best["values"]

    return {name: float(value) for name, value in zip(names, values)}


def box_step_size(
    initializer: Any,
    expert_elicited_statistics: dict[str, tf.Tensor],
    parameters: list[Parameter],
) -> Any:
    """
    Read the first step size of the training out of the initialization box

    The box states the scale of every hyperparameter. A coordinate with a
    wide radius needs a wide first step. The box is only a scale and a start
    point here: it does not bound the search.

    Parameters
    ----------
    initializer
        Specification of the initialization method.

    expert_elicited_statistics
        Elicited statistics of the expert. The default box reads their scale.

    parameters
        List including dictionary with all information about the
        (hyper-)parameters.

    Returns
    -------
    sigma0 :
        Step size per hyperparameter name, or ``DEFAULT_SIGMA0`` if the
        initialization method does not hand its box to the training.

    """
    method = el.initialization.resolve_init_method(initializer)
    if not method.skips_search(dict(optimizer=CMAES)):
        return DEFAULT_SIGMA0

    names = el.initialization.hyper_names(parameters)
    box = el.initialization.build_box(
        dict(initializer["distribution"]), expert_elicited_statistics, parameters
    )
    radius = el.warmstart._box_vector(box, names, "radius")
    return {name: value / SIGMA_FRACTION for name, value in zip(names, radius)}


def _step_size(
    sigma0: Any, names: list[str], default: Any = DEFAULT_SIGMA0
) -> tuple[float, list[float] | None]:
    """
    Split the ``sigma0`` setting into a scalar and a per-coordinate vector

    CMA-ES uses one step size for every coordinate. A different step size per
    hyperparameter is handled by ``CMA_stds``, which rescales each coordinate.
    The step size of coordinate i is then ``sigma0 * CMA_stds[i]``, so a
    per-hyperparameter setting keeps the scalar at 1.0.

    Parameters
    ----------
    sigma0
        One step size for every coordinate, or a step size per hyperparameter
        name. A name that is not given gets its step size from ``default``.

    names
        Hyperparameter name of each trainable variable.

    default
        Step size of a hyperparameter that ``sigma0`` does not name. One
        number, or one number per hyperparameter name.

    Raises
    ------
    ValueError
        A key of ``sigma0`` is not a hyperparameter of the model.

    Returns
    -------
    sigma0 :
        First step size, as one number.

    stds :
        One multiplier per coordinate, or None if ``sigma0`` is a number.

    """
    if not isinstance(sigma0, dict):
        return float(sigma0), None

    unknown = sorted(set(sigma0) - set(names))
    if unknown:
        msg = (
            f"optimizer(sigma0=...) has the unknown hyperparameter(s) "
            f"{unknown}. The hyperparameters of this model are "
            f"{sorted(set(names))}."
        )
        raise ValueError(msg)

    if not isinstance(default, dict):
        default = dict.fromkeys(names, default)
    stds = [
        float(sigma0.get(name, default.get(name, DEFAULT_SIGMA0))) for name in names
    ]
    return 1.0, stds


def cma_training(  # noqa: PLR0913, PLR0915
    expert_elicited_statistics: dict[str, tf.Tensor],
    prior_model_init: Any,
    trainer: Trainer,
    optimizer: dict[str, Any],
    model: dict[str, Any],
    targets: list[Target],
    parameters: list[Parameter],
    seed: int,
    progress: int,
    default_sigma0: Any = DEFAULT_SIGMA0,
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """
    Fit the hyperparameters of a parametric prior with CMA-ES

    The search replaces the gradient descent of
    [`sgd_training`][elicito.optimization.sgd_training]. It needs no gradient,
    so it cannot diverge through an exploding gradient, and it can leave a
    local basin that a gradient step cannot leave. It costs more forward
    simulations for the same number of history points, because one generation
    needs one simulation per candidate.

    The search starts at the values that the initialization produced. It is
    not bounded. Every candidate is scored on the same seed, so two candidates
    differ only in their hyperparameters.

    The two return values have the same structure as those of
    ``sgd_training``, with one generation in the place of one epoch. The
    gradient history is empty.

    Parameters
    ----------
    expert_elicited_statistics
        Elicited statistics of the expert.

    prior_model_init
        Initialized prior model. Its variables hold the start value.

    trainer
        Settings for the optimization phase. ``epochs`` is read as the budget
        in forward simulations.

    optimizer
        Settings of the search. ``sigma0`` is the first step size, on the
        unconstrained scale. ``popsize`` is the number of candidates in one
        generation; the CMA-ES rule is used if it is not given.

    model
        Generative model.

    targets
        List of target quantities.

    parameters
        List of model parameters.

    seed
        Internally used seed for reproducible results.

    progress
        Whether the progress of the training is printed.

    default_sigma0
        First step size, used if ``optimizer`` does not give ``sigma0``. See
        [`box_step_size`][elicito.cmaes.box_step_size].

    Raises
    ------
    MissingOptionalDependencyError
        ``cma`` is required for the search.

    Returns
    -------
    res_ep :
        Results saved for each generation (history).

    output_res :
        Results of the best point found (results).

    """
    try:
        import cma
    except ImportError as exc:
        raise MissingOptionalDependencyError("cma_training", requirement="cma") from exc

    tf.random.set_seed(seed)

    prior_model = prior_model_init
    method = el.methods.get_method(trainer["method"])
    res_dict = method.new_history(prior_model, parameters)
    # the same objects during the whole run, so an assignment reaches the
    # prior model
    variables = method.trainable_variables(prior_model)

    start = [float(var.numpy()) for var in variables]
    budget = int(trainer["epochs"])

    options: dict[str, Any] = {
        "maxfevals": budget,
        # cma reads a seed of 0 as "draw a random seed"
        "seed": int(seed) + 1,
        "verbose": -9,
        "verb_log": 0,
    }
    if optimizer.get("popsize") is not None:
        options["popsize"] = int(optimizer["popsize"])

    # the box sets the step size of a hyperparameter that `sigma0` does not
    # name, so a partial `sigma0` overrides the box one coordinate at a time
    sigma0, stds = _step_size(
        optimizer.get("sigma0", default_sigma0),
        el.warmstart._variable_names(variables),
        default_sigma0,
    )
    if stds is not None:
        options["CMA_stds"] = stds

    def assign(values: Any) -> None:
        for var, value in zip(variables, values):
            var.assign(tf.constant(float(value), dtype=var.dtype))

    # traced once, then re-used. The graph reads the variables, so `assign`
    # reaches the next simulation.
    run = el.warmstart.compile_evaluate(
        prior_model, model, targets, expert_elicited_statistics, seed
    )

    kappa = trainer.get("kappa", 0.0)

    def objective(values: Any) -> tuple[float, dict[str, Any]]:
        assign(values)
        value, output = el.warmstart.evaluate(
            prior_model=prior_model,
            expert_elicited_statistics=expert_elicited_statistics,
            model=model,
            targets=targets,
            seed=seed,
            run=run,
        )
        # the penalty enters the score of the search, not the recorded loss.
        # A failed point already carries the PENALTY sentinel, and must keep
        # its order against the usable points.
        if kappa and value < el.warmstart.PENALTY:
            value += kappa * float(el.losses.spread_penalty(output["prior_samples"]))
        return value, output

    total_losses = []
    component_losses = []
    penalties = []
    time_per_epoch = []

    # A failed point is scored as a large finite number, so a later point can
    # be worse than a point seen before. Keep the best usable point.
    best: dict[str, Any] = {"value": el.warmstart.PENALTY, "values": None}

    bar = ProgressTable(
        "Training",
        total=budget,
        disable=progress == 0,
        loss=float("nan"),
        best=float("nan"),
    )

    strategy = cma.CMAEvolutionStrategy(start, sigma0, options)
    used = 0
    while used < budget and not strategy.stop():
        generation_time_start = time.time()

        candidates = strategy.ask()
        losses = []
        leader: dict[str, Any] = {"value": None, "values": None, "output": None}
        for candidate in candidates:
            value, output = objective(candidate)
            losses.append(value)
            if leader["value"] is None or value < leader["value"]:
                leader = {"value": value, "values": candidate, "output": output}
            if value < best["value"]:
                best = {
                    "value": value,
                    "values": np.array(candidate, dtype=np.float64),
                    "output": output,
                }
        used += len(candidates)
        strategy.tell(candidates, losses)

        # the history point is the best candidate of this generation. Its
        # loss and its hyperparameter values then describe the same point.
        assign(leader["values"])
        method.record_epoch(
            res_dict, leader["output"]["prior_samples"], variables, parameters
        )
        # the history keeps the score of the search, not the raw loss. A
        # point whose forward simulation overflows carries a small raw loss,
        # and would else read as progress.
        total_losses.append(tf.cast(leader["value"], leader["output"]["loss"].dtype))
        component_losses.append(leader["output"]["loss_component"])
        penalties.append(el.losses.spread_penalty(leader["output"]["prior_samples"]))
        time_per_epoch.append(time.time() - generation_time_start)

        bar.update(
            advance=len(candidates),
            loss=float(leader["value"]),
            best=float(best["value"]),
        )

    bar.close()

    if best["values"] is None:
        logger.warning(
            "CMA-ES: every evaluated point failed. The start value is kept."
            " Re-check the initialization, or the generative model."
        )
        final = start
    else:
        logger.info(f"CMA-ES: loss {best['value']:.4f} after {used} evaluations.")
        final = list(best["values"])

    # the results must describe the point that the run returns
    (_, output) = objective(final)

    res_ep = {
        "loss": total_losses,
        "loss_component": component_losses,
        "penalty": penalties,
        "time": time_per_epoch,
        "hyperparameter": res_dict,
    }
    output_res = {
        key: output[key] for key in output if key not in ("loss", "loss_component")
    }

    method.finalize(res_ep, output_res, [], variables)

    return res_ep, output_res
