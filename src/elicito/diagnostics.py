"""
Diagnostics for expert information that is not consistent
"""

from typing import Any

import numpy as np
import xarray as xr

from elicito.elicit import Elicit
from elicito.exceptions import MissingOptionalDependencyError
from elicito.optimizers.cmaes import CMAES
from elicito.utils import parallel


def _refit(eliobj: Elicit, targets: list[Any], seeds: list[int]) -> Elicit:
    """Refit with new targets and the settings of the original fit"""
    # eliobj stores the default initializer of CMA-ES, which CMA-ES rejects
    cmaes = eliobj.optimizer["optimizer"] == CMAES
    refit = Elicit(
        model=eliobj.model,
        parameters=eliobj.parameters,
        targets=targets,
        expert=eliobj.expert,
        trainer=eliobj.trainer,
        optimizer=eliobj.optimizer,
        network=eliobj.network,
        initializer=None if cmaes else eliobj.initializer,
        meta_settings=eliobj.meta_settings,
    )
    refit.fit(parallel=parallel(runs=len(seeds), seeds=seeds))
    return refit


def _final_losses(eliobj: Elicit) -> xr.Dataset:
    """Return the loss of each statistic in the last epoch"""
    return eliobj.results["history_stats/loss"].to_dataset().isel(epoch=-1)


def baseline(
    eliobj: Elicit,
    seeds: tuple[int, ...] = (1, 2, 3),
) -> xr.Dataset:
    """
    Compute the reference loss of each statistic

    The refits use the expert data and the settings of ``eliobj``.
    :func:`leave_one_out` refits with the same seeds and compares the loss
    seed by seed.

    Parameters
    ----------
    eliobj
        fitted eliobj with ``method="parametric_prior"``

    seeds
        one refit for each seed; at least two seeds

    Returns
    -------
    :
        the loss of each statistic in the last epoch, for each seed, and
        ``distance``, the in-fit distance of each statistic to its expert
        values, in units of the expert interval. ``attrs["seeds"]`` lists
        the seeds.

    Raises
    ------
    ValueError
        fewer than two seeds
    """
    if len(seeds) < 2:  # noqa: PLR2004
        raise ValueError(f"baseline needs at least two seeds. Got {seeds=}.")  # noqa: TRY003

    refit = _refit(eliobj, eliobj.targets, list(seeds))
    losses = _final_losses(refit)
    statistics = _statistics(_components(losses))
    expert = _expert_statistics(eliobj)
    distance = [_distance(refit, name, expert) for name in statistics]
    reference = losses.assign(distance=("statistic", distance), statistic=statistics)
    reference.attrs["seeds"] = list(seeds)
    return reference


def _components(losses: xr.Dataset) -> list[str]:
    """Return the names of the loss components, in the order of the targets"""
    names = [str(name) for name in losses.data_vars if "_loss_" in str(name)]
    return sorted(names, key=lambda name: int(name.rsplit("_loss_", 1)[1]))


def _statistics(names: list[str]) -> list[str]:
    """Return the name of the statistic of each loss component"""
    return [name.rsplit("_loss_", 1)[0] for name in names]


def _expert_statistics(eliobj: Elicit) -> dict[str, Any]:
    """Return the expert statistics, from expert data or from an oracle"""
    group = "expert" if "expert" in eliobj.results.children else "oracle"
    statistics = eliobj.results[group].to_dataset().isel(replication=0, batch=0)
    return {str(name): statistics[name].values for name in statistics.data_vars}


def _predicted(refit: Elicit, name: str) -> Any:
    """Return the mean of a statistic over the batches, one row for each seed"""
    simulated = refit.sample()["elicited_summary"].to_dataset()[name].values
    return simulated.mean(axis=1)


def _distance(refit: Elicit, name: str, data: dict[str, Any]) -> float:
    """Return the largest distance to the expert values, in expert intervals"""
    predicted = _predicted(refit, name).mean(axis=0)
    expert = np.asarray(data[name], dtype=float)
    width = expert.max() - expert.min()
    if width == 0:
        return float("nan")
    return float(np.max(np.abs(predicted - expert)) / width)


def _miss(refit: Elicit, name: str, data: dict[str, Any]) -> Any:
    """Return the largest distance to the expert values, one for each seed"""
    expert = np.asarray(data[name], dtype=float)
    return np.abs(_predicted(refit, name) - expert).max(axis=1)


def _relative_drop(full: Any, loo: Any) -> Any:
    """
    Return the drop of the loss for each seed, as a symmetric % change

    ``full`` and ``loo`` hold the loss of the other statistics for each seed,
    in the full fit and in the fit without one statistic. The drop is divided
    by the mean of ``full`` and ``loo`` of the same seed, so it lies in
    [-200, 200] even when ``full`` is near 0.
    """
    full = np.asarray(full, dtype=float)
    loo = np.asarray(loo, dtype=float)
    return 200 * (full - loo) / (full + loo)


def _t_interval(values: Any) -> tuple[float, float, float]:
    """
    Return the mean of paired differences and its 95 % t-interval

    The values are paired differences, one for each seed, so the interval
    has n - 1 degrees of freedom.
    """
    try:
        from scipy.stats import t
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "leave_one_out", requirement="scipy"
        ) from exc

    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    half = t.ppf(0.975, df=n - 1) * values.std(ddof=1) / np.sqrt(n)
    return float(mean), float(mean - half), float(mean + half)


def _leading(values: Any, threshold: float) -> list[bool]:
    """
    Return which statistics conflict most with the others

    ``values`` has one row for each left-out statistic and one column for
    each seed. The statistic with the largest mean leads. Another statistic
    also leads if its own interval is above ``threshold`` and its difference
    to the largest mean is not clear: the paired interval of the difference
    contains 0. No statistic leads if no interval is above ``threshold``.
    """
    values = np.asarray(values, dtype=float)
    lower = np.array([_t_interval(row)[1] for row in values])
    if not (lower > threshold).any():
        return [False] * len(values)
    top = int(np.argmax(values.mean(axis=1)))
    return [
        k == top
        or (lower[k] > threshold and _t_interval(values[top] - values[k])[1] <= 0)
        for k in range(len(values))
    ]


def leave_one_out(
    eliobj: Elicit,
    reference: xr.Dataset,
    tolerance: dict[str, float],
) -> xr.Dataset:
    """
    Refit without one statistic at a time

    Each refit uses the seeds of ``reference`` and predicts the left-out
    statistic. The miss is the largest distance of this prediction to the
    expert values, in units of ``tolerance``. A miss above 1 is more than the
    precision of the expert explains.

    Parameters
    ----------
    eliobj
        fitted eliobj with ``method="parametric_prior"``

    reference
        result of :func:`baseline` for ``eliobj``

    tolerance
        for each statistic, the largest distance to the expert values that
        the precision of the expert explains, in the units of the statistic.
        The miss of a consistent statement is not 0, because the Monte Carlo
        sample and the fit add error. A tolerance below this error flags
        consistent statements.

    Returns
    -------
    :
        for each left-out statistic: ``miss``, for each seed; ``mean_miss``,
        its mean; ``lower`` and ``upper``, the bounds of its 95 % interval;
        ``leading``, the statistics that conflict most with the others;
        ``drop`` and ``gain``, the drop in the loss of the other statistics,
        for each seed and its mean, in %. ``attrs["inconsistent"]`` is true
        if an interval is above 1.

    Raises
    ------
    ValueError
        ``tolerance`` has no positive value for a statistic
    """
    seeds = list(reference.attrs["seeds"])
    names = _components(reference)
    statistics = _statistics(names)
    invalid = [name for name in statistics if not tolerance.get(name, 0) > 0]
    if invalid:
        raise ValueError(f"tolerance needs a positive value for {invalid}.")  # noqa: TRY003
    drops, misses = [], []
    expert = _expert_statistics(eliobj)
    for k, name in enumerate(names):
        targets = [dict(target) for target in eliobj.targets]
        targets[k]["weight"] = 0.0
        refit = _refit(eliobj, targets, seeds)
        after = _final_losses(refit)
        others = [other for other in names if other != name]
        full = sum(reference[other].values for other in others)
        loo = sum(after[other].values for other in others)
        drops.append(_relative_drop(full, loo))
        miss = _miss(refit, statistics[k], expert)
        misses.append(miss / tolerance[statistics[k]])
    mean, lower, upper = (list(values) for values in zip(*map(_t_interval, misses)))
    return xr.Dataset(
        {
            "miss": (("statistic", "seed"), np.array(misses)),
            "mean_miss": ("statistic", mean),
            "lower": ("statistic", lower),
            "upper": ("statistic", upper),
            "leading": ("statistic", _leading(misses, threshold=1.0)),
            "drop": (("statistic", "seed"), np.array(drops)),
            "gain": ("statistic", np.array(drops).mean(axis=1)),
        },
        coords={"statistic": statistics, "seed": seeds},
        attrs={"inconsistent": bool(max(lower) > 1)},
    )
