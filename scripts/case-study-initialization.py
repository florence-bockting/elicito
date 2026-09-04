"""
Case study: how the initialization box controls training failure

For ``method="parametric_prior"`` the user must give a start value for every
hyperparameter. Start values live on the *unconstrained* scale. The default
initialization box is centred at 0 with radius 1, whatever the scale and the
role of the hyperparameter.

This script measures what that costs. It varies two things and holds
everything else fixed:

- the initialization box, over five scenarios, from well centred to extreme;
- the prior family of the noise parameter, over HalfNormal, LogNormal and
  Weibull.

The family matters. A HalfNormal is linear in its scale, so a bad start value
only makes the fit poor. A LogNormal exponentiates its location, and a Weibull
has a shape parameter, so for those a bad start value can overflow and produce
a NAN.

It reports, per scenario, how many chains fail, what loss the surviving chains
reach, and how far the learned hyperparameters are from the ground truth.

Run it before and after a change to the initialization code, and compare the
two tables.

Usage
-----
    uv run python scripts/case-study-initialization.py
    uv run python scripts/case-study-initialization.py --family lognormal
    uv run python scripts/case-study-initialization.py --runs 4 --clipnorm 0
"""

from __future__ import annotations

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import traceback
from pathlib import Path
from typing import Any, NamedTuple

import cloudpickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import elicito as el

tfd = tfp.distributions


class Family(NamedTuple):
    """One choice of prior family for the noise parameter."""

    prior: Any
    """the oracle prior, used to simulate the expert"""

    hyperparams: dict[str, Any]
    """the hyperparameters to be learned"""

    true_hyper: dict[str, float]
    """the oracle values, on the natural (constrained) scale"""

    comment: str


# The noise parameter is the interesting one. The family decides whether a bad
# start value can produce a NAN at all.
#
# - HalfNormal is linear in its scale. A bad start value makes the fit poor,
#   but the loss stays finite. It is the benign reference.
# - LogNormal exponentiates its location. A start value a few units too large
#   makes the likelihood scale overflow, and the loss becomes NAN.
# - Weibull has a shape parameter. A small concentration makes the
#   distribution extremely heavy-tailed, so single draws overflow. Weibull also
#   has no analytic prior predictive, so no closed-form start value exists.
FAMILIES = {
    "halfnormal": Family(
        prior=tfd.HalfNormal(scale=7.0),
        hyperparams=dict(scale=el.hyper("sigma2", lower=0)),
        true_hyper={"sigma2": 7.0},
        comment="linear in its scale; the benign reference",
    ),
    "lognormal": Family(
        prior=tfd.LogNormal(loc=1.0, scale=0.3),
        hyperparams=dict(loc=el.hyper("mu2"), scale=el.hyper("sigma2", lower=0)),
        true_hyper={"mu2": 1.0, "sigma2": 0.3},
        comment="exponentiates its location; overflows",
    ),
    "weibull": Family(
        prior=tfd.Weibull(concentration=2.0, scale=5.0),
        hyperparams=dict(
            concentration=el.hyper("k2", lower=0), scale=el.hyper("lambda2", lower=0)
        ),
        true_hyper={"k2": 2.0, "lambda2": 5.0},
        comment="heavy tail at small shape; no analytic start value",
    ),
}

# The two regression coefficients are the same for every family.
BASE_TRUTH = {
    "beta0": tfd.Normal(loc=5, scale=1),
    "beta1": tfd.Normal(loc=2, scale=1),
}
BASE_HYPER = {"mu0": 5.0, "sigma0": 1.0, "mu1": 2.0, "sigma1": 1.0}


class Scenario(NamedTuple):
    """One initialization box to be measured."""

    name: str
    mean: float
    radius: float
    comment: str
    derived: bool = False


# The regression targets on the unconstrained scale are
# mu0=5.0, sigma0=0.54, mu1=2.0, sigma1=0.54, because softplus^-1(1) = 0.54.
SCENARIOS = [
    Scenario("centred", 3.0, 5.0, "box contains the targets"),
    Scenario("default", 0.0, 1.0, "current default; contains few"),
    Scenario("wide", 0.0, 10.0, "contains all, but is very wide"),
    Scenario("far", -5.0, 2.0, "contains none, and is on the wrong side"),
    Scenario("extreme", -20.0, 2.0, "all scales collapse to zero"),
    Scenario("from_elicits", 0.0, 0.0, "box derived from the expert data", True),
]


def X_design(N: int, quantiles: list[float]) -> np.ndarray:
    """
    Compute design matrix

    Parameters
    ----------
    N
        number of observations

    quantiles
        list of quantiles

    Returns
    -------
    :
        design matrix
    """
    X = tf.cast(np.arange(N), tf.float32)
    X_std = X / tf.math.reduce_std(X)
    X_sel = tfp.stats.percentile(X_std, quantiles)
    X_design = tf.stack([tf.ones(X_sel.shape), X_sel], -1)
    return X_design


class ToyModel:
    """
    generative model
    """

    def __call__(self, prior_samples: Any, design_matrix: Any) -> dict[str, Any]:
        """
        Compute target quantities from generative model

        Parameters
        ----------
        prior_samples
            prior samples

        design_matrix
            design matrix

        Returns
        -------
        :
            dictionary with target quantities
        """
        # linear predictor
        epred = tf.matmul(prior_samples[:, :, :-1], design_matrix, transpose_b=True)

        # data-generating model
        likelihood = tfd.Normal(
            loc=epred, scale=tf.expand_dims(prior_samples[:, :, -1], -1)
        )
        # prior predictive distribution
        ypred = likelihood.sample()

        # selected observations
        y_X0, y_X1, y_X2 = (ypred[:, :, 0], ypred[:, :, 1], ypred[:, :, 2])

        return dict(y_X0=y_X0, y_X1=y_X1, y_X2=y_X2, ypred=ypred, epred=epred)


def custom_r2(ypred: Any, epred: Any) -> Any:
    """
    Compute coefficient of determination

    Parameters
    ----------
    ypred
        samples from the prior predictive distribution

    epred
        samples from the linear predictor

    Returns
    -------
    :
        coefficient of determination
    """
    var_epred = tf.math.reduce_variance(epred, -1)
    # variance of difference between ypred and epred
    var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
    var_total = var_epred + var_diff
    # variance of linear predictor divided by total variance
    return tf.divide(var_epred, var_total)


def build_eliobj(  # noqa: PLR0913
    family: Family,
    mean: float,
    radius: float,
    epochs: int,
    clipnorm: float | None = 1.0,
    seed: int = 2025,
    derived: bool = False,
    warmstart: bool = False,
) -> el.Elicit:
    """
    Build the elicitation object for one initialization box

    Everything except the initialization box is held fixed, so a difference
    between two scenarios can only come from the box.

    Parameters
    ----------
    family
        prior family used for the noise parameter

    mean
        centre of the uniform initialization box, on the unconstrained scale

    radius
        half-width of the uniform initialization box

    epochs
        number of training epochs

    clipnorm
        gradient clipping passed to the Adam optimizer. ``None`` disables it.
        Clipping hides the divergence this script tries to measure.

    seed
        seed passed to the trainer

    derived
        build the box from the expert data with
        [`from_elicits`][elicito.initialization.from_elicits]. ``mean`` and
        ``radius`` are then ignored.

    warmstart
        search the start value with Nelder-Mead instead of drawing candidates
        from the box. The box then only provides the start point.

    Returns
    -------
    :
        the unfitted elicitation object
    """
    adam_kwargs: dict[str, Any] = dict(learning_rate=0.1)
    if clipnorm is not None:
        adam_kwargs["clipnorm"] = clipnorm

    parameters = [
        el.parameter(
            name="beta0",
            family=tfd.Normal,
            hyperparams=dict(loc=el.hyper("mu0"), scale=el.hyper("sigma0", lower=0)),
        ),
        el.parameter(
            name="beta1",
            family=tfd.Normal,
            hyperparams=dict(loc=el.hyper("mu1"), scale=el.hyper("sigma1", lower=0)),
        ),
        el.parameter(
            name="sigma",
            family=type(family.prior),
            hyperparams=family.hyperparams,
        ),
    ]

    targets = [
        el.target(
            name="y_X0",
            query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0,
        ),
        el.target(
            name="y_X1",
            query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0,
        ),
        el.target(
            name="y_X2",
            query=el.queries.quantiles((0.05, 0.25, 0.50, 0.75, 0.95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0,
        ),
        el.target(
            name="R2",
            query=el.queries.quantiles((0.05, 0.50, 0.95)),
            loss=el.losses.L2,
            weight=0.5,
            target_method=custom_r2,
        ),
    ]

    return el.Elicit(
        model=el.model(
            obj=ToyModel, design_matrix=X_design(N=30, quantiles=[25, 50, 75])
        ),
        parameters=parameters,
        targets=targets,
        expert=el.expert.simulator(
            ground_truth={**BASE_TRUTH, "sigma": family.prior}, num_samples=10_000
        ),
        optimizer=el.optimizer(optimizer=tf.keras.optimizers.Adam, **adam_kwargs),
        trainer=el.trainer(
            method="parametric_prior", seed=seed, epochs=epochs, progress=0
        ),
        initializer=el.initializer(
            method="warmstart" if warmstart else "sobol",
            iterations=300 if warmstart else 32,
            distribution=(
                el.initialization.from_elicits()
                if derived
                else el.initialization.uniform(radius=radius, mean=mean)
            ),
        ),
    )


class Report(NamedTuple):
    """The measurements for one scenario."""

    scenario: Scenario
    n_runs: int
    n_failed: int
    final_loss: float | None
    hyper_error: float | None
    crash: str | None


def picklable(obj: Any) -> bool:
    """
    Report whether joblib can send this object to a worker process

    A ``tfd.Weibull`` instance cannot be pickled: it holds an ``Invert``
    bijector that fails to deserialize. Such a family must be fitted
    sequentially.

    Parameters
    ----------
    obj
        the object to test

    Returns
    -------
    :
        ``True`` when the object survives a pickle round trip
    """
    try:
        cloudpickle.loads(cloudpickle.dumps(obj))
    except Exception:
        return False
    return True


def chain_records(eliobj: el.Elicit, n_reps: int, family: Family) -> list[Any]:
    """
    Extract one record per surviving chain

    Parameters
    ----------
    eliobj
        a fitted elicitation object

    n_reps
        number of chains this object was fitted with

    family
        prior family used for the noise parameter

    Returns
    -------
    :
        one ``(final_loss, mean relative error)`` pair per surviving chain
    """
    # _check_NaN counts a chain as failed when it ran fewer epochs than
    # requested. It is private, which is acceptable in a script but not in a
    # test.
    _, success, _ = el.plots._check_NaN(eliobj, n_reps)
    if len(success) == 0:
        return []

    loss = eliobj.results.history_stats.loss.total_loss
    hyp = eliobj.results.history_stats.hyperparameter
    true_hyper = {**BASE_HYPER, **family.true_hyper}

    records = []
    for i in success:
        final = float(loss.isel(epoch=-1, replication=i))
        errors = [
            abs(float(hyp[name].isel(epoch=-1, replication=i)) - true) / true
            for name, true in true_hyper.items()
        ]
        records.append((final, float(np.mean(errors))))
    return records


def run_scenario(  # noqa: PLR0913
    family: Family,
    scenario: Scenario,
    n_runs: int,
    epochs: int,
    clipnorm: float | None,
    sequential: bool,
    warmstart: bool,
) -> Report:
    """
    Fit one scenario and measure it

    A scenario is expected to fail. The failure is the thing being measured,
    so a crash must not end the sweep.

    Parameters
    ----------
    family
        prior family used for the noise parameter

    scenario
        the initialization box to measure

    n_runs
        number of chains to run

    epochs
        number of training epochs

    clipnorm
        gradient clipping passed to the Adam optimizer. ``None`` disables it.

    warmstart
        search the start value with Nelder-Mead instead of drawing candidates
        from the box

    sequential
        fit one chain at a time, instead of using joblib. Required for a
        family whose oracle prior cannot be pickled.

    Returns
    -------
    :
        the measurements for this scenario
    """
    print(f"--- {scenario.name}: mean={scenario.mean}, radius={scenario.radius}")

    records: list[Any] = []
    crash = None
    # A chain that stops early leaves a ragged history, which breaks the
    # results tree. Record the exception instead of losing the whole sweep.
    if sequential:
        for i in range(n_runs):
            eliobj = build_eliobj(
                family,
                scenario.mean,
                scenario.radius,
                epochs,
                clipnorm,
                seed=2025 + i,
                derived=scenario.derived,
                warmstart=warmstart,
            )
            try:
                eliobj.fit()
            except Exception:
                crash = traceback.format_exc().strip().splitlines()[-1]
                print(f"    chain {i} crashed: {crash}")
                continue
            records += chain_records(eliobj, 1, family)
    else:
        eliobj = build_eliobj(
            family,
            scenario.mean,
            scenario.radius,
            epochs,
            clipnorm,
            derived=scenario.derived,
            warmstart=warmstart,
        )
        try:
            eliobj.fit(parallel=el.utils.parallel(runs=n_runs))
        except Exception:
            crash = traceback.format_exc().strip().splitlines()[-1]
            print(f"    crashed: {crash}")
            return Report(scenario, n_runs, n_runs, None, None, crash)
        records = chain_records(eliobj, n_runs, family)

    if not records:
        return Report(scenario, n_runs, n_runs, None, None, crash)
    return Report(
        scenario,
        n_runs,
        n_runs - len(records),
        float(np.mean([r[0] for r in records])),
        float(np.mean([r[1] for r in records])),
        crash,
    )


def to_markdown(reports: list[Report]) -> str:
    """
    Format the reports as a markdown table

    Parameters
    ----------
    reports
        one report per scenario

    Returns
    -------
    :
        the table, as a markdown string
    """
    lines = [
        "| scenario | mean | radius | failed | final loss | hyper. error | note |",
        "| :-- | --: | --: | --: | --: | --: | :-- |",
    ]
    for r in reports:
        loss = "-" if r.final_loss is None else f"{r.final_loss:.4f}"
        err = "-" if r.hyper_error is None else f"{r.hyper_error:.3f}"
        note = r.crash if r.crash else r.scenario.comment
        lines.append(
            f"| {r.scenario.name} | {r.scenario.mean} | {r.scenario.radius} "
            f"| {r.n_failed}/{r.n_runs} | {loss} | {err} | {note} |"
        )
    return "\n".join(lines)


def main(  # noqa: PLR0913
    runs: int,
    epochs: int,
    out: Path,
    clipnorm: float | None,
    families: list[str],
    warmstart: bool,
) -> None:
    """
    Fit every scenario for every family and write the comparison tables

    Parameters
    ----------
    runs
        number of chains per scenario

    epochs
        number of training epochs per chain

    out
        path of the markdown file to write

    clipnorm
        gradient clipping passed to the Adam optimizer. ``None`` disables it.

    families
        names of the noise families to sweep

    warmstart
        search the start value with Nelder-Mead instead of drawing candidates
        from the box
    """
    sections = []
    for name in families:
        family = FAMILIES[name]
        print(f"=== family: {name} ({family.comment})")
        # A family whose oracle prior cannot be pickled must be fitted one
        # chain at a time, because joblib cannot send it to a worker.
        sequential = not picklable(family.prior)
        if sequential:
            print("    oracle prior does not pickle; fitting sequentially")
        reports = [
            run_scenario(family, s, runs, epochs, clipnorm, sequential, warmstart)
            for s in SCENARIOS
        ]
        table = to_markdown(reports)
        print()
        print(table)
        print()
        note = f"{family.comment}."
        if sequential:
            note += " Fitted sequentially: the oracle prior does not pickle."
        sections.append(f"## {name}\n\n{note}\n\n{table}\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Initialization case study\n\n"
        f"`runs={runs}`, `epochs={epochs}`, "
        f"`iterations={300 if warmstart else 32}`, "
        f"`method={'warmstart' if warmstart else 'sobol'}`, "
        f"`clipnorm={clipnorm}`.\n\n"
        "The hyperparameter error is the mean relative error against the\n"
        "ground truth, over every hyperparameter and the surviving chains.\n\n"
    )
    out.write_text(header + "\n".join(sections))
    print(f"written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=int, default=4, help="Number of chains per scenario."
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="Training epochs per chain."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "case-study-initialization.md",
        help="Where to write the markdown table.",
    )
    parser.add_argument(
        "--clipnorm",
        type=float,
        default=1.0,
        help="Gradient clipping for Adam. Use 0 to disable it.",
    )
    parser.add_argument(
        "--warmstart",
        action="store_true",
        help="Search the start value with Nelder-Mead in every scenario.",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=sorted(FAMILIES),
        help="Noise family to sweep. Repeatable. Defaults to all of them.",
    )
    args = parser.parse_args()
    main(
        args.runs,
        args.epochs,
        args.out,
        args.clipnorm or None,
        args.family or sorted(FAMILIES),
        args.warmstart,
    )
