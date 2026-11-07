"""
Regression test for the text output of `Fit.summary()`

The expected output is stored in ``snapshots/``. After an intended change
of the layout, rewrite the files with

    ELICITO_UPDATE_SNAPSHOTS=1 uv run pytest tests/regression

and review the diff before you commit it.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from elicito.fit import _format

SNAPSHOTS = Path(__file__).parent / "snapshots"
UPDATE = os.environ.get("ELICITO_UPDATE_SNAPSHOTS") == "1"


def summary_tree(reps: list[int], n_reps: int) -> xr.DataTree:
    """Build a summary tree by hand, with the layout of `Fit.summary()`"""
    hyper = np.array([[0.315, 0.952], [0.0234, 0.681]])[:n_reps]
    hyper_da = xr.DataArray(
        hyper,
        dims=("replication", "hyperparameter"),
        coords=dict(replication=range(n_reps), hyperparameter=["mu0", "sigma0"]),
    )
    hyperparameters = xr.Dataset(dict(estimate=hyper_da.isel(replication=reps)))
    if n_reps > 1:
        hyperparameters["rep_sd"] = hyper_da.std("replication", ddof=1)
        hyperparameters["min"] = hyper_da.min("replication")
        hyperparameters["max"] = hyper_da.max("replication")

    def per_rep(values: list[list[float]], dim: str, labels: list[str]) -> xr.DataArray:
        return xr.DataArray(
            np.array(values)[reps],
            dims=("replication", dim),
            coords={"replication": reps, dim: labels},
        )

    params = ["b0", "sigma"]
    priors = xr.Dataset(
        dict(
            mean=per_rep([[0.309, 0.81], [0.0656, 0.52]], "parameter", params).T,
            sd=per_rep([[0.465, 0.52], [0.87, 0.31]], "parameter", params).T,
            q05=per_rep([[-0.496, 0.179], [-1.32, 0.05]], "parameter", params).T,
            q95=per_rep([[1.07, 1.82], [1.66, 1.1]], "parameter", params).T,
        )
    )
    stats = ["y0[0]", "y0[1]", "cor[0]"]
    expert = xr.DataArray(
        [-1.0, 0.0, 0.01], dims="statistic", coords=dict(statistic=stats)
    )
    elicited = per_rep(
        [[-0.916, 0.113, -0.002], [-0.326, 0.694, 0.004]], "statistic", stats
    )
    expert_fit = xr.Dataset(
        dict(expert=expert, elicited=elicited, error=elicited - expert)
    )
    loss = xr.Dataset(
        dict(
            final_loss=xr.DataArray(
                [2.45, 5.16][:n_reps],
                dims="replication",
                coords=dict(replication=range(n_reps)),
            )
        )
    )
    terms = ["y0", "cor", "y2"]
    target_loss = xr.Dataset(
        dict(loss=per_rep([[0.216, 0.031, 0.555], [0.9, 0.2, 1.4]], "target", terms).T)
    ).assign_coords(technique=("target", ["quantiles", "correlation", "identity"]))
    tree = xr.DataTree.from_dict(
        dict(
            priors=priors,
            expert_fit=expert_fit,
            target_loss=target_loss,
            loss=loss,
            hyperparameters=hyperparameters,
        )
    )
    label = "all" if len(reps) > 1 else "best"
    tree.attrs = dict(
        method="parametric_prior",
        epochs=600,
        replication=label,
        reps=reps,
        draw_targets=["y2"],
    )
    return tree


@pytest.mark.parametrize(
    "name, reps, n_reps",
    [
        ("one_replication", [0], 1),
        ("best_of_two", [0], 2),
        ("all_of_two", [0, 1], 2),
    ],
)
def test_summary_output_matches_snapshot(name, reps, n_reps):
    """The text output must equal the reviewed snapshot"""
    output = _format(summary_tree(reps, n_reps)) + "\n"
    snapshot = SNAPSHOTS / f"summary_{name}.txt"
    if UPDATE or not snapshot.exists():
        SNAPSHOTS.mkdir(exist_ok=True)
        snapshot.write_text(output)
        pytest.skip(f"wrote {snapshot.name}; review it, then run the test again")
    assert output == snapshot.read_text()
