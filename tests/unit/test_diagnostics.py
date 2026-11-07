"""
tests for the helpers of the consistency diagnostics
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from elicito import diagnostics


class _Refit:
    """Stand-in for a fitted eliobj that returns fixed statistics"""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def sample(self):
        # one replication
        dims = ("replication", "batch", "element")
        summary = xr.Dataset({"quantiles_y": (dims, self.values[None])})
        return {"elicited_summary": xr.DataTree(summary)}


def test_components_are_sorted_by_the_index_of_the_target():
    losses = xr.Dataset(
        {
            name: ((), 0.0)
            for name in ["total_loss", "penalty", "b_loss_1", "c_loss_10", "d_loss_2"]
        }
    ).assign(a_loss_0=0.0)

    assert diagnostics._components(losses) == [
        "a_loss_0",
        "b_loss_1",
        "d_loss_2",
        "c_loss_10",
    ]


def test_statistics_strip_the_loss_suffix():
    names = ["quantiles_k4_t4_loss_0", "quantiles_k4_t59_loss_2"]

    assert diagnostics._statistics(names) == ["quantiles_k4_t4", "quantiles_k4_t59"]


def test_distance_is_zero_for_an_exact_match():
    refit = _Refit([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    assert diagnostics._distance(refit, "quantiles_y", {"quantiles_y": [1, 2, 3]}) == 0


def test_distance_is_in_units_of_the_expert_interval():
    # the mean prediction is (1, 2, 5); the expert interval is 3 - 1 = 2
    refit = _Refit([[1.0, 2.0, 4.0], [1.0, 2.0, 6.0]])

    assert diagnostics._distance(refit, "quantiles_y", {"quantiles_y": [1, 2, 3]}) == 1


def test_distance_is_nan_for_a_statistic_with_one_value():
    refit = _Refit([[1.0], [1.0]])

    assert math.isnan(diagnostics._distance(refit, "quantiles_y", {"quantiles_y": [1]}))


def test_miss_is_the_largest_distance_for_each_seed():
    # the mean prediction is (1, 2, 5); the largest distance is 5 - 3 = 2
    refit = _Refit([[1.0, 2.0, 4.0], [1.0, 2.0, 6.0]])

    miss = diagnostics._miss(refit, "quantiles_y", {"quantiles_y": [1, 2, 3]})

    assert np.array_equal(miss, [2.0])


def test_relative_drop_is_a_symmetric_percent_change():
    drop = diagnostics._relative_drop([2.0, 2.0, 2.0], [1.1, 1.0, 0.9])

    assert np.allclose(drop, [200 * 0.9 / 3.1, 200 / 3.0, 200 * 1.1 / 2.9])


def test_relative_drop_is_bounded_when_the_full_loss_is_near_zero():
    drop = diagnostics._relative_drop([1e-9, 1.0], [1.0, 1e-9])

    assert np.allclose(drop, [-200.0, 200.0])


def test_t_interval_is_zero_without_a_difference():
    assert diagnostics._t_interval([0.0, 0.0, 0.0]) == (0.0, 0.0, 0.0)


def test_t_interval_has_the_t_half_width():
    # t(0.975, 2) * 5 / sqrt(3) = 12.42
    mean, lower, upper = diagnostics._t_interval([45.0, 50.0, 55.0])

    assert np.isclose(mean, 50.0)
    assert np.isclose(upper - mean, 12.42, atol=1e-2)
    assert np.isclose(mean - lower, upper - mean)


def test_leading_is_the_clearly_largest_gain():
    # c is far above a and b, like the location case
    drop = [[6.0, 7.0, 6.5], [6.0, 7.0, 6.5], [16.0, 17.0, 16.5]]

    assert diagnostics._leading(drop, threshold=0.0) == [False, False, True]


def test_leading_keeps_gains_that_do_not_differ_clearly():
    # a and c are about equal, b is 0, like the case "all differ"
    drop = [[14.0, 22.0, 18.0], [0.0, 0.2, -0.1], [16.0, 19.0, 17.0]]

    assert diagnostics._leading(drop, threshold=0.0) == [True, False, True]


def test_leading_is_empty_without_an_interval_above_zero():
    drop = [[0.1, -0.1, 0.0], [0.2, -0.3, 0.1], [0.0, 0.1, -0.1]]

    assert diagnostics._leading(drop, threshold=0.0) == [False, False, False]


def test_leading_is_empty_below_the_threshold():
    # every interval is above 0, but no interval is above 1
    values = [[0.5, 0.6, 0.55], [0.4, 0.5, 0.45], [0.6, 0.7, 0.65]]

    assert diagnostics._leading(values, threshold=1.0) == [False, False, False]


def test_leave_one_out_needs_a_positive_tolerance_for_each_statistic():
    reference = xr.Dataset(
        {"quantiles_a_loss_0": ("seed", [1.0, 2.0])}, attrs={"seeds": [1, 2]}
    )

    with pytest.raises(ValueError, match="quantiles_a"):
        diagnostics.leave_one_out(None, reference, tolerance={"quantiles_a": 0.0})


class _Fit:
    """Stand-in for a refit with a flat loss and zero statistics"""

    def __init__(self):
        dims = ("replication", "epoch")
        names = ["total_loss", "quantiles_a_loss_0", "quantiles_b_loss_1"]
        loss = np.ones((2, 10))
        losses = xr.Dataset({name: (dims, loss) for name in names})
        self.results = xr.DataTree.from_dict({"history_stats/loss": losses})

    def sample(self):
        dims = ("replication", "batch", "element")
        values = np.zeros((2, 1, 3))
        summary = xr.Dataset(
            {name: (dims, values) for name in ["quantiles_a", "quantiles_b"]}
        )
        return {"elicited_summary": xr.DataTree(summary)}


def test_leave_one_out_finds_no_conflict_without_a_miss(monkeypatch):
    monkeypatch.setattr(diagnostics, "_refit", lambda *args: _Fit())
    eliobj = SimpleNamespace(
        targets=[{"weight": 1.0}, {"weight": 1.0}],
        results=xr.DataTree.from_dict(
            {
                "expert": xr.Dataset(
                    {
                        name: (("replication", "batch", "element"), np.zeros((1, 1, 3)))
                        for name in ["quantiles_a", "quantiles_b"]
                    }
                )
            }
        ),
    )
    reference = xr.Dataset(
        {
            name: ("replication", [1.0, 1.0])
            for name in ["quantiles_a_loss_0", "quantiles_b_loss_1"]
        },
        attrs={"seeds": [1, 2]},
    )
    tolerance = {"quantiles_a": 1.0, "quantiles_b": 1.0}

    diagnostic = diagnostics.leave_one_out(eliobj, reference, tolerance)

    assert diagnostic.attrs["inconsistent"] is False
    assert not diagnostic["leading"].values.any()


def _diagnostic(leading, inconsistent=True):
    mean_miss = np.array([1.0, 2.0, 3.0])
    return xr.Dataset(
        {
            "mean_miss": ("statistic", mean_miss),
            "lower": ("statistic", mean_miss - 1),
            "upper": ("statistic", mean_miss + 1),
            "leading": ("statistic", leading),
        },
        coords={"statistic": ["a", "b", "c"]},
        attrs={"inconsistent": inconsistent},
    )


def _plot(diagnostic, **kwargs):
    import matplotlib

    matplotlib.use("Agg")
    from elicito import plots

    _, axes = plots.consistency(diagnostic, **kwargs)
    return axes[0]


def _bar_colors(ax):
    import matplotlib

    return [
        matplotlib.colors.to_hex(bar.get_facecolor())
        for container in ax.containers
        if isinstance(container, matplotlib.container.BarContainer)
        for bar in container.patches
    ]


def test_consistency_plot_uses_the_given_xlim():
    ax = _plot(_diagnostic([False, False, True]), xlim=(-10, 10))

    assert ax.get_xlim() == (-10, 10)


def test_consistency_plot_colors_only_the_leading_bars():
    colors = _bar_colors(_plot(_diagnostic([True, False, True])))

    assert colors[0] == colors[2]
    assert colors[0] != colors[1]


def test_consistency_plot_title_says_whether_the_information_is_inconsistent():
    assert _plot(_diagnostic([False] * 3, False)).get_title() == (
        "no evidence of a conflict"
    )
    assert _plot(_diagnostic([False, False, True])).get_title() == "inconsistent"
