"""
Tests for plotting functions
"""

import numpy as np
import pytest

import elicito as el

matplotlib = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")

matplotlib.use("Agg")


@pytest.fixture
def fitted_eliobj():
    """Fixture providing a fitted elicit object for testing."""
    from tests.utils import eliobj as base_eliobj

    eliobj_copy = el.Elicit(
        model=base_eliobj.model,
        parameters=base_eliobj.parameters,
        targets=base_eliobj.targets,
        expert=base_eliobj.expert,
        optimizer=base_eliobj.optimizer,
        trainer=el.trainer(method="parametric_prior", seed=0, epochs=1, progress=0),
        initializer=el.initializer(
            method="sobol",
            iterations=1,
            distribution=el.initializers.uniform(radius=1.0, mean=0.0),
        ),
    )
    eliobj_copy.fit()
    return eliobj_copy


@pytest.fixture
def unfitted_eliobj():
    """Fixture providing an unfitted elicit object for testing."""
    from tests.utils import eliobj as base_eliobj

    uniform_dist = el.initializers.uniform(radius=1.0, mean=0.0)

    eliobj_copy = el.Elicit(
        model=base_eliobj.model,
        parameters=base_eliobj.parameters,
        targets=base_eliobj.targets,
        expert=base_eliobj.expert,
        optimizer=base_eliobj.optimizer,
        trainer=el.trainer(method="parametric_prior", seed=0, epochs=2),
        initializer=el.initializer(
            method="sobol",
            iterations=2,
            distribution=uniform_dist,
        ),
    )
    return eliobj_copy


@pytest.fixture(autouse=True)
def close_plots():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestPlottingFunctions:
    """Test suite for all plotting functions."""

    def test_initialization_plot(self, fitted_eliobj):
        """Test the initialization plot function."""
        fig, axes = el.plots.initialization(fitted_eliobj)
        assert fig is not None
        assert axes.shape == (5,)
        assert axes[0].get_gridspec().get_geometry()[1] == 4
        plt.close(fig)

    def test_initialization_plot_with_custom_params(self, fitted_eliobj):
        """Test initialization plot with custom parameters."""
        titles = [r"$\mu$", "Param 2", "Param 3", "Param 4", "Param 5"]
        fig, axes = el.plots.initialization(fitted_eliobj, cols=3, titles=titles)
        assert fig is not None
        assert axes.shape == (5,)
        assert axes[0].get_gridspec().get_geometry()[1] == 3
        assert [ax.get_title() for ax in axes] == titles
        plt.close(fig)

    def test_loss_plot(self, fitted_eliobj):
        """Test the loss plot function."""
        fig, axes = el.plots.loss(fitted_eliobj)
        assert fig is not None
        assert axes.shape == (2,)
        plt.close(fig)

    def test_hyperparameter_plot(self, fitted_eliobj):
        """Test the hyperparameter plot function."""
        fig, axes = el.plots.hyperparameter(fitted_eliobj)
        assert fig is not None
        assert axes is not None
        plt.close(fig)

    def test_hyperparameter_plot_with_titles(self, fitted_eliobj):
        """Test hyperparameter plot with custom titles."""
        titles = ["μ₀", "σ₀", "μ₁", "σ₁", "σ₂"]
        fig, axes = el.plots.hyperparameter(fitted_eliobj, titles=titles, cols=5)
        assert fig is not None
        assert axes.shape == (5,)
        assert [ax.get_title() for ax in axes] == titles
        plt.close(fig)

    def test_prior_joint_plot(self, fitted_eliobj):
        """Test the prior joint plot function."""
        titles = ["a", "b", "c"]
        fig, axes = el.plots.prior_joint(fitted_eliobj, titles=titles)
        assert fig is not None
        assert axes.shape == (3, 3)
        assert [ax.get_xlabel() for ax in np.diag(axes)] == titles
        plt.close(fig)

    def test_prior_joint_params(self, fitted_eliobj):
        """Test that 'params' selects a subset of the parameters."""
        names = [param["name"] for param in fitted_eliobj.parameters]
        selected = [names[2], names[0]]
        fig, axes = el.plots.prior_joint(fitted_eliobj, params=selected)
        assert axes.shape == (2, 2)
        assert [ax.get_xlabel() for ax in np.diag(axes)] == selected
        plt.close(fig)

    def test_prior_joint_unknown_param(self, fitted_eliobj):
        """Test that an unknown name in 'params' raises a ValueError."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            el.plots.prior_joint(fitted_eliobj, params=["not_a_parameter"])

    def test_prior_marginals_plot(self, fitted_eliobj):
        """Test the prior marginals plot function."""
        titles = ["$\beta_0$", "$\beta_1$", r"$\sigma$"]
        fig, axes = el.plots.prior_marginals(fitted_eliobj, titles=titles)
        assert fig is not None
        assert axes.shape == (3,)
        assert [ax.get_title() for ax in axes] == titles
        plt.close(fig)

    def test_prior_marginals_params(self, fitted_eliobj):
        """Test that 'params' selects a subset of the parameters."""
        names = [param["name"] for param in fitted_eliobj.parameters]
        selected = [names[2], names[0]]
        fig, axes = el.plots.prior_marginals(fitted_eliobj, params=selected)
        assert axes.shape == (2,)
        assert [ax.get_title() for ax in axes] == selected
        plt.close(fig)

    def test_prior_marginals_unknown_param(self, fitted_eliobj):
        """Test that an unknown name in 'params' raises a ValueError."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            el.plots.prior_marginals(fitted_eliobj, params=["not_a_parameter"])

    def test_elicits_plot(self, fitted_eliobj):
        """Test elicits plot with custom column layout."""
        fig, axes = el.plots.elicits(fitted_eliobj, cols=1)
        assert fig is not None
        assert axes.shape == (3,)
        plt.close(fig)

    def test_priorpredictive_plot(self, fitted_eliobj):
        """Test the prior predictive plot function."""
        target_name = fitted_eliobj.targets[0].name
        fig, axes = el.plots.priorpredictive(fitted_eliobj, target=target_name)
        assert fig is not None
        assert axes.shape == (1,)
        plt.close(fig)

    def test_prior_averaging_plot(self, fitted_eliobj):
        """Test the prior averaging plot function."""
        fig, axes = el.plots.prior_averaging(fitted_eliobj)
        assert fig is not None
        assert axes.shape == (2,)
        assert axes[0].get_suptitle() == "Prior averaging (weights)"
        assert axes[1].get_suptitle() == "Prior distributions"
        plt.close(fig)

    def test_plots_require_fitted_object(self, unfitted_eliobj):
        """Test that loss plotting function requires a fitted object."""
        with pytest.raises((AttributeError, ValueError, TypeError, KeyError)):
            el.plots.loss(unfitted_eliobj)

    def test_plots_with_zero_cols(self, fitted_eliobj):
        """Test plotting functions with invalid column count."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            el.plots.initialization(fitted_eliobj, cols=0)

    def test_priorpredictive_invalid_target(self, fitted_eliobj):
        """Test priorpredictive plot with non-existent target."""
        with pytest.raises((ValueError, KeyError)):
            el.plots.priorpredictive(fitted_eliobj, target="nonexistent_target")


def test_density_of_a_collapsed_parameter():
    """two distinct draws have no density; the panel must not raise"""
    fig, ax = plt.subplots()

    two_values = np.where(np.arange(1000) % 2 == 0, 0.0, 17.5)
    el.plots._plot_density(ax, two_values, "ts", color="black")
    # one vertical line per value, instead of a density
    assert len(ax.lines) == 2

    el.plots._plot_density(ax, np.full(1000, 3.0), "s0", color="black")
    assert len(ax.lines) == 3

    plt.close(fig)


def test_density_leaves_out_non_finite_draws():
    """a single non-finite draw must not remove the density"""
    fig, ax = plt.subplots()

    draws = np.concatenate([np.random.default_rng(0).normal(size=999), [np.inf]])
    el.plots._plot_density(ax, draws, "b", color="black")

    assert len(ax.lines) == 1
    assert bool(np.isfinite(ax.lines[0].get_xdata()).all())
    plt.close(fig)


def test_prior_joint_survives_a_density_that_fails(fitted_eliobj, monkeypatch):
    """a density that raises must cost one panel, not the whole figure"""
    from arviz_stats.base import array_stats

    def failing_kde(ary, **kwargs):
        msg = "cannot convert float infinity to integer"
        raise OverflowError(msg)

    monkeypatch.setattr(array_stats, "kde", failing_kde)

    fig, axes = el.plots.prior_joint(fitted_eliobj)

    # every density is empty, and the scatter panels are still drawn
    assert all(len(axes[i, i].lines) == 0 for i in range(axes.shape[0]))
    assert len(axes[0, 1].lines) == 1
    plt.close(fig)


def test_prior_joint_panels_share_the_axis_of_their_column(fitted_eliobj):
    """the scatter of column j uses the parameter of the density in column j"""
    draws = fitted_eliobj.sample()
    priors = (
        draws["prior"]
        .sel(replication=0)
        .to_dataset()
        .to_array()
        .stack(stacked=("batch", "draw"))
        .values
    )
    fig, axes = el.plots.prior_joint(fitted_eliobj, samples=draws)

    # the column sets x, the row sets y
    scatter = axes[0, 1].lines[0]
    np.testing.assert_allclose(scatter.get_xdata(), priors[1], rtol=1e-6)
    np.testing.assert_allclose(scatter.get_ydata(), priors[0], rtol=1e-6)
    plt.close(fig)


def test_plots_without_an_initialization_group():
    """`initializer(method="cmaes")` draws no candidates, so it writes no group"""
    pytest.importorskip("cma")
    from tests.utils import eliobj as base

    eliobj = el.Elicit(
        model=base.model,
        parameters=base.parameters,
        targets=base.targets,
        expert=base.expert,
        optimizer=el.optimizer(
            optimizer=el.optimizers.cmaes.CMAES, sigma0=0.5, popsize=4
        ),
        trainer=el.trainer(method="parametric_prior", seed=0, epochs=8, progress=0),
    )
    eliobj.fit()
    assert "initialization" not in eliobj.results.children

    # the names come from the model, so the convergence plot still works
    fig, axes = el.plots.hyperparameter(eliobj)
    names = el.optimizers.search.hyper_names(eliobj.parameters)
    assert [ax.get_title() for ax in axes] == names
    plt.close(fig)

    # the ecdf of the initialization distribution has nothing to show
    with pytest.raises(KeyError, match="draws no candidates"):
        el.plots.initialization(eliobj)


def test_hyperparameter_plot_leaves_out_the_gradients(fitted_eliobj):
    """the history holds a gradient per hyperparameter; they are not panels"""
    recorded = list(fitted_eliobj.results.history_stats.hyperparameter.data_vars)
    assert any(name.startswith("grad_") for name in recorded)

    fig, axes = el.plots.hyperparameter(fitted_eliobj)
    assert axes.shape == (
        len(el.optimizers.search.hyper_names(fitted_eliobj.parameters)),
    )
    plt.close(fig)


def test_plots_accept_a_cmaes_fit():
    """the plots read the epoch axis as generations, not as gradient steps"""
    pytest.importorskip("cma")
    from tests.utils import eliobj as base

    eliobj = el.Elicit(
        model=base.model,
        parameters=base.parameters,
        targets=base.targets,
        expert=base.expert,
        optimizer=el.optimizer(
            optimizer=el.optimizers.cmaes.CMAES, sigma0=0.5, popsize=4
        ),
        trainer=el.trainer(method="parametric_prior", seed=0, epochs=12, progress=0),
    )
    eliobj.fit()

    # 12 evaluations of 4 candidates are 3 generations, not 12 epochs
    el.plots.loss(eliobj)
    el.plots.hyperparameter(eliobj)
    el.plots.prior_joint(eliobj)
    plt.close("all")
