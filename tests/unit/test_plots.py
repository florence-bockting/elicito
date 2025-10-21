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
            distribution=el.initialization.uniform(radius=1.0, mean=0.0),
        ),
    )
    eliobj_copy.fit()
    return eliobj_copy


@pytest.fixture
def unfitted_eliobj():
    """Fixture providing an unfitted elicit object for testing."""
    from tests.utils import eliobj as base_eliobj

    uniform_dist = el.initialization.uniform(radius=1.0, mean=0.0)

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

    def test_prior_marginals_plot(self, fitted_eliobj):
        """Test the prior marginals plot function."""
        titles = ["$\beta_0$", "$\beta_1$", r"$\sigma$"]
        fig, axes = el.plots.prior_marginals(fitted_eliobj, titles=titles)
        assert fig is not None
        assert axes.shape == (3,)
        assert [ax.get_title() for ax in axes] == titles
        plt.close(fig)

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
