"""
plotting helpers
"""

import itertools
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorflow as tf

from elicito.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def initialization(
    eliobj: Any, cols: int = 4, **kwargs: Any
) -> tuple["matplotlib.figure.Figure", list["matplotlib.axes.Axes"]]:
    """
    Plot the ecdf of the initialization distribution per hyperparameter

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.
    cols : int, optional
        number of columns for arranging the subplots in the figure.
        The default is ``4``.
    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.initialization(eliobj, cols=6)  # doctest: +SKIP

    >>> el.plots.initialization(eliobj, cols=4, figsize=(8, 3))  # doctest: +SKIP

    Raises
    ------
    KeyError
        Can't find 'init_matrix' in eliobj.results. Have you excluded it from
        saving?

    ValueError
        if `eliobj.results["init_matrix"]` is None: No samples from
        initialization distribution found. This plot function cannot be used
        if initial values were fixed by the user through the `hyperparams`
        argument in :func:`elicit.elicit.initializer`.

    """
    eliobj_res, _, _, _ = _check_parallel(eliobj)
    # get number of hyperparameter
    n_par = len(eliobj_res["init_matrix"].keys())
    # prepare plot axes
    (cols, rows, k, low, high) = _prep_subplots(eliobj, cols, n_par, bounderies=True)

    kwargs.setdefault("figsize", (cols * 2, rows * 2))
    kwargs.setdefault("constrained_layout", True)
    kwargs.setdefault("sharex", True)

    # check that all information can be assessed
    try:
        eliobj_res["init_matrix"]
    except KeyError:
        logger.warning(
            "Can't find 'init_matrix' in eliobj.results."
            + " Have you excluded it from saving?"
        )

    if eliobj_res["init_matrix"] is None:
        raise ValueError(
            "No samples from initialization distribution found."
            + " This plot function cannot be used if initial values were"
            + " fixed by the user through the `hyperparams` argument of"
            + " `initializer`."
        )

    # plot ecdf of initialization distribution
    # differentiate between subplots that have (1) only one row vs.
    # (2) subplots with multiple rows

    fig, axes = _setup_grid(rows, cols, k, **kwargs)

    for ax, hyp, lo, hi in zip(axes, eliobj_res["init_matrix"], low, high):
        [
            ax.ecdf(
                tf.squeeze(eliobj.results[j]["init_matrix"][hyp]),
                color="black",
                lw=2,
                alpha=0.5,
            )
            for j in range(len(eliobj.results))
        ]
        ax.set_title(f"{hyp}", fontsize="small")
        ax.axline((lo, 0), (hi, 1), color="grey", linestyle="dashed", lw=1)
        ax.grid(color="lightgrey", linestyle="dotted", linewidth=1)
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(axis="y", labelsize="x-small")
        ax.tick_params(axis="x", labelsize="x-small")

    fig.suptitle("ecdf of initialization distributions", fontsize="medium")

    return fig, axes


def loss(
    eliobj: Any,
    weighted: bool = True,
    **kwargs: Any,
) -> tuple["matplotlib.figure.Figure", list["matplotlib.axes.Axes"]]:
    """
    Plot the total loss and the loss per component.

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.

    weighted : bool, optional
        Weight the loss per component.

    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.loss(eliobj, figsize=(8, 3))  # doctest: +SKIP

    Raises
    ------
    KeyError
        Can't find 'loss_component' in 'eliobj.history'. Have you excluded
        'loss_components' from history savings?

        Can't find 'loss' in 'eliobj.history'. Have you excluded 'loss' from
        history savings?

        Can't find 'elicited_statistics' in 'eliobj.results'. Have you
        excluded 'elicited_statistics' from results savings?

    """
    eliobj_res, eliobj_hist, parallel, n_reps = _check_parallel(eliobj)
    # names of loss_components
    names_losses = eliobj_res["elicited_statistics"].keys()
    # get weights in targets
    if weighted:
        in_title = "weighted "
        weights = [eliobj.targets[i]["weight"] for i in range(len(eliobj.targets))]
    else:
        in_title = ""
        weights = [1.0] * len(eliobj.targets)
    # check chains that yield NaN
    if parallel:
        _, success, _ = _check_NaN(eliobj, n_reps)
    else:
        success = [0]
    # check that all information can be assessed
    try:
        eliobj_hist["loss_component"]
    except KeyError:
        logger.warning(
            "No information about 'loss_component' found in 'eliobj.history'."
            + "Have you excluded 'loss_components' from history savings?"
        )
    try:
        eliobj_hist["loss"]
    except KeyError:
        logger.warning(
            "No information about 'loss' found in 'eliobj.history'."
            + "Have you excluded 'loss' from history savings?"
        )
    try:
        eliobj_res["elicited_statistics"]
    except KeyError:
        logger.warning(
            "No information about 'elicited_statistics' found in "
            + "'eliobj.results'. Have you excluded 'elicited_statistics' from"
            + "results savings?"
        )

    kwargs.setdefault("figsize", (6, 2))
    kwargs.setdefault("constrained_layout", True)
    kwargs.setdefault("sharex", True)

    fig, axes = _setup_grid(1, 2, **kwargs)
    # plot total loss
    [
        axes[0].plot(eliobj.history[i]["loss"], color="black", alpha=0.5, lw=2)
        for i in success
    ]
    # plot loss per component
    for i, name in enumerate(names_losses):
        for j in success:
            # preprocess loss_component results
            indiv_losses = tf.stack(eliobj.history[j]["loss_component"])
            if j == 0:
                axes[1].plot(
                    indiv_losses[:, i] * weights[i], label=name, lw=2, alpha=0.5
                )
            else:
                axes[1].plot(indiv_losses[:, i] * weights[i], lw=2, alpha=0.5)
        axes[1].legend(fontsize="small", handlelength=0.4, frameon=False)
    [
        axes[i].set_title(t, fontsize="small")
        for i, t in enumerate(["total loss", in_title + "individual losses"])
    ]
    for i in range(2):
        axes[i].set_xlabel("epochs", fontsize="small")
        axes[i].grid(color="lightgrey", linestyle="dotted", linewidth=1)
        axes[i].spines[["right", "top"]].set_visible(False)
        axes[i].tick_params(axis="y", labelsize="x-small")
        axes[i].tick_params(axis="x", labelsize="x-small")

    return fig, axes


def hyperparameter(
    eliobj: Any, cols: int = 4, **kwargs: Any
) -> tuple["matplotlib.figure.Figure", list["matplotlib.axes.Axes"]]:
    """
    Plot the convergence of each hyperparameter across epochs.

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.

    cols : int, optional
        number of columns for arranging the subplots in the figure.
        The default is ``4``.

    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.hyperparameter(eliobj)  # doctest: +SKIP

    Raises
    ------
    KeyError
        Can't find 'hyperparameter' in 'eliobj.history'. Have you excluded
        'hyperparameter' from history savings?

    """
    _, eliobj_hist, parallel, n_reps = _check_parallel(eliobj)
    # names of hyperparameter
    names_par = list(eliobj_hist["hyperparameter"].keys())
    # get number of hyperparameter
    n_par = len(names_par)
    # check chains that yield NaN
    if parallel:
        _, success, _ = _check_NaN(eliobj, n_reps)
    else:
        success = [0]
    # prepare subplot axes
    (cols, rows, k) = _prep_subplots(eliobj, cols, n_par, bounderies=False)

    kwargs.setdefault("figsize", (cols * 2, rows * 2))
    kwargs.setdefault("constrained_layout", True)
    kwargs.setdefault("sharex", True)

    # check that all information can be assessed
    try:
        eliobj_hist["hyperparameter"]
    except KeyError:
        logger.warning(
            "No information about 'hyperparameter' found in "
            + "'eliobj.history'. Have you excluded 'hyperparameter' from"
            + "history savings?"
        )

    fig, axes = _setup_grid(rows, cols, **kwargs)
    for ax, hyp in zip(axes, names_par):
        for i in success:
            ax.plot(
                eliobj.history[i]["hyperparameter"][hyp],
                color="black",
                lw=2,
                alpha=0.5,
            )
        ax.set_title(f"{hyp}", fontsize="small")
        ax.tick_params(axis="y", labelsize="x-small")
        ax.tick_params(axis="x", labelsize="x-small")
        ax.set_xlabel("epochs", fontsize="small")
        ax.grid(color="lightgrey", linestyle="dotted", linewidth=1)
        ax.spines[["right", "top"]].set_visible(False)

    fig.suptitle("Convergence of hyperparameter", fontsize="medium")

    for i in range(k):
        axes[-(i + 1)].set_axis_off()

    return fig, axes


def prior_joint(
    eliobj: Any,
    idx: int | list[int] | None = None,
    **kwargs: dict[Any, Any],
) -> tuple["matplotlib.figure.Figure", list["matplotlib.axes.Axes"]]:
    """
    Plot learned prior distributions

    Plot prior of each model parameter based on prior samples from last epoch.
    If parallelization has been used, select which replication you want to
    investigate by indexing it through the 'idx' argument.

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.

    idx : int or list of int, optional
        only required if parallelization is used for fitting the method.
        Indexes the replications and allows to choose for which replication(s) the
        joint prior should be shown.

    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.prior_joint(eliobj, figsize=(4, 4))  # doctest: +SKIP

    Raises
    ------
    ValueError
        Currently only 'positive' can be used as constraint. Found unsupported
        constraint type.

        The value for 'idx' is larger than the number of parallelizations.

    KeyError
        Can't find 'prior_samples' in 'eliobj.results'. Have you excluded
        'prior_samples' from results savings?

    """
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    try:
        from arviz_stats.base import array_stats  # type: ignore
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="arviz_stats"
        ) from exc

    if idx is None:
        idx = [0]
    if type(idx) is not list:
        idx = [idx]  # type: ignore
    if len(idx) > len(eliobj.results):
        raise ValueError(
            "The value for 'idx' is larger than the number"
            + " of parallelizations. 'idx' should not exceed"
            + f" {len(eliobj.results)} but got {len(idx)}."
        )
    if len(eliobj.history[0]["loss"]) < eliobj.trainer["epochs"]:
        raise ValueError(
            f"Training failed for seed with index={idx} (loss is NAN)."
            + " No results for plotting available."
        )
    # select one result set
    eliobj_res = eliobj.results[0]
    # check that all information can be assessed
    try:
        eliobj_res["prior_samples"]
    except KeyError:
        logger.warning(
            "No information about 'prior_samples' found in "
            + "'eliobj.results'. Have you excluded 'prior_samples' from"
            + "results savings?"
        )
    cmap = mpl.colormaps["turbo"]
    # get shape of prior samples
    B, n_samples, n_params = eliobj_res["prior_samples"].shape
    # get parameter names
    name_params = [eliobj.parameters[i]["name"] for i in range(n_params)]

    fig, axs = plt.subplots(n_params, n_params, constrained_layout=True, **kwargs)  # type: ignore
    colors = cmap(np.linspace(0, 1, len(idx)))
    for c, k in enumerate(idx):
        for i in range(n_params):
            # reshape samples by merging batches and number of samples
            priors = tf.reshape(
                eliobj.results[k]["prior_samples"], (B * n_samples, n_params)
            )
            grid, pdf, _ = array_stats.kde(priors[:, i])  # type: ignore
            axs[i, i].plot(grid, pdf, color=colors[c], lw=2)

            axs[i, i].set_xlabel(name_params[i], size="small")
            [axs[i, i].tick_params(axis=a, labelsize="x-small") for a in ["x", "y"]]
            axs[i, i].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[i, i].spines[["right", "top"]].set_visible(False)

        for i, j in itertools.combinations(range(n_params), 2):
            grid, pdf, _ = array_stats.kde(priors[:, i])  # type: ignore
            axs[i, i].plot(grid, pdf, color=colors[c], lw=2)
            axs[i, j].plot(priors[:, i], priors[:, j], ",", color=colors[c], alpha=0.1)
            [axs[i, j].tick_params(axis=a, labelsize=7) for a in ["x", "y"]]
            axs[j, i].set_axis_off()
            axs[i, j].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[i, j].spines[["right", "top"]].set_visible(False)
    fig.suptitle("Learned joint prior", fontsize="medium")

    return fig, axs


def prior_marginals(
    eliobj: Any, cols: int = 4, **kwargs: Any
) -> tuple["matplotlib.figure.Figure", list["matplotlib.axes.Axes"]]:
    """
    Plot the convergence of each hyperparameter across epochs.

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.

    cols : int, optional
        number of columns for arranging the subplots in the figure.
        The default is ``4``.

    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.prior_marginals(eliobj)  # doctest: +SKIP

    Raises
    ------
    KeyError
        Can't find 'prior_samples' in 'eliobj.results'. Have you excluded
        'prior_samples' from results savings?

    """
    try:
        from arviz_stats.base import array_stats  # type: ignore
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="arviz_stats"
        ) from exc

    eliobj_res, _, parallel, n_reps = _check_parallel(eliobj)
    # check chains that yield NaN
    if parallel:
        _, success, _ = _check_NaN(eliobj, n_reps)
    else:
        success = [0]
    # get shape of prior samples
    B, n_samples, n_par = eliobj_res["prior_samples"].shape
    # get parameter names
    name_params = [eliobj.parameters[i]["name"] for i in range(n_par)]
    # prepare plot axes
    (cols, rows, k) = _prep_subplots(eliobj, cols, n_par, bounderies=False)

    kwargs.setdefault("figsize", (cols * 2, rows * 2))
    kwargs.setdefault("constrained_layout", True)

    # check that all information can be assessed
    try:
        eliobj_res["prior_samples"]
    except KeyError:
        logger.warning(
            "No information about 'prior_samples' found in "
            + "'eliobj.results'. Have you excluded 'prior_samples' from"
            + "results savings?"
        )

    fig, axes = _setup_grid(rows, cols, **kwargs)

    for j, (ax, par) in enumerate(zip(axes, name_params)):
        for i in success:
            priors = tf.reshape(
                eliobj.results[i]["prior_samples"], (B * n_samples, n_par)
            )
            grid, pdf, _ = array_stats.kde(priors[:, j])  # type: ignore
            ax.plot(grid, pdf, color="black", lw=2, alpha=0.5)

        ax.set_title(f"{par}", fontsize="small")
        ax.tick_params(axis="y", labelsize="x-small")
        ax.tick_params(axis="x", labelsize="x-small")
        ax.set_xlabel("\u03b8", fontsize="small")
        ax.set_ylabel("density", fontsize="small")
        ax.grid(color="lightgrey", linestyle="dotted", linewidth=1)
        ax.spines[["right", "top"]].set_visible(False)

    fig.suptitle("Learned marginal priors", fontsize="medium")

    return fig, axes


def elicits(
    eliobj: Any, cols: int = 4, **kwargs: Any
) -> tuple["matplotlib.figure.Figure", list["matplotlib.axes.Axes"]]:
    """
    Plot the expert-elicited vs. model-simulated statistics.

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.

    cols : int, optional
        number of columns for arranging the subplots in the figure.
        The default is ``4``.

    **kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.elicits(eliobj, cols=4, figsize=(7, 3))  # doctest: +SKIP

    Raises
    ------
    KeyError
        Can't find 'expert_elicited_statistics' in 'eliobj.results'. Have you
        excluded 'expert_elicited_statistics' from results savings?

        Can't find 'elicited_statistics' in 'eliobj.results'. Have you
        excluded 'elicited_statistics' from results savings?

    """
    # check whether parallelization has been used
    eliobj_res, eliobj_hist, parallel, n_reps = _check_parallel(eliobj)
    # get number of hyperparameter
    n_elicits = len(eliobj_res["expert_elicited_statistics"].keys())
    # check chains that yield NaN
    if parallel:
        _, success, _ = _check_NaN(eliobj, n_reps)
    else:
        success = [0]
    # prepare plot axes
    (cols, rows, k) = _prep_subplots(eliobj, cols, n_elicits, bounderies=False)

    kwargs.setdefault("figsize", (cols * 2, rows * 2))
    kwargs.setdefault("constrained_layout", True)

    # extract quantities of interest needed for plotting
    name_elicits = list(eliobj_res["expert_elicited_statistics"].keys())
    method_name = [name_elicits[i].split("_")[0] for i in range(n_elicits)]

    # check that all information can be assessed
    try:
        eliobj_res["expert_elicited_statistics"]
    except KeyError:
        logger.warning(
            "No information about 'expert_elicited_statistics' found in "
            + "'eliobj.results'. Have you excluded 'expert_elicited_statistics'"
            + " from results savings?"
        )
    try:
        eliobj_res["elicited_statistics"]
    except KeyError:
        logger.warning(
            "No information about 'elicited_statistics' found in "
            + "'eliobj.results'. Have you excluded 'elicited_statistics'"
            + " from results savings?"
        )

    # plotting
    fig, axes = _setup_grid(rows, cols, **kwargs)

    for ax, elicit, meth in zip(axes, name_elicits, method_name):
        # Configure plotting method and preparation
        if meth == "quantiles":
            labels: list[tuple[str, str] | tuple[None, None] | None] = [None] * n_reps
            prep = (
                ax.axline((0, 0), slope=1, color="darkgrey", linestyle="dashed", lw=1),
            )
            method = _quantiles

        elif meth == "cor":
            labels = [("expert", "train")] + [(None, None) for _ in range(n_reps - 1)]
            method = _correlation
            num_cor = eliobj_res["elicited_statistics"][elicit].shape[-1]
            prep = (
                ax.set_ylim(-1, 1),
                ax.set_xlim(-0.5, num_cor),
                ax.set_xticks(
                    [i for i in range(num_cor)],
                    [f"cor{i}" for i in range(num_cor)],
                ),  # type: ignore
            )

        # Plot each successful run
        for i in success:
            (
                method(
                    ax,
                    eliobj.results[i]["expert_elicited_statistics"][elicit],
                    eliobj.results[i]["elicited_statistics"][elicit],
                    labels[i],
                )
                + prep
            )

        # Configure labels, legend, title
        if elicit.endswith("_cor"):
            ax.legend(fontsize="x-small", markerscale=0.5, frameon=False)
        ax.set_title(elicit, fontsize="small")
        ax.grid(color="lightgrey", linestyle="dotted", linewidth=1)
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(axis="y", labelsize="x-small")
        ax.tick_params(axis="x", labelsize="x-small")
        if not elicit.endswith("_cor"):
            ax.set_xlabel("expert", fontsize="small")
            ax.set_ylabel("model-sim.", fontsize="small")

    fig.suptitle("Expert vs. model-simulated elicited statistics", fontsize="medium")

    return fig, axes


def marginals(
    eliobj: Any,
    cols: int = 4,
    span: int = 30,
    **kwargs: Any,
) -> tuple["matplotlib.figure.Figure", np.ndarray[Any, Any]]:
    """
    Plot convergence of mean and sd of the prior marginals

    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.

    cols : int, optional
        number of columns for arranging the subplots in the figure.
        The default is ``4``.

    span : int, optional
        number of last epochs used to get a final averaged value for mean and
        sd of the prior marginal. The default is ``30``.

    kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_

    Examples
    --------
    >>> el.plots.marginals(eliobj)  # doctest: +SKIP

    Raises
    ------
    KeyError
        Can't find 'hyperparameter' in 'eliobj.history'. Have you excluded
        'hyperparameter' from history savings?

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    # check whether parallelization has been used
    (_, eliobj_hist, parallel, n_reps) = _check_parallel(eliobj)
    # check chains that yield NaN
    if parallel:
        _, success, _ = _check_NaN(eliobj, n_reps)
    else:
        success = [0]
    # number of marginals
    n_elicits = tf.stack(eliobj_hist["hyperparameter"]["means"]).shape[-1]
    # prepare plot axes
    cols, rows, k = _prep_subplots(eliobj, cols, n_elicits, bounderies=False)

    kwargs.setdefault("figsize", (cols * 2, rows * 2))
    kwargs.setdefault("layout", "constrained")

    # check that all information can be assessed
    try:
        eliobj_hist["hyperparameter"]
    except KeyError:
        logger.warning(
            "No information about 'hyperparameter' found in 'eliobj.history'"
            + " Have you excluded 'hyperparameter' from history savings?"
        )

    elicits_means = tf.stack(
        [eliobj.history[i]["hyperparameter"]["means"] for i in success]
    )
    elicits_std = tf.stack(
        [eliobj.history[i]["hyperparameter"]["stds"] for i in success]
    )

    fig = plt.figure(**kwargs)
    subfigs = fig.subfigures(2, 1, wspace=0.07)
    _convergence_plot(
        subfigs[0],
        elicits_means,
        span=span,
        label="mean",
        parallel=parallel,
        rows=rows,
        cols=cols,
        k=k,
        success=success,
    )
    _convergence_plot(
        subfigs[1],
        elicits_std,
        span=span,
        label="sd",
        parallel=parallel,
        rows=rows,
        cols=cols,
        k=k,
        success=success,
    )
    fig.suptitle("Convergence of prior marginals mean and sd", fontsize="medium")

    return fig, subfigs


def priorpredictive(
    eliobj: Any, **kwargs: Any
) -> tuple["matplotlib.figure.Figure", list["matplotlib.axes.Axes"]]:
    """
    Plot prior predictive distribution (PPD)

    PPD of samples from the generative model in the last epoch

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.
    kwargs : any, optional
        additional keyword arguments that can be passed to specify
        `plt.subplots() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_


    Examples
    --------
    >>> el.plots.priorpredictive(eliobj)  # doctest: +SKIP

    Raises
    ------
    KeyError
        Can't find 'target_quantities' in 'eliobj.results'. Have you excluded
        'target_quantities' from results savings?

    """
    # check that all information can be assessed
    try:
        eliobj.results[-1]["target_quantities"]
    except KeyError:
        logger.warning(
            "No information about 'target_quantities' found in 'eliobj.results'."
            + " Have you excluded 'target_quantities' from results savings?"
        )

    kwargs.setdefault("figsize", (6, 2))
    kwargs.setdefault("constrained_layout", True)

    target_reshaped = []
    for k in eliobj.results[-1]["target_quantities"]:
        target = eliobj.results[-1]["target_quantities"][k]
        target_reshaped.append(tf.reshape(target, (target.shape[0] * target.shape[1])))

    targets = tf.stack(target_reshaped, -1)

    fig, axes = _setup_grid(1, 1, **kwargs)
    axes[0].grid(color="lightgrey", linestyle="dotted", linewidth=1)
    for i in range(targets.shape[-1]):
        shade = i / (targets.shape[-1])
        axes[0].hist(
            targets[:, i],
            bins="auto",
            density=True,
            label=eliobj.targets[i]["name"],
            color=f"{shade}",
            alpha=0.5,
        )
    axes[0].legend(fontsize="small", handlelength=0.9, frameon=False)
    axes[0].set_title("prior predictive distribution", fontsize="small")
    axes[0].spines[["right", "top"]].set_visible(False)
    axes[0].tick_params(axis="y", labelsize="x-small")
    axes[0].tick_params(axis="x", labelsize="x-small")
    axes[0].set_xlabel(r"$y_{pred}$", fontsize="small")

    return fig, axes


def prior_averaging(  # noqa: PLR0913, PLR0915
    eliobj: Any,
    cols: int = 4,
    n_sim: int = 10_000,
    height_ratio: list[int | float] = [1, 1.5],
    weight_factor: float = 1.0,
    seed: int = 123,
    xlim_weights: float = 0.2,
    **kwargs: dict[Any, Any],
) -> tuple["matplotlib.figure.Figure", np.ndarray[Any, Any]]:
    """
    Plot prior averaging

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.

    cols : int, optional
        number of columns in plot

    n_sim : int, optional
        number of simulations

    height_ratio : list of int or float, optional
        height ratio of prior averaging plot

    weight_factor : float, optional
        weighting factor of each model in prior averaging

    xlim_weights : float, optional
        limit of x-axis of weights plot

    kwargs : any, optional
        additional arguments passed to matplotlib
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    try:
        from arviz_stats.base import array_stats  # type: ignore
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="arviz_stats"
        ) from exc

    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "data_wrangling", requirement="pandas"
        ) from exc

    # prepare plotting
    n_par = len(eliobj.parameters)
    name_par = [eliobj.parameters[i]["name"] for i in range(n_par)]
    label_avg = [" "] * (len(name_par) - 1) + ["average"]
    n_reps = len(eliobj.results)
    # prepare plot axes
    (cols, rows, k) = _prep_subplots(eliobj, cols, n_par)
    # modify success for non-parallel case
    if len(eliobj.results) == 1:
        success = [0]
        success_name = eliobj.trainer["seed"]
    else:
        # remove chains for which training yield NaN
        (fail, success, success_name) = _check_NaN(eliobj, n_reps)

    # perform model averaging
    (w_MMD, averaged_priors, B, n_samples) = _model_averaging(
        eliobj, weight_factor, success, n_sim, seed
    )
    # store results in data frame
    df = pd.DataFrame(dict(weight=w_MMD, seed=success_name))
    # sort data frame according to weight values
    df_sorted = df.sort_values(by="weight", ascending=False).reset_index(drop=True)

    # plot average and single priors
    fig = plt.figure(layout="constrained", **kwargs)  # type: ignore
    subfigs = fig.subfigures(2, 1, height_ratios=height_ratio)
    subfig0 = subfigs[0].subplots(1, 1)
    subfig1 = subfigs[1].subplots(rows, cols)

    # plot weights of model averaging
    seeds = df_sorted["seed"].tolist()
    weights = df_sorted["weight"].tolist()
    subfig0.barh(y=seeds, width=weights, color="darkgrey")
    subfig0.spines[["right", "top"]].set_visible(False)
    subfig0.grid(color="lightgrey", linestyle="dotted", linewidth=1)
    subfig0.set_xlabel("weight", fontsize="small")
    subfig0.set_ylabel("seed", fontsize="small")
    subfig0.tick_params(axis="y", labelsize="x-small")
    subfig0.tick_params(axis="x", labelsize="x-small")
    subfig0.set_xlim(0, xlim_weights)
    subfig0.set_yticks(seeds)
    subfig0.set_yticklabels(seeds)

    # plot individual priors and averaged prior
    axes = subfig1.ravel()

    for j, (ax, par, lab) in enumerate(zip(axes, name_par, label_avg)):
        # Plot prior samples for each success
        for i in success:
            prior = tf.reshape(
                eliobj.results[i]["prior_samples"], (B * n_samples, n_par)
            )
            grid, pdf, _ = array_stats.kde(prior[:, j])  # type: ignore
            ax.plot(grid, pdf, color="black", lw=2, alpha=0.5)

        # Plot averaged prior (in red)
        avg_prior = tf.reshape(averaged_priors, (B * n_sim, n_par))
        grid, pdf, _ = array_stats.kde(avg_prior[:, j])  # type: ignore

        if j == len(name_par) - 1:  # last subplot gets legend
            ax.plot(grid, pdf, color="red", lw=2, alpha=0.5, label=lab)
            ax.legend(handlelength=0.3, fontsize="small", frameon=False)
        else:
            ax.plot(grid, pdf, color="red", lw=2, alpha=0.5)

        # Formatting
        ax.set_title(f"{par}", fontsize="small")
        ax.tick_params(axis="y", labelsize="x-small")
        ax.tick_params(axis="x", labelsize="x-small")
        ax.set_ylabel("density", fontsize="small")
        ax.grid(color="lightgrey", linestyle="dotted", linewidth=1)
        ax.spines[["right", "top"]].set_visible(False)

    # Turn off unused axes
    for k_idx in range(k):
        axes[-k_idx - 1].set_axis_off()

    subfigs[0].suptitle("Prior averaging (weights)", fontsize="small", ha="left", x=0.0)
    subfigs[1].suptitle("Prior distributions", fontsize="small", ha="left", x=0.0)
    fig.suptitle("Prior averaging", fontsize="medium")

    return fig, subfigs


def _model_averaging(
    eliobj: Any, weight_factor: float, success: Any, n_sim: int, seed: int
) -> tuple[Any, ...]:
    # compute final loss per run by averaging over last x values
    mean_losses = np.stack([np.mean(eliobj.history[i]["loss"]) for i in success])
    # retrieve min MMD
    min_loss = min(mean_losses)
    # compute Delta_i MMD
    delta_MMD = mean_losses - min_loss
    # relative likelihood
    rel_likeli = np.exp(float(weight_factor) * delta_MMD)
    # compute Akaike weights
    w_MMD = rel_likeli / np.sum(rel_likeli)

    # model averaging
    # extract prior samples; shape = (num_sims, B*sim_prior, num_param)
    prior_samples = tf.stack([eliobj.results[i]["prior_samples"] for i in success], 0)
    num_success, B, n_samples, n_par = prior_samples.shape

    # sample component
    rng = np.random.default_rng(seed)
    sampled_component = rng.choice(
        np.arange(num_success), size=n_sim, replace=True, p=w_MMD
    )
    # sample observation index
    sampled_obs = rng.choice(np.arange(n_samples), size=n_sim, replace=True)

    # select prior
    averaged_priors = tf.stack(
        [
            prior_samples[rep, :, obs, :]
            for rep, obs in zip(sampled_component, sampled_obs)
        ]
    )

    return tuple((w_MMD, averaged_priors, B, n_samples))


def _check_parallel(eliobj: Any) -> tuple[Any, ...]:
    eliobj_res = eliobj.results[0]
    eliobj_hist = eliobj.history[0]

    if len(eliobj.results) > 1:
        parallel = True
        num_reps = len(eliobj.results)
    else:
        parallel = False
        num_reps = 1

    return tuple((eliobj_res, eliobj_hist, parallel, num_reps))


def _quantiles(
    axs: Any,
    expert: tf.Tensor,
    training: tf.Tensor,
    labels: list[tuple[str]],  # do not remove
) -> tuple[Any]:
    return (
        axs.plot(
            expert[0, :],
            tf.reduce_mean(training, axis=0),
            "o",
            ms=5,
            color="black",
            alpha=0.5,
        ),
    )


def _correlation(
    axs: Any, expert: tf.Tensor, training: tf.Tensor, labels: list[tuple[str]]
) -> tuple[Any, ...]:
    return (
        axs.plot(0, expert[:, 0], "*", color="red", label=labels[0], zorder=2),
        axs.plot(
            0,
            tf.reduce_mean(training[:, 0]),
            "s",
            color="black",
            label=labels[1],
            alpha=0.5,
            zorder=1,
        ),
        [
            axs.plot(i, expert[:, i], "*", color="red", zorder=2)
            for i in range(1, training.shape[-1])  # type: ignore
        ],
        [
            axs.plot(
                i,
                tf.reduce_mean(training[:, i]),
                "s",
                color="black",
                alpha=0.5,
                zorder=1,
            )
            for i in range(1, training.shape[-1])  # type: ignore
        ],
    )


def _prep_subplots(
    eliobj: Any, cols: int, n_quant: Any, bounderies: bool = False
) -> tuple[Any, ...]:
    # make sure that user uses only as many columns as hyperparameter
    # such that session does not crash...
    if cols > n_quant:
        cols = n_quant
        logger.info(f"Reset cols={cols}")
    # compute number of rows for subplots
    rows, remainder = np.divmod(n_quant, cols)

    if bounderies:
        # get lower and upper boundary of initialization distr. (x-axis)
        low = tf.subtract(
            eliobj.initializer["distribution"]["mean"],
            eliobj.initializer["distribution"]["radius"],
        )
        high = tf.add(
            eliobj.initializer["distribution"]["mean"],
            eliobj.initializer["distribution"]["radius"],
        )
        try:
            len(low)
        except TypeError:
            low = [low] * n_quant
            high = [high] * n_quant
        else:
            pass

    # use remainder to track which plots should be turned-off/hidden
    if remainder != 0:
        rows += 1
        k = cols - remainder
    else:
        k = remainder

    if bounderies:
        return tuple((cols, rows, k, low, high))
    else:
        return tuple((cols, rows, k))


def _convergence_plot(  # noqa: PLR0913
    subfigs: Any,
    elicits: tf.Tensor,
    span: int,
    label: str,
    parallel: bool,
    rows: int,
    cols: int,
    k: int,
    success: list[Any],
) -> Any:
    axes = subfigs.subplots(rows, cols)
    axes = axes.ravel()

    # Iterate over hyperparameters and axes
    for ax, n_hyp in zip(axes, range(elicits.shape[-1])):  # type: ignore
        if parallel:
            for i in success:
                # Compute mean of last span values (not used for plotting here)
                avg_hyp = tf.reduce_mean(elicits[i, -span:, n_hyp])
                # Plot convergence
                ax.plot(elicits[i, :, n_hyp], color="black", lw=2, alpha=0.5)
        else:
            avg_hyp = tf.reduce_mean(elicits[-span:, n_hyp])
            ax.axhline(avg_hyp.numpy(), color="darkgrey", linestyle="dotted")
            ax.plot(elicits[:, n_hyp], color="black", lw=2)

        # Formatting
        ax.set_title(rf"{label}($\theta_{n_hyp}$)", fontsize="small")
        ax.set_xlabel("epochs", fontsize="small")
        ax.tick_params(axis="y", labelsize="x-small")
        ax.tick_params(axis="x", labelsize="x-small")
        ax.grid(color="lightgrey", linestyle="dotted", linewidth=1)
        ax.spines[["right", "top"]].set_visible(False)

    # Turn off extra axes
    for k_idx in range(k):
        axes[-(k_idx + 1)].set_axis_off()
    return axes


def _check_NaN(eliobj: Any, n_reps: int) -> tuple[Any, ...]:
    # check whether some replications stopped with NAN
    ep_run = [len(eliobj.history[i]["loss"]) for i in range(n_reps)]
    seed_rep = [eliobj.results[i]["seed"] for i in range(n_reps)]
    # extract successful and failed seeds and indices for further plotting
    fail = []
    success = []
    success_name = []
    for i, ep in enumerate(ep_run):
        if ep < eliobj.trainer["epochs"]:
            fail.append((i, seed_rep[i]))
        else:
            success.append(i)
            success_name.append(seed_rep[i])
    if len(fail) > 0:
        logger.info(
            f"{len(fail)} of {n_reps} replications yield loss NAN and are"
            + f" excluded from the plot. Failed seeds: {fail} (index, seed)"
        )
    return (fail, success, success_name)


def _setup_grid(
    rows: int, cols: int, k: int = 0, **kwargs: Any
) -> tuple["matplotlib.figure.Figure", list["matplotlib.axes.Axes"]]:
    """
    Create a flattened grid of subplots and handles unused axes.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.
    k : int
        Number of unused axes to disable at the end.
    kwargs : dict
        Passed to `plt.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list[matplotlib.axes.Axes]
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    fig, axs = plt.subplots(rows, cols, **kwargs)
    axes = axs.ravel() if rows * cols > 1 else [axs]

    for i in range(k):
        axes[-(i + 1)].set_axis_off()

    return fig, axes
