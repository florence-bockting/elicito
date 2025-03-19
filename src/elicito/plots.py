"""
plotting helpers
"""

import itertools
from typing import Any, Callable, Optional, Union

import numpy as np
import tensorflow as tf

from elicito.exceptions import MissingOptionalDependencyError


def initialization(eliobj: Any, cols: int = 4, **kwargs: dict[Any, Any]) -> None:
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
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    try:
        import seaborn as sns
    except ImportError as exc:
        raise MissingOptionalDependencyError("plotting", requirement="seaborn") from exc

    eliobj_res, eliobj_hist, parallel, num_reps = _check_parallel(eliobj)
    # get number of hyperparameter
    n_par = len(eliobj_res["init_matrix"].keys())
    # prepare plot axes
    (cols, rows, k, low, high) = _prep_subplots(eliobj, cols, n_par, bounderies=True)

    # check that all information can be assessed
    try:
        eliobj_res["init_matrix"]
    except KeyError:
        print(
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

    # plot ecdf of initialiaztion distribution
    # differentiate between subplots that have (1) only one row vs.
    # (2) subplots with multiple rows

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, sharey=True, **kwargs)  # type: ignore
    if rows == 1:
        for c, hyp, lo, hi in zip(tf.range(cols), eliobj_res["init_matrix"], low, high):
            [
                sns.ecdfplot(
                    tf.squeeze(eliobj.results[j]["init_matrix"][hyp]),
                    ax=axs[c],
                    color="black",
                    lw=2,
                    alpha=0.5,
                )
                for j in range(len(eliobj.results))
            ]
            axs[c].set_title(f"{hyp}", fontsize="small")
            axs[c].axline((lo, 0), (hi, 1), color="grey", linestyle="dashed", lw=1)
            axs[c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[c].spines[["right", "top"]].set_visible(False)
            axs[c].tick_params(axis="y", labelsize="x-small")
            axs[c].tick_params(axis="x", labelsize="x-small")
        for k_idx in range(k):
            axs[cols - k_idx - 1].set_axis_off()
    else:
        for (r, c), hyp, lo, hi in zip(
            itertools.product(tf.range(rows), tf.range(cols)),
            eliobj_res["init_matrix"],
            low,
            high,
        ):
            [
                sns.ecdfplot(
                    tf.squeeze(eliobj.results[j]["init_matrix"][hyp]),
                    ax=axs[r, c],
                    color="black",
                    lw=2,
                    alpha=0.5,
                )
                for j in range(len(eliobj.results))
            ]
            axs[r, c].set_title(f"{hyp}", fontsize="small")
            axs[r, c].axline((lo, 0), (hi, 1), color="grey", linestyle="dashed", lw=1)
            axs[r, c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[r, c].spines[["right", "top"]].set_visible(False)
            axs[r, c].tick_params(axis="y", labelsize="x-small")
            axs[r, c].tick_params(axis="x", labelsize="x-small")
        for k_idx in range(k):
            axs[rows - 1, cols - k_idx - 1].set_axis_off()
    fig.suptitle("ecdf of initialization distributions", fontsize="medium")
    plt.show()


def loss(  # noqa: PLR0912
    eliobj: Any, weighted: bool = True, **kwargs: dict[Any, Any]
) -> None:
    """
    Plot the total loss and the loss per component.

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.
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
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

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
        fails, success, success_name = _check_NaN(eliobj, n_reps)
    else:
        success = [0]
    # check that all information can be assessed
    try:
        eliobj_hist["loss_component"]
    except KeyError:
        print(
            "No information about 'loss_component' found in 'eliobj.history'."
            + "Have you excluded 'loss_components' from history savings?"
        )
    try:
        eliobj_hist["loss"]
    except KeyError:
        print(
            "No information about 'loss' found in 'eliobj.history'."
            + "Have you excluded 'loss' from history savings?"
        )
    try:
        eliobj_res["elicited_statistics"]
    except KeyError:
        print(
            "No information about 'elicited_statistics' found in "
            + "'eliobj.results'. Have you excluded 'elicited_statistics' from"
            + "results savings?"
        )

    fig, axs = plt.subplots(1, 2, constrained_layout=True, sharex=True, **kwargs)  # type: ignore
    # plot total loss
    [
        axs[0].plot(eliobj.history[i]["loss"], color="black", alpha=0.5, lw=2)
        for i in success
    ]
    # plot loss per component
    for i, name in enumerate(names_losses):
        for j in success:
            # preprocess loss_component results
            indiv_losses = tf.stack(eliobj.history[j]["loss_component"])
            if j == 0:
                axs[1].plot(
                    indiv_losses[:, i] * weights[i], label=name, lw=2, alpha=0.5
                )
            else:
                axs[1].plot(indiv_losses[:, i] * weights[i], lw=2, alpha=0.5)
        axs[1].legend(fontsize="small", handlelength=0.4, frameon=False)
    [
        axs[i].set_title(t, fontsize="small")
        for i, t in enumerate(["total loss", in_title + "individual losses"])
    ]
    for i in range(2):
        axs[i].set_xlabel("epochs", fontsize="small")
        axs[i].grid(color="lightgrey", linestyle="dotted", linewidth=1)
        axs[i].spines[["right", "top"]].set_visible(False)
        axs[i].tick_params(axis="y", labelsize="x-small")
        axs[i].tick_params(axis="x", labelsize="x-small")
    plt.show()


def hyperparameter(eliobj: Any, cols: int = 4, **kwargs: dict[Any, Any]) -> None:
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
    >>> el.plots.hyperparameter(eliobj, figuresize=(8, 3))  # doctest: +SKIP

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

    eliobj_res, eliobj_hist, parallel, n_reps = _check_parallel(eliobj)
    # names of hyperparameter
    names_par = list(eliobj_hist["hyperparameter"].keys())
    # get number of hyperparameter
    n_par = len(names_par)
    # check chains that yield NaN
    if parallel:
        fails, success, success_name = _check_NaN(eliobj, n_reps)
    else:
        success = [0]
    # prepare subplot axes
    (cols, rows, k) = _prep_subplots(eliobj, cols, n_par, bounderies=False)

    # check that all information can be assessed
    try:
        eliobj_hist["hyperparameter"]
    except KeyError:
        print(
            "No information about 'hyperparameter' found in "
            + "'eliobj.history'. Have you excluded 'hyperparameter' from"
            + "history savings?"
        )

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, **kwargs)  # type: ignore
    if rows == 1:
        for c, hyp in zip(tf.range(cols), names_par):
            # plot convergence
            [
                axs[c].plot(
                    eliobj.history[i]["hyperparameter"][hyp],
                    color="black",
                    lw=2,
                    alpha=0.5,
                )
                for i in success
            ]
            axs[c].set_title(f"{hyp}", fontsize="small")
            axs[c].tick_params(axis="y", labelsize="x-small")
            axs[c].tick_params(axis="x", labelsize="x-small")
            axs[c].set_xlabel("epochs", fontsize="small")
            axs[c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[c].spines[["right", "top"]].set_visible(False)
        for k_idx in range(k):
            axs[cols - k_idx - 1].set_axis_off()
    else:
        for (r, c), hyp in zip(
            itertools.product(tf.range(rows), tf.range(cols)), names_par
        ):
            [
                axs[r, c].plot(
                    eliobj.history[i]["hyperparameter"][hyp],
                    color="black",
                    lw=2,
                    alpha=0.5,
                )
                for i in success
            ]
            axs[r, c].set_title(f"{hyp}", fontsize="small")
            axs[r, c].tick_params(axis="y", labelsize="x-small")
            axs[r, c].tick_params(axis="x", labelsize="x-small")
            axs[r, c].set_xlabel("epochs", fontsize="small")
            axs[r, c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[r, c].spines[["right", "top"]].set_visible(False)
        for k_idx in range(k):
            axs[rows - 1, cols - k_idx - 1].set_axis_off()
    fig.suptitle("Convergence of hyperparameter", fontsize="medium")
    plt.show()


def prior_joint(
    eliobj: Any, idx: Optional[Union[int, list[int]]] = None, **kwargs: dict[Any, Any]
) -> None:
    """
    Plot learned prior distributions

    Plot prior of each model parameter based on prior samples from last epoch.
    If parallelization has been used, select which replication you want to
    investigate by indexing it through the 'idx' argument.

    Parameters
    ----------
    eliobj : instance of :func:`elicit.elicit.Elicit`
        fitted ``eliobj`` object.
    idx : int or list or None:
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
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    try:
        import matplotlib as mpl
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    try:
        import seaborn as sns
    except ImportError as exc:
        raise MissingOptionalDependencyError("plotting", requirement="seaborn") from exc

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
        print(
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
            sns.kdeplot(priors[:, i], ax=axs[i, i], color=colors[c], lw=2)
            axs[i, i].set_xlabel(name_params[i], size="small")
            [axs[i, i].tick_params(axis=a, labelsize="x-small") for a in ["x", "y"]]
            axs[i, i].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[i, i].spines[["right", "top"]].set_visible(False)

        for i, j in itertools.combinations(range(n_params), 2):
            sns.kdeplot(priors[:, i], ax=axs[i, i], color=colors[c], lw=2)
            axs[i, j].plot(priors[:, i], priors[:, j], ",", color=colors[c], alpha=0.1)
            [axs[i, j].tick_params(axis=a, labelsize=7) for a in ["x", "y"]]
            axs[j, i].set_axis_off()
            axs[i, j].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[i, j].spines[["right", "top"]].set_visible(False)
    fig.suptitle("Learned joint prior", fontsize="medium")
    plt.show()


def prior_marginals(  # noqa: PLR0912
    eliobj: Any, cols: int = 4, **kwargs: dict[Any, Any]
) -> None:
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
    >>> el.plots.prior_marginals(eliobj, figuresize=(8, 3))  # doctest: +SKIP

    Raises
    ------
    KeyError
        Can't find 'prior_samples' in 'eliobj.results'. Have you excluded
        'prior_samples' from results savings?

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    try:
        import seaborn as sns
    except ImportError as exc:
        raise MissingOptionalDependencyError("plotting", requirement="seaborn") from exc

    eliobj_res, eliobj_hist, parallel, n_reps = _check_parallel(eliobj)
    # check chains that yield NaN
    if parallel:
        fails, success, success_name = _check_NaN(eliobj, n_reps)
    else:
        success = [0]
    # get shape of prior samples
    B, n_samples, n_par = eliobj_res["prior_samples"].shape
    # get parameter names
    name_params = [eliobj.parameters[i]["name"] for i in range(n_par)]
    # prepare plot axes
    (cols, rows, k) = _prep_subplots(eliobj, cols, n_par, bounderies=False)

    # check that all information can be assessed
    try:
        eliobj_res["prior_samples"]
    except KeyError:
        print(
            "No information about 'prior_samples' found in "
            + "'eliobj.results'. Have you excluded 'prior_samples' from"
            + "results savings?"
        )

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, **kwargs)  # type: ignore
    if rows == 1:
        for c, par in zip(tf.range(cols), name_params):
            for i in success:
                # reshape samples by merging batches and number of samples
                priors = tf.reshape(
                    eliobj.results[i]["prior_samples"], (B * n_samples, n_par)
                )
                sns.kdeplot(priors[:, c], ax=axs[c], color="black", lw=2, alpha=0.5)

            axs[c].set_title(f"{par}", fontsize="small")
            axs[c].tick_params(axis="y", labelsize="x-small")
            axs[c].tick_params(axis="x", labelsize="x-small")
            axs[c].set_xlabel(r"$\theta$", fontsize="small")
            axs[c].set_ylabel("density", fontsize="small")
            axs[c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[c].spines[["right", "top"]].set_visible(False)
        for k_idx in range(k):
            axs[cols - k_idx - 1].set_axis_off()
    else:
        for j, ((r, c), par) in enumerate(
            zip(itertools.product(tf.range(rows), tf.range(cols)), name_params)
        ):
            for i in success:
                # reshape samples by merging batches and number of samples
                priors = tf.reshape(
                    eliobj.results[i]["prior_samples"], (B * n_samples, n_par)
                )
                sns.kdeplot(priors[:, j], ax=axs[r, c], color="black", lw=2, alpha=0.5)
            axs[r, c].set_title(f"{par}", fontsize="small")
            axs[r, c].tick_params(axis="y", labelsize="x-small")
            axs[r, c].tick_params(axis="x", labelsize="x-small")
            axs[r, c].set_xlabel(r"$\theta$", fontsize="small")
            axs[r, c].set_ylabel("density", fontsize="small")
            axs[r, c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[r, c].spines[["right", "top"]].set_visible(False)
        for k_idx in range(k):
            axs[rows - 1, cols - k_idx - 1].set_axis_off()
    fig.suptitle("Learned marginal priors", fontsize="medium")
    plt.show()


def elicits(  # noqa: PLR0912, PLR0915
    eliobj: Any, cols: int = 4, **kwargs: dict[Any, Any]
) -> None:
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
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    # check whether parallelization has been used
    eliobj_res, eliobj_hist, parallel, n_reps = _check_parallel(eliobj)
    # get number of hyperparameter
    n_elicits = len(eliobj_res["expert_elicited_statistics"].keys())
    # check chains that yield NaN
    if parallel:
        fails, success, success_name = _check_NaN(eliobj, n_reps)
    else:
        success = [0]
    # prepare plot axes
    (cols, rows, k) = _prep_subplots(eliobj, cols, n_elicits, bounderies=False)
    # extract quantities of interest needed for plotting
    name_elicits = list(eliobj_res["expert_elicited_statistics"].keys())
    method_name = [name_elicits[i].split("_")[0] for i in range(n_elicits)]

    # check that all information can be assessed
    try:
        eliobj_res["expert_elicited_statistics"]
    except KeyError:
        print(
            "No information about 'expert_elicited_statistics' found in "
            + "'eliobj.results'. Have you excluded 'expert_elicited_statistics'"
            + " from results savings?"
        )
    try:
        eliobj_res["elicited_statistics"]
    except KeyError:
        print(
            "No information about 'elicited_statistics' found in "
            + "'eliobj.results'. Have you excluded 'elicited_statistics'"
            + " from results savings?"
        )

    # plotting
    fig, axs = plt.subplots(rows, cols, constrained_layout=True, **kwargs)  # type: ignore
    if rows == 1:
        labels: list[Any]
        method: Callable[[Any], Any]

        for c, (elicit, meth) in enumerate(zip(name_elicits, method_name)):
            if meth == "quantiles":
                labels = [None] * n_reps
                prep = (
                    axs[c].axline(
                        (0, 0), slope=1, color="darkgrey", linestyle="dashed", lw=1
                    ),
                )
                method = _quantiles  # type: ignore

            elif meth == "cor":
                # prepare labels for plotting
                labels = [("expert", "train")] + [
                    (None, None) for i in range(n_reps - 1)
                ]
                # select method function
                method = _correlation  # type: ignore
                # get number of correlations
                num_cor = eliobj_res["elicited_statistics"][elicit].shape[-1]
                prep = (
                    axs[c].set_ylim(-1, 1),
                    axs[c].set_xlim(-0.5, num_cor),
                    axs[c].set_xticks(
                        [i for i in range(num_cor)],
                        [f"cor{i}" for i in range(num_cor)],
                    ),  # type: ignore
                )

            for i in success:
                (
                    method(
                        axs[c],
                        eliobj.results[i]["expert_elicited_statistics"][elicit],
                        eliobj.results[i]["elicited_statistics"][elicit],
                        labels[i],
                    )  # type: ignore
                    + prep
                )

            if elicit.endswith("_cor"):
                axs[c].legend(fontsize="x-small", markerscale=0.5, frameon=False)
            axs[c].set_title(elicit, fontsize="small")
            axs[c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[c].spines[["right", "top"]].set_visible(False)
            axs[c].tick_params(axis="y", labelsize="x-small")
            axs[c].tick_params(axis="x", labelsize="x-small")
            if not elicit.endswith("_cor"):
                axs[c].set_xlabel("expert", fontsize="small")
                axs[c].set_ylabel("model-sim.", fontsize="small")
        for k_idx in range(k):
            axs[cols - k_idx - 1].set_axis_off()

    else:
        for (r, c), elicit, meth in zip(
            itertools.product(tf.range(rows), tf.range(cols)), name_elicits, method_name
        ):
            if meth == "quantiles":
                labels = [None] * n_reps
                prep = (
                    axs[r, c].axline(
                        (0, 0), slope=1, color="darkgrey", linestyle="dashed", lw=1
                    ),
                )
                method = _quantiles  # type: ignore

            if meth == "cor":
                labels = [("expert", "train")] + [
                    (None, None) for i in range(n_reps - 1)
                ]
                method = _correlation  # type: ignore
                num_cor = eliobj_res["elicited_statistics"][elicit].shape[-1]
                prep = (  # type: ignore
                    axs[r, c].set_ylim(-1, 1),
                    axs[r, c].set_xlim(-0.5, num_cor),
                    axs[r, c].set_xticks(
                        [i for i in range(num_cor)],
                        [f"cor{i}" for i in range(num_cor)],
                    ),
                )

            for i in success:
                (
                    method(
                        axs[r, c],
                        eliobj.results[i]["expert_elicited_statistics"][elicit],
                        eliobj.results[i]["elicited_statistics"][elicit],
                        labels[i],
                    )  # type: ignore
                    + prep
                )

            if elicit.endswith("_cor"):
                axs[r, c].legend(fontsize="x-small", markerscale=0.5, frameon=False)
            axs[r, c].set_title(elicit, fontsize="small")
            axs[r, c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[r, c].spines[["right", "top"]].set_visible(False)
            axs[r, c].tick_params(axis="y", labelsize="x-small")
            axs[r, c].tick_params(axis="x", labelsize="x-small")
            if not elicit.endswith("_cor"):
                axs[r, c].set_xlabel("expert", fontsize="small")
                axs[r, c].set_ylabel("model-sim.", fontsize="small")
        for k_idx in range(k):
            axs[rows - 1, cols - k_idx - 1].set_axis_off()

    fig.suptitle("Expert vs. model-simulated elicited statistics", fontsize="medium")
    plt.show()


def marginals(
    eliobj: Any, cols: int = 4, span: int = 30, **kwargs: dict[Any, Any]
) -> None:
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
    >>> el.plots.marginals(eliobj, figuresize=(8, 3))  # doctest: +SKIP

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
    (eliobj_res, eliobj_hist, parallel, n_reps) = _check_parallel(eliobj)
    # check chains that yield NaN
    if parallel:
        fails, success, success_name = _check_NaN(eliobj, n_reps)
    else:
        success = [0]
    # number of marginals
    n_elicits = tf.stack(eliobj_hist["hyperparameter"]["means"]).shape[-1]
    # prepare plot axes
    cols, rows, k = _prep_subplots(eliobj, cols, n_elicits, bounderies=False)
    # check that all information can be assessed
    try:
        eliobj_hist["hyperparameter"]
    except KeyError:
        print(
            "No information about 'hyperparameter' found in 'eliobj.history'"
            + " Have you excluded 'hyperparameter' from history savings?"
        )

    elicits_means = tf.stack(
        [eliobj.history[i]["hyperparameter"]["means"] for i in success]
    )
    elicits_std = tf.stack(
        [eliobj.history[i]["hyperparameter"]["stds"] for i in success]
    )

    fig = plt.figure(layout="constrained", **kwargs)  # type: ignore
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
    plt.show()


def priorpredictive(eliobj: Any, **kwargs: dict[Any, Any]) -> None:
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
    >>> el.plots.priorpredictive(eliobj, figuresize=(6, 2))  # doctest: +SKIP

    Raises
    ------
    KeyError
        Can't find 'target_quantities' in 'eliobj.results'. Have you excluded
        'target_quantities' from results savings?

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    try:
        import seaborn as sns
    except ImportError as exc:
        raise MissingOptionalDependencyError("plotting", requirement="seaborn") from exc

    # check that all information can be assessed
    try:
        eliobj.results["target_quantities"]
    except KeyError:
        msg = (
            "No information about 'target_quantities' found in 'eliobj.results'",
            " Have you excluded 'target_quantities' from results savings?",
        )
        print(msg)

    target_reshaped = []
    for k in eliobj.results["target_quantities"]:
        target = eliobj.results["target_quantities"][k]
        target_reshaped.append(tf.reshape(target, (target.shape[0] * target.shape[1])))

    targets = tf.stack(target_reshaped, -1)

    fig, axs = plt.subplots(1, 1, constrained_layout=True, **kwargs)  # type: ignore
    axs.grid(color="lightgrey", linestyle="dotted", linewidth=1)
    for i in range(targets.shape[-1]):
        shade = i / (targets.shape[-1] - 1)
        color = plt.cm.gray(shade)  # type: ignore
        sns.histplot(
            targets[:, i],
            stat="probability",
            bins=40,
            label=eliobj.targets[i]["name"],
            ax=axs,
            color=color,
        )
    plt.legend(fontsize="small", handlelength=0.9, frameon=False)
    axs.set_title("prior predictive distribution", fontsize="small")
    axs.spines[["right", "top"]].set_visible(False)
    axs.tick_params(axis="y", labelsize="x-small")
    axs.tick_params(axis="x", labelsize="x-small")
    axs.set_xlabel(r"$y_{pred}$", fontsize="small")
    plt.show()


def prior_averaging(  # noqa: PLR0912, PLR0913, PLR0915
    eliobj: Any,
    cols: int = 4,
    n_sim: int = 10_000,
    height_ratio: list[Union[int, float]] = [1, 1.5],
    weight_factor: float = 1.0,
    seed: int = 123,
    **kwargs: dict[Any, Any],
) -> None:
    """
    Plot prior averaging

    Parameters
    ----------
    eliobj:
        instance of :func:`elicit.elicit.Elicit`

    cols:
        number of columns in plot

    n_sim:
        number of simulations

    height_ratio:
        height ratio of prior averaging plot

    weight_factor:
        weighting factor of each model in prior averaging

    kwargs:
        additional arguments passed to matplotlib
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    try:
        import seaborn as sns
    except ImportError as exc:
        raise MissingOptionalDependencyError("plotting", requirement="seaborn") from exc

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
    sns.barplot(
        y="seed",
        x="weight",
        data=df_sorted,
        ax=subfig0,
        color="darkgrey",
        orient="h",
        order=df_sorted["seed"],
    )
    subfig0.spines[["right", "top"]].set_visible(False)
    subfig0.grid(color="lightgrey", linestyle="dotted", linewidth=1)
    subfig0.set_xlabel("weight", fontsize="small")
    subfig0.set_ylabel("seed", fontsize="small")
    subfig0.tick_params(axis="y", labelsize="x-small")
    subfig0.tick_params(axis="x", labelsize="x-small")
    subfig0.set_xlim(0, df_sorted["weight"][0] + 0.1)

    # plot individual priors and averaged prior
    if rows == 1:
        for c, par, lab in zip(tf.range(cols), name_par, label_avg):
            for i in success:
                # reshape samples by merging batches and number of samples
                prior = tf.reshape(
                    eliobj.results[i]["prior_samples"], (B * n_samples, n_par)
                )
                sns.kdeplot(prior[:, c], ax=subfig1[c], color="black", lw=2, alpha=0.5)
            avg_prior = tf.reshape(averaged_priors, (B * n_sim, n_par))
            sns.kdeplot(avg_prior[:, c], color="red", ax=subfig1[c], label=lab)
            subfig1[c].set_title(f"{par}", fontsize="small")
            subfig1[c].tick_params(axis="y", labelsize="x-small")
            subfig1[c].tick_params(axis="x", labelsize="x-small")
            subfig1[c].set_xlabel(r"$\theta$", fontsize="small")
            subfig1[c].set_ylabel("density", fontsize="small")
            subfig1[c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            subfig1[c].spines[["right", "top"]].set_visible(False)
            subfig1[c].legend(handlelength=0.3, fontsize="small", frameon=False)
        for k_idx in range(k):
            subfig1[cols - k_idx - 1].set_axis_off()
    else:
        for j, ((r, c), par, lab) in enumerate(
            zip(itertools.product(tf.range(rows), tf.range(cols)), name_par, label_avg)
        ):
            for i in success:
                # reshape samples by merging batches and number of samples
                priors = tf.reshape(
                    eliobj.results[i]["prior_samples"], (B * n_samples, n_par)
                )
                sns.kdeplot(
                    priors[:, j], ax=subfig1[r, c], color="black", lw=2, alpha=0.5
                )
            avg_prior = tf.reshape(averaged_priors, (B * n_sim, n_par))
            sns.kdeplot(avg_prior[:, j], color="red", ax=subfig1[r, c], label=lab)
            subfig1[r, c].set_title(f"{par}", fontsize="small")
            subfig1[r, c].tick_params(axis="y", labelsize="x-small")
            subfig1[r, c].tick_params(axis="x", labelsize="x-small")
            subfig1[r, c].set_xlabel(r"$\theta$", fontsize="small")
            subfig1[r, c].set_ylabel("density", fontsize="small")
            subfig1[r, c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            subfig1[r, c].spines[["right", "top"]].set_visible(False)
            subfig1[r, c].legend(handlelength=0.3, fontsize="small", frameon=False)
        for k_idx in range(k):
            subfig1[rows - 1, cols - k_idx - 1].set_axis_off()
    subfigs[0].suptitle("Prior averaging (weights)", fontsize="small", ha="left", x=0.0)
    subfigs[1].suptitle("Prior distributions", fontsize="small", ha="left", x=0.0)
    fig.suptitle("Prior averaging", fontsize="medium")
    plt.show()


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
        print(f"INFO: Reset cols={cols}")
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
    axs = subfigs.subplots(rows, cols)
    if rows == 1:
        for c, n_hyp in zip(tf.range(cols), tf.range(elicits.shape[-1])):
            if parallel:
                for i in success:
                    # compute mean of last c hyperparameter values
                    avg_hyp = tf.reduce_mean(elicits[i, -span:, n_hyp])
                    # plot convergence
                    axs[c].plot(elicits[i, :, n_hyp], color="black", lw=2, alpha=0.5)
            else:
                # compute mean of last c hyperparameter values
                avg_hyp = tf.reduce_mean(elicits[-span:, n_hyp])
                axs[c].axhline(avg_hyp.numpy(), color="darkgrey", linestyle="dotted")
                # plot convergence
                axs[c].plot(elicits[:, n_hyp], color="black", lw=2)
            axs[c].set_title(rf"{label}($\theta_{n_hyp}$)", fontsize="small")
            axs[c].tick_params(axis="y", labelsize="x-small")
            axs[c].tick_params(axis="x", labelsize="x-small")
            axs[c].set_xlabel("epochs", fontsize="small")
            axs[c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[c].spines[["right", "top"]].set_visible(False)
        for k_idx in range(k):
            axs[cols - k_idx - 1].set_axis_off()
    else:
        for (r, c), n_hyp in zip(
            itertools.product(tf.range(rows), tf.range(cols)),
            tf.range(elicits.shape[-1]),
        ):
            if parallel:
                for i in success:
                    # compute mean of last c hyperparameter values
                    avg_hyp = tf.reduce_mean(elicits[i, -span:, n_hyp])
                    # plot convergence
                    axs[r, c].plot(elicits[i, :, n_hyp], color="black", lw=2)
            else:
                # compute mean of last c hyperparameter values
                avg_hyp = tf.reduce_mean(elicits[-span:, n_hyp])
                # plot convergence
                axs[r, c].axhline(avg_hyp.numpy(), color="darkgrey", linestyle="dotted")
                axs[r, c].plot(elicits[:, n_hyp], color="black", lw=2)
            axs[r, c].set_title(rf"$\theta_{n_hyp}$", fontsize="small")
            axs[r, c].tick_params(axis="y", labelsize="x-small")
            axs[r, c].tick_params(axis="x", labelsize="x-small")
            axs[r, c].set_xlabel("epochs", fontsize="small")
            axs[r, c].grid(color="lightgrey", linestyle="dotted", linewidth=1)
            axs[r, c].spines[["right", "top"]].set_visible(False)
        for k_idx in range(k):
            axs[rows - 1, cols - k_idx - 1].set_axis_off()
    return axs


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
        print(
            f"INFO: {len(fail)} of {n_reps} replications yield loss NAN and are"
            + f" excluded from the plot. Failed seeds: {fail} (index, seed)"
        )
    return (fail, success, success_name)
