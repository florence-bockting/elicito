from collections.abc import Iterable
from typing import Optional

import numpy as np
import tensorflow as tf
import xarray as xr

MAIN_DIMS = ["replication", "epoch"]
"""main dimensions for history result objects"""


def create_hist_corrds(eliobj) -> dict[str, Iterable]:
    """Create coordinates for dim: replication, epoch

    Parameters
    ----------
    eliobj :
        fitted eliobj

    Returns
    -------
    :
       mapping for coords property of xr.Dataset
    """
    return dict(
        replication=range(len(eliobj.history)),
        epoch=range(len(eliobj.history[0]["loss"])),
    )


def combine_reps(obj, group: str) -> tf.Tensor:
    """
    Stack replication dimension

    Helper function to handle the raw nested dictionary output

    Parameters
    ----------
    obj :
        fitted eliobj

    group :
        corresponds to key for selecting result in
        eliobj

    Returns
    -------
    tf.Tensor
        stacked tensor with shape (replication, ...)
    """
    return tf.stack([obj[i][group] for i in range(len(obj))])


def to_dataarray(
    obj, group: str, dims: list[str], name: str, attrs: Optional[dict[str, str]] = None
) -> xr.DataArray:
    """
    Create xr.DataArray for eliobj result

    Parameters
    ----------
    obj :
        fitted eliobj

    group :
        corresponds to key for selecting result in
        eliobj

    dims :
        dimensions of array (e.g., [replication, epoch])

    name :
        name that should appear in the dataset

    attrs :
        optional attributes for the dataarray provided
        as a dictionary

    Returns
    -------
    :
        xr.DataArray from eliobj result dictionary
    """
    obj_reps = combine_reps(obj, group)
    return xr.DataArray(data=np.stack(obj_reps), dims=dims, name=name, attrs=attrs)


def to_dataset(obj, group: str, dims: list[str], names_subgroups: str) -> xr.Dataset:
    """
    Create a xr.Dataset from eliobj results

    Parameters
    ----------
    obj :
        fitted eliobj

    group :
        key corresponding to section in eliobj dictionary

    dims :
        dimensions of array (e.g., [replication, epoch])

    names_subgroups :
        name of single data variables in xr.Dataset

    Returns
    -------
    :
        xr.Dataset including subgroups as data variables
    """
    ds = xr.Dataset()
    obj_reps = combine_reps(obj, group)
    for i, name in enumerate(names_subgroups):
        ds[name] = xr.DataArray(data=np.stack(obj_reps)[..., i], dims=dims, name=name)
    return ds


def create_initialization_group(eliobj) -> xr.Dataset:
    """
    Create result group for initialization runs

    Parameters
    ----------
    eliobj :
        fitted eliobj

    Returns
    -------
    :
        xr.Dataset including information about initial
        hyperparameter values and corresponding loss per
        iteration.
    """
    init_loss = combine_reps(eliobj.results, "init_loss_list")

    hyp_names = [
        eliobj.parameters[i]["hyperparams"][k]["name"]
        for i in range(len(eliobj.parameters))
        for k in eliobj.parameters[i]["hyperparams"]
    ]

    init_hyp = xr.DataArray(
        data=tf.stack(
            [
                tf.stack(
                    [
                        eliobj.results[i]["init_matrix"][k]
                        for i in range(len(eliobj.results))
                    ]
                )
                for k in eliobj.results[0]["init_matrix"].keys()
            ],
            -1,
        ),
        dims=["replication", "iteration", "hyperparameter"],
        coords=dict(
            replication=range(len(eliobj.results)),
            iteration=range(len(eliobj.results[0]["init_loss_list"])),
            hyperparameter=hyp_names,
        ),
        name="hyperparameter",
        attrs=dict(description="Initial hyperparameter values per iteration."),
    )

    init_loss = xr.DataArray(
        data=init_loss[..., 0],
        dims=["replication", "iteration"],
        coords=dict(
            replication=range(len(eliobj.results)),
            iteration=range(len(eliobj.results[0]["init_loss_list"])),
        ),
        name="loss",
        attrs=dict(
            description=(
                "Total loss value corresponding to sampled "
                "hyperparameter vector per iteration."
            )
        ),
    )

    init = xr.Dataset()
    init["loss"] = init_loss
    init["hyperparameters"] = init_hyp
    init = init.assign_attrs(
        dict(
            description=(
                "Initial hyperparameter vectors and their corresponding "
                "losses for each iteration during initialization. "
                "After evaluating all iterations, the hyperparameter vector "
                "yielding the minimum loss is selected as initial values."
            )
        )
    )
    return init


def create_hyperparameter_group(eliobj) -> xr.Dataset:
    """Create xr.Dataset from hyperparameter results in eliobj

    Parameters
    ----------
    eliobj :
        fitted eliobj

    Returns
    -------
    :
        xr.Dataset with variables corresponding to
        hyperparameter values and their gradients per epoch
    """
    hyp_names = [
        eliobj.parameters[i]["hyperparams"][k]["name"]
        for i in range(len(eliobj.parameters))
        for k in eliobj.parameters[i]["hyperparams"]
    ]

    obj_hyp = tf.stack(
        [
            tf.stack(
                [
                    eliobj.history[i]["hyperparameter"][k][1:]
                    for i in range(len(eliobj.history))
                ]
            )
            for k in eliobj.history[0]["hyperparameter"]
        ],
        -1,
    )

    ds_hyp = xr.Dataset(
        {
            k: xr.DataArray(data=obj_hyp[..., i], dims=MAIN_DIMS)
            for i, k in enumerate(hyp_names)
        }
    )

    ds_hyp_grad = to_dataset(
        obj=eliobj.history,
        group="hyperparameter_gradient",
        dims=MAIN_DIMS,
        names_subgroups=[f"grad_{k}" for k in hyp_names],
    )
    hyp_group = ds_hyp.merge(ds_hyp_grad)

    hyp_group = hyp_group.assign_coords(create_hist_corrds(eliobj))
    hyp_group = hyp_group.assign_attrs(
        {
            "description": (
                "Update of model hyperparameters and their gradients "
                "across epochs and per replication."
            )
        }
    )
    return hyp_group


def create_marginal_group(eliobj) -> xr.Dataset:
    """
    Create xr.Dataset from marginal prior updates

    Summaries information about mean and standard deviation
    for each marginal prior distribution based on prior
    samples across epochs.

    Parameters
    ----------
    eliobj :
        fitted eliobj

    Returns
    -------
    :
        xr.Dataset including information about mean and sd
        of the marginal priors across epochs
    """
    param_names = [eliobj.parameters[i]["name"] for i in range(len(eliobj.parameters))]

    marginal_group = xr.Dataset()
    for m in ["means", "stds"]:
        marginal_group[m[:-1]] = xr.DataArray(
            data=tf.stack(
                [
                    eliobj.history[i]["hyperparameter"][m]
                    for i in range(len(eliobj.history))
                ]
            ),
            dims=["replication", "epoch", "parameter"],
            coords=dict(
                replication=range(len(eliobj.history)),
                epoch=range(len(eliobj.history[0]["loss"])),
                parameter=param_names,
            ),
            name=m[:-1],
        )

    marginal_group = marginal_group.assign_attrs(
        {
            "description": (
                "Updates of mean and standard deviation of the marginal "
                "prior distribution for each model parameter across epochs. "
                "Computations are based on samples from the prior distributions."
            )
        }
    )
    return marginal_group


def create_loss_group(eliobj) -> xr.Dataset:
    """
    Create xr.Dataset for loss section

    loss-xr.Dataset incl. total loss value as well
    single loss components per epoch

    Parameters
    ----------
    eliobj :
        fitted eliobj

    Returns
    -------
    :
        xr.Dataset including information about total loss
        and loss components per epoch
    """
    ds_loss = xr.Dataset()
    ds_loss["total_loss"] = to_dataarray(
        obj=eliobj.history,
        group="loss",
        dims=MAIN_DIMS,
        name="loss",
    )

    ds_loss_comp = to_dataset(
        obj=eliobj.history,
        group="loss_component",
        dims=MAIN_DIMS,
        names_subgroups=eliobj.results[0]["loss_tensor_model"].keys(),
    )
    loss_group = ds_loss.merge(ds_loss_comp)

    loss_group = loss_group.assign_coords(create_hist_corrds(eliobj))
    loss_group = loss_group.assign_attrs(
        {
            "description": (
                "Update of total loss and single loss components"
                " across epochs and for each replication."
            )
        }
    )
    return loss_group


def create_result_group(
    eliobj,
    group: str,
    description: str,
    dim_name: Optional[str] = None,
    base_dims: list[str] = ["replication", "batch", "draw"],
) -> xr.Dataset:
    """
    Build an xarray.Dataset from eliobj results for a given group.

    Parameters
    ----------
    eliobj :
        Object containing results with structure eliobj.results[rep][group][var].

    group :
        Name of the group to extract (e.g. "model_samples").

    description :
        Description to attach to each DataArray.

    dim_name :
        prefix used to name dimensions additional to base dimensions.

    base_dims :
        Dimension names for the first axes (default: ["replication","batch","draw"]).

    Returns
    -------
    :
        Dataset containing one DataArray per variable in the group.
    """
    ds_group = xr.Dataset(attrs=dict(description=description))

    n_replications = len(eliobj.results)
    for num, (k, _) in enumerate(eliobj.results[0][group].items()):
        # stack over replications
        var = tf.stack([eliobj.results[i][group][k] for i in range(n_replications)])
        shape = var.shape

        # separate base dims and extra dims
        n_extra = len(shape) - len(base_dims)
        extra_dims = [
            f"{dim_name}{num}_dim{j}" for j in range(n_extra)
        ]  # unique per variable
        dims = base_dims + extra_dims

        # coords for base dims
        coords = {dim: tf.range(shape[i]) for i, dim in enumerate(base_dims)}

        # coords for extra dims
        for j, dim in enumerate(extra_dims):
            coords[dim] = tf.range(shape[len(base_dims) + j])

        da = xr.DataArray(data=var, dims=dims, coords=coords, name=k)

        ds_group[k] = da

    return ds_group


def create_prior_ds(eliobj) -> xr.Dataset:
    """Create prior group for Inference data

    Parameters
    ----------
    eliobj :
        eliobj containing results section with training information
        about prior samples

    Returns
    -------
    :
        dataset containing prior samples per model parameter
    """
    ds_prior = xr.Dataset(
        attrs={
            "description": (
                "Samples drawn from the model parameter's "
                "prior distribution in the last epoch."
            )
        }
    )
    for j, k in enumerate(
        [eliobj.parameters[k]["name"] for k in range(len(eliobj.parameters))]
    ):
        prior = tf.stack(
            [
                eliobj.results[i]["prior_samples"][:, :, j]
                for i in range(len(eliobj.results))
            ]
        )

        da_prior = xr.DataArray(
            data=prior,
            dims=["replication", "batch", "draw"],
            coords=dict(
                replication=tf.range(prior.shape[0]),
                batch=tf.range(prior.shape[1]),
                draw=tf.range(prior.shape[2]),
            ),
            name=k,
        )

        ds_prior[k] = da_prior

    return ds_prior


def create_oracle_ds(eliobj) -> xr.Dataset:
    """Create oracle group for Inference data

    Parameters
    ----------
    eliobj :
        eliobj containing results section with training information
        about ground truth containt prior_samples and elicited_summaries
        used for learning

    Returns
    -------
    :
        xr.Dataset containing oracle information
    """
    ds_oracle = xr.Dataset(
        attrs={
            "description": (
                "The concept of an 'oracle' refers to a situation in"
                " which a true prior distribution has been specified "
                "via el.expert.simulator (representing the ground truth). "
                "In this case, the method is run once in forward mode by "
                "sampling from the true prior, simulating from the generative"
                " model, and computing a set of 'true' elicited summaries."
                " These 'true' elicited summaries are then incorporated as "
                "expert information in the model. This procedure is useful for"
                " assessing the self-consistency of the algorithm and the "
                "informativeness of the elicited summaries wrt model hyperparameters."
            )
        }
    )
    priors_oracle = tf.stack(
        [eliobj.results[i]["expert_prior_samples"] for i in range(len(eliobj.results))],
        0,
    )

    da_priors_oracle = xr.DataArray(
        data=priors_oracle,
        dims=["replication", "batch", "draw", "parameter"],
        coords=dict(
            replication=tf.range(priors_oracle.shape[0]),
            batch=tf.range(priors_oracle.shape[1]),
            draw=tf.range(priors_oracle.shape[2]),
            parameter=[
                eliobj.parameters[k]["name"] for k in range(len(eliobj.parameters))
            ],
        ),
        name="prior samples",
        attrs=dict(description="Prior samples from ground truth (oracle)"),
    )

    ds_oracle["prior"] = da_priors_oracle

    ds_elicit = create_result_group(
        eliobj,
        group="expert_elicited_statistics",
        description="Expert-elicited summaries",
        dim_name="summary",
        base_dims=["replication", "batch"],
    )

    return xr.merge([ds_oracle, ds_elicit])


def create_expert_ds(eliobj) -> xr.Dataset:
    """
    Create expert group

    Only used if expert information is provided
    via el.expert.data.

    Parameters
    ----------
    eliobj :
        fitted eliobj

    Returns
    -------
    :
        xr.Dataset including information about the
        expert-elicited summaries
    """
    ds_elicit = create_result_group(
        eliobj,
        group="expert_elicited_statistics",
        description=(
            "Expert-elicited summaries used to train " "the optimization algorithm."
        ),
        dim_name="summary",
        base_dims=["replication", "batch"],
    )

    return ds_elicit


def create_datatree(eliobj) -> xr.DataTree:
    """
    Create data tree as final result object

    Parameters
    ----------
    eliobj :
        fitted eliobj containing training information
        in nested dict format incl. history and result
        attribute

    Returns
    -------
    :
        xr.DataTree with final results
    """
    # Create the base DataTree
    res = xr.DataTree(name="results")

    # Create datasets for history_stats group
    coords = create_hist_corrds(eliobj)

    # loss information across epochs
    loss_ds = create_loss_group(eliobj)

    # time per epoch
    time_ds = to_dataarray(
        eliobj.history,
        "time",
        MAIN_DIMS,
        "time_epoch",
        {
            "description": "Elapsed time (i.e., wall time) per epoch in seconds.",
            "unit": "seconds",
        },
    ).to_dataset()

    # seed per replication
    seed_ds = to_dataarray(
        eliobj.results,
        "seed",
        ["replication"],
        "seed_replication",
        {"description": "The seed used for each replication."},
    ).to_dataset()

    time_seed_ds = time_ds.merge(seed_ds)

    # hyperparameter or mean and sd for each marginal prior
    # dependening on the used method
    history_dict = dict()
    history_dict["loss"] = xr.DataTree(loss_ds.assign_coords(coords))
    if eliobj.trainer["method"] == "deep_prior":
        marginal_ds = create_marginal_group(eliobj)
        history_dict["prior_marginal"] = xr.DataTree(marginal_ds)
    else:
        hyp_ds = create_hyperparameter_group(eliobj)
        history_dict["hyperparameter"] = xr.DataTree(hyp_ds.assign_coords(coords))

    # combine history information in a tree
    history_stats = xr.DataTree(
        time_seed_ds.assign_coords(coords), children=history_dict
    )

    # Create datasets for remaining groups
    prior_ds = create_prior_ds(eliobj)
    model_ds = create_result_group(
        eliobj,
        group="model_samples",
        dim_name="model",
        description=(
            "Simulated quantities returned from the user-specified"
            " generative model in the last epoch."
        ),
    )
    target_ds = create_result_group(
        eliobj,
        group="target_quantities",
        dim_name="target",
        description=(
            "Simulated target quantities as specified in eliobj.targets."
            " Simulated target quantities refer either directly to returned "
            "values from the user-specified generative model or are computed"
            " from them via a custom target-function."
        ),
    )
    elicit_ds = create_result_group(
        eliobj,
        group="elicited_statistics",
        dim_name="summary",
        description=(
            "Simulated elicited summaries. This data structure should match "
            "the expert-elicited summaries. The quantities are computed by "
            "applying a summary function (cf. elicitation technique) to the "
            "target quantities. A commonly used summary function is the "
            "computation of quantiles."
        ),
        base_dims=["replication", "batch"],
    )

    # create final results object
    res = res.assign({"history_stats": history_stats})
    res = res.assign({"prior": xr.DataTree(prior_ds)})
    res = res.assign({"model": xr.DataTree(model_ds)})
    res = res.assign({"target_quantity": xr.DataTree(target_ds)})
    res = res.assign({"elicited_summary": xr.DataTree(elicit_ds)})

    try:
        eliobj.expert["ground_truth"]
    except KeyError:
        expert_ds = create_expert_ds(eliobj)
        res = res.assign({"expert": xr.DataTree(expert_ds)})
    else:
        oracle_ds = create_oracle_ds(eliobj)
        res = res.assign({"oracle": xr.DataTree(oracle_ds)})

    if (eliobj.trainer["method"] == "parametric_prior") and (
        eliobj.results[0]["init_loss_list"] is not None
    ):
        init_ds = create_initialization_group(eliobj)
        res = res.assign({"initialization": xr.DataTree(init_ds)})

    return res
