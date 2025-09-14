from typing import Optional

import arviz as az
import tensorflow as tf
import xarray as xr


def history_to_xarray(eliobj) -> xr.Dataset:
    """
    Convert an Elicit history object into an xarray.Dataset.

    Parameters
    ----------
    eliobj :
        Object containing training history as dict with keys:
        - loss
        - loss_component
        - time
        - hyperparameter
        - hyperparameter_gradient

    Returns
    -------
    :
        Dataset with loss, loss components, time, hyperparameters, and gradients.
    """
    history = eliobj.history
    n_rep = len(history)

    # --- Loss ---
    loss = tf.squeeze(
        tf.stack([history[i]["loss"] for i in range(len(eliobj.history))])
    )

    da_loss = xr.DataArray(
        loss,
        dims=["replication", "epoch"],
        coords=dict(
            replication=tf.range(n_rep),
            epoch=tf.range(loss.shape[1]),
        ),
        name="loss",
        attrs=dict(description="Total loss value"),
    )

    # --- Loss components ---
    loss_components = tf.stack(
        [history[i]["loss_component"] for i in range(len(history))]
    )
    component_names = [
        "_".join(k.split("_")[:-2])
        for k in eliobj.results[0]["loss_tensor_model"].keys()
    ]
    da_loss_components = xr.DataArray(
        loss_components,
        dims=["replication", "epoch", "loss_component"],
        coords=dict(
            replication=tf.range(n_rep),
            epoch=tf.range(loss_components.shape[1]),
            loss_component=component_names,
        ),
        name="loss_components",
        attrs=dict(
            description="Loss value per component contributing to "
            "the multi-objective loss function"
        ),
    )

    # --- Time ---
    time = tf.stack([history[i]["time"] for i in range(len(history))])
    da_time = xr.DataArray(
        time,
        dims=["replication", "epoch"],
        coords=dict(
            replication=tf.range(n_rep),
            epoch=tf.range(time.shape[1]),
        ),
        name="time",
        attrs=dict(description="Time per epoch", units="ms"),
    )

    if eliobj.trainer["method"] == "parametric_prior":
        # --- Hyperparameters ---
        hp_keys = list(history[0]["hyperparameter"].keys())
        hyperparams = tf.stack(
            [
                [history[i]["hyperparameter"][k][1:] for i in range(len(history))]
                for k in history[0]["hyperparameter"].keys()
            ],
            -1,
        )
        da_hyperparams = xr.DataArray(
            hyperparams,
            dims=["replication", "epoch", "hyperparameter"],
            coords=dict(
                replication=tf.range(n_rep),
                epoch=tf.range(hyperparams.shape[1]),
                hyperparameter=hp_keys,
            ),
            name="hyperparameter",
            attrs=dict(description="Convergence of model hyperparameters"),
        )

        # --- Gradients ---
        gradients = tf.stack(
            [
                [history[i]["hyperparameter_gradient"][k] for i in range(len(history))]
                for k in range(len(history[0]["hyperparameter_gradient"]))
            ],
            1,
        )
        da_gradients = xr.DataArray(
            gradients,
            dims=["replication", "epoch", "hyperparameter"],
            coords=dict(
                replication=tf.range(n_rep),
                epoch=tf.range(gradients.shape[1]),
                hyperparameter=hp_keys,
            ),
            name="gradients",
            attrs=dict(description="Gradients of model hyperparameters"),
        )

    # --- Final Dataset ---
    vars_dict = dict(loss=da_loss, loss_component=da_loss_components, time=da_time)
    if eliobj.trainer["method"] == "parametric_prior":
        vars_dict["hyperparameters"] = da_hyperparams
        vars_dict["gradients"] = da_gradients
    else:
        raise NotImplementedError

    return xr.Dataset(data_vars=vars_dict)


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
    ds_group = xr.Dataset()

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

        da = xr.DataArray(
            data=var,
            dims=dims,
            coords=coords,
            name=k,
            attrs=dict(description=description),
        )

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
    ds_prior = xr.Dataset()
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
            attrs=dict(description=f"Prior samples of model parameter {k}"),
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
    ds_oracle = xr.Dataset()
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


def create_inference_data_obj(eliobj) -> az.InferenceData:
    """Crate inference data object

    Parameters
    ----------
    eliobj :
        eliobj containing training information included in a
        history and result section

    Returns
    -------
    :
        inference data object
    """
    inf_dat = az.InferenceData()

    inf_dat.add_groups(group_dict={"prior": create_prior_ds(eliobj)})
    inf_dat.add_groups(
        group_dict={
            "model": create_result_group(
                eliobj,
                group="model_samples",
                dim_name="model",
                description="simulations from generative model",
            )
        }
    )
    inf_dat.add_groups(
        group_dict={
            "target_quantity": create_result_group(
                eliobj,
                group="target_quantities",
                dim_name="target",
                description="simulated target quantities",
            )
        }
    )
    inf_dat.add_groups(
        group_dict={
            "elicited_summary": create_result_group(
                eliobj,
                group="elicited_statistics",
                dim_name="summary",
                description="simulated elicited summaries",
                base_dims=["replication", "batch"],
            )
        }
    )
    # oracle or expert group
    try:
        eliobj.expert["ground_truth"]
    except KeyError:
        raise NotImplementedError
    else:
        inf_dat.add_groups(group_dict={"oracle": create_oracle_ds(eliobj)})
    inf_dat.add_groups(group_dict={"history_stats": history_to_xarray(eliobj)})

    return inf_dat
