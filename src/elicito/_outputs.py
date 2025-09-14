import tensorflow as tf
import xarray as xr


def history_to_xarray(eliobj):
    loss = tf.squeeze(
        tf.stack([eliobj.history[i]["loss"] for i in range(len(eliobj.history))], 0)
    )

    da_loss = xr.DataArray(
        data=loss,
        dims=["replications", "epochs"],
        coords=dict(
            replications=tf.range(0, loss.shape[0]),
            epochs=tf.range(1, loss.shape[1] + 1),
        ),
        name="loss",
        attrs=dict(description="Total loss value"),
    )

    loss_components = tf.stack(
        [eliobj.history[i]["loss_component"] for i in range(len(eliobj.history))], 0
    )

    da_losscomponents = xr.DataArray(
        data=loss_components,
        dims=["replications", "epochs", "loss_components"],
        coords=dict(
            replications=tf.range(0, loss_components.shape[0]),
            epochs=tf.range(1, loss_components.shape[1] + 1),
            loss_components=[
                "_".join(k.split("_")[:-2])
                for k in eliobj.results[0]["loss_tensor_model"].keys()
            ],
        ),
        name="loss_components",
        attrs=dict(
            description="Loss value per loss component which"
            + " enter the multi-objective loss function"
        ),
    )

    time = tf.stack([eliobj.history[i]["time"] for i in range(len(eliobj.history))], 0)

    da_time = xr.DataArray(
        data=time,
        dims=["replications", "epochs"],
        coords=dict(
            replications=tf.range(0, time.shape[0]),
            epochs=tf.range(1, time.shape[1] + 1),
        ),
        name="time",
        attrs=dict(description="Time per epoch", units="ms"),
    )

    hyperparameter = tf.stack(
        [
            [
                eliobj.history[i]["hyperparameter"][k][1:]
                for i in range(len(eliobj.history))
            ]
            for k in eliobj.history[0]["hyperparameter"].keys()
        ],
        -1,
    )

    da_hyperparameter = xr.DataArray(
        data=hyperparameter,
        dims=["replications", "epochs", "hyperparameter"],
        coords=dict(
            replications=tf.range(0, hyperparameter.shape[0]),
            epochs=tf.range(1, hyperparameter.shape[1] + 1),
            hyperparameter=list(eliobj.history[0]["hyperparameter"].keys()),
        ),
        name="hyperparameter",
        attrs=dict(description="Convergence of model hyperparameter"),
    )

    gradients = tf.stack(
        [
            [
                eliobj.history[i]["hyperparameter_gradient"][k]
                for i in range(len(eliobj.history))
            ]
            for k in range(len(eliobj.history[0]["hyperparameter_gradient"]))
        ],
        1,
    )

    da_gradients = xr.DataArray(
        data=gradients,
        dims=["replications", "epochs", "hyperparameter"],
        coords=dict(
            replications=tf.range(0, gradients.shape[0]),
            epochs=tf.range(1, gradients.shape[1] + 1),
            hyperparameter=list(eliobj.history[0]["hyperparameter"].keys()),
        ),
        name="gradients",
        attrs=dict(description="Gradients of model hyperparameter"),
    )

    ds_history = xr.Dataset(
        data_vars=dict(
            loss=da_loss,
            loss_component=da_losscomponents,
            time=da_time,
            hyperparameters=da_hyperparameter,
            gradients=da_gradients,
        )
    )

    return ds_history
