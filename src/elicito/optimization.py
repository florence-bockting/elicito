"""
Defines the optimization algorithm
"""

import time
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore
from tqdm import tqdm

from elicito.losses import total_loss
from elicito.simulations import Priors
from elicito.types import Parameter, Target, Trainer
from elicito.utils import one_forward_simulation

tfd = tfp.distributions


def sgd_training(  # noqa: PLR0912, PLR0913, PLR0915
    expert_elicited_statistics: dict[str, tf.Tensor],
    prior_model_init: Priors,
    trainer: Trainer,
    optimizer: dict[str, Any],
    model: dict[str, Any],
    targets: list[Target],
    parameters: list[Parameter],
    seed: int,
    progress: int,
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """
    Run the optimization algorithms for E epochs.

    Parameters
    ----------
    expert_elicited_statistics
        expert data or simulated data representing a prespecified ground truth.

    prior_model_init
        Initialisation and sampling from prior distributions.

    trainer
        Settings for optimization phase

    optimizer
        Settings for SGD-optimizer

    model
        Generative model

    targets
        List of target quantities

    parameters
        List of model parameters

    seed
        Internally used seed for reproducible results

    progress
        Whether progress of training is printed

    Returns
    -------
    res_ep :
        results saved for each epoch (history)

    output_res :
        results saved for the last epoch (results)

    Raises
    ------
    ValueError
        Training has been stopped because loss value is NAN.

    """
    # set seed
    tf.random.set_seed(seed)

    # prepare generative model
    prior_model = prior_model_init
    total_losses = []
    component_losses = []
    gradients_ep = []
    time_per_epoch = []

    if trainer["method"] == "parametric_prior":
        # save initialized trainable variables of epoch=0 (before first update)
        init_vars_values = [
            prior_model.trainable_variables[i].numpy().copy()
            for i in range(len(prior_model.trainable_variables))
        ]
        init_vars_names = [
            prior_model.trainable_variables[i].name[:-2].split(".")[1]
            for i in range(len(prior_model.trainable_variables))
        ]

    # initialize the adam optimizer
    optimizer_copy = optimizer.copy()
    init_sgd_optimizer = optimizer["optimizer"]
    optimizer_copy.pop("optimizer")
    sgd_optimizer = init_sgd_optimizer(**optimizer_copy)

    # start training loop
    if progress == 0:
        epochs = tf.range(trainer["epochs"])
    else:
        print("Training")
        epochs = tqdm(tf.range(trainer["epochs"]))

    for epoch in epochs:
        # runtime of one epoch
        epoch_time_start = time.time()

        with tf.GradientTape() as tape:
            # generate simulations from model
            (train_elicits, prior_sim, model_sim, target_quants) = (
                one_forward_simulation(
                    prior_model=prior_model, model=model, targets=targets, seed=seed
                )
            )
            # compute total loss as weighted sum
            (loss, indiv_losses, loss_components_expert, loss_components_training) = (
                total_loss(
                    elicit_training=train_elicits,
                    elicit_expert=expert_elicited_statistics,
                    targets=targets,
                )
            )

            # compute gradient of loss wrt trainable_variables
            gradients = tape.gradient(loss, prior_model.trainable_variables)

            # update trainable_variables using gradient info with adam
            # optimizer
            sgd_optimizer.apply_gradients(
                zip(gradients, prior_model.trainable_variables)
            )

        # time end of epoch
        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_time_start

        # break for loop if loss is NAN and inform about cause
        if tf.math.is_nan(loss):
            print("Loss is NAN and therefore training stops.")
            break

        # Saving of results
        if trainer["method"] == "parametric_prior":
            # create a list with constraints for re-transforming hyperparameter
            # before saving them
            constraint_dict = dict()

            for i in range(len(parameters)):
                hyp_dict = parameters[i]["hyperparams"]
                for hyp in hyp_dict:  # type: ignore
                    constraint_dict[hyp_dict[hyp]["name"]] = hyp_dict[hyp]["constraint"]  # type: ignore

            # save gradients per epoch
            gradients_ep.append(gradients)

            # save learned hyperparameter values for each prior and epoch
            # extract learned hyperparameter values
            hyperparams = prior_model.trainable_variables
            if epoch == 0:
                # prepare list for saving hyperparameter values
                hyp_list = []
                for i in range(len(hyperparams)):
                    hyp_list.append(hyperparams[i].name[:-2].split(".")[1])
                # create a dict with empty list for each hyperparameter
                res_dict: dict[str, Any] = {f"{k}": [] for k in hyp_list}
                # create final dict with initial train. variables
                for val, name in zip(init_vars_values, init_vars_names):
                    res_dict[name].append(float(constraint_dict[name](val)))  # type: ignore
            # save names and values of hyperparameters
            vars_values = [
                hyperparams[i].numpy().copy() for i in range(len(hyperparams))
            ]
            vars_names = [
                hyperparams[i].name[:-2].split(".")[1] for i in range(len(hyperparams))
            ]
            # create a final dict of hyperparameter values
            for val, name in zip(vars_values, vars_names):
                res_dict[name].append(float(constraint_dict[name](val)))  # type: ignore

        if trainer["method"] == "deep_prior":
            # save mean and std for each sampled marginal prior for each epoch

            if epoch == 0:
                res_dict = {"means": [], "stds": []}

            means = tf.reduce_mean(prior_sim, (0, 1))
            sds = tf.reduce_mean(tf.math.reduce_std(prior_sim, 1), 0)

            for val, name in zip([means, sds], ["means", "stds"]):  # type: ignore
                res_dict[name].append(val)

        # savings per epoch (independent from chosen method)
        time_per_epoch.append(epoch_time)
        total_losses.append(loss)
        component_losses.append(indiv_losses)

    res_ep = {
        "loss": total_losses,
        "loss_component": component_losses,
        "time": time_per_epoch,
        "hyperparameter": res_dict,
    }

    output_res = {
        "target_quantities": target_quants,
        "elicited_statistics": train_elicits,
        "prior_samples": prior_sim,
        "model_samples": model_sim,
        "loss_tensor_expert": loss_components_expert,
        "loss_tensor_model": loss_components_training,
    }

    if trainer["method"] == "parametric_prior":
        res_ep["hyperparameter_gradient"] = gradients_ep

    return res_ep, output_res
