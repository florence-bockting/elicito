"""
Defines the optimization algorithm
"""

import time
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore
from tqdm import tqdm

from elicito.losses import total_loss
from elicito.methods import get_method
from elicito.simulations import Priors
from elicito.types import Parameter, Target, Trainer
from elicito.utils import one_forward_simulation

tfd = tfp.distributions


def sgd_training(  # noqa: PLR0913
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

    method = get_method(trainer["method"])
    res_dict = method.new_history(prior_model, parameters)

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
            trainable_vars = method.trainable_variables(prior_model)

            # compute gradient of loss wrt trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # update trainable_variables using gradient info with adam
            # optimizer
            sgd_optimizer.apply_gradients(zip(gradients, trainable_vars))

        # time end of epoch
        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_time_start

        # break for loop if loss is NAN and inform about cause
        if tf.math.is_nan(loss):
            print("Loss is NAN and therefore training stops.")
            break

        gradients_ep.append(gradients)
        method.record_epoch(res_dict, prior_sim, trainable_vars, parameters)

        # savings per epoch (independent from chosen method)
        time_per_epoch.append(epoch_time)
        total_losses.append(tf.squeeze(loss))
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

    method.finalize(res_ep, output_res, gradients_ep, trainable_vars)

    return res_ep, output_res
