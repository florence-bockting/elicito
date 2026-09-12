"""
Training of the prior with a gradient-based optimizer
"""

import logging
import time
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito._progress import ProgressTable
from elicito.losses import spread_penalty, total_loss
from elicito.methods import get_method
from elicito.simulations import Priors, simulate_and_elicit
from elicito.types import Parameter, Target, Trainer

tfd = tfp.distributions

logger = logging.getLogger(__name__)

MAX_SKIPPED_STEPS = 5
"""Number of non-finite steps in a row after which training stops."""


def _halve_learning_rate(sgd_optimizer: Any) -> None:
    """
    Halve the learning rate of the optimizer

    Used after a non-finite step. A smaller step often keeps the next update
    inside the finite region.

    Parameters
    ----------
    sgd_optimizer
        the optimizer whose learning rate is reduced
    """
    lr = sgd_optimizer.learning_rate
    if hasattr(lr, "assign"):
        lr.assign(lr * 0.5)


def sgd_training(  # noqa: PLR0913, PLR0915
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
    penalties = []
    kappa = trainer.get("kappa", 0.0)
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
    epochs = tf.range(trainer["epochs"])
    bar = ProgressTable(
        "Training",
        total=trainer["epochs"],
        disable=progress == 0,
        loss=float("nan"),
        skipped=0,
    )

    # A single non-finite step must not end the run. The update is skipped and
    # the learning rate is halved, which often lets the run recover. Training
    # stops only after MAX_SKIPPED_STEPS steps in a row have failed.
    n_skipped = 0
    n_skipped_total = 0

    trainable_vars = method.trainable_variables(prior_model)

    @tf.function(reduce_retracing=True)  # type: ignore [misc]
    def train_step() -> Any:
        with tf.GradientTape() as tape:
            # generate simulations from model
            (train_elicits, prior_sim, model_sim, target_quants) = simulate_and_elicit(
                prior_model=prior_model, model=model, targets=targets, seed=seed
            )
            # compute total loss as weighted sum
            (loss, indiv_losses, loss_components_expert, loss_components_training) = (
                total_loss(
                    elicit_training=train_elicits,
                    elicit_expert=expert_elicited_statistics,
                    targets=targets,
                )
            )
            # keep a non-identified prior away from a point mass
            penalty = spread_penalty(prior_sim)
            loss += kappa * penalty
        # compute gradient of loss wrt trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        step_ok = tf.math.is_finite(tf.squeeze(loss))
        for gradient in gradients:
            if gradient is not None:
                step_ok = tf.logical_and(
                    step_ok, tf.reduce_all(tf.math.is_finite(gradient))
                )
        return (
            loss,
            penalty,
            indiv_losses,
            gradients,
            step_ok,
            train_elicits,
            prior_sim,
            model_sim,
            target_quants,
            loss_components_expert,
            loss_components_training,
        )

    @tf.function(reduce_retracing=True)  # type: ignore [misc]
    def apply_update(gradients: Any) -> None:
        sgd_optimizer.apply_gradients(zip(gradients, trainable_vars))

    for epoch in epochs:
        # runtime of one epoch
        epoch_time_start = time.time()

        (
            loss,
            penalty,
            indiv_losses,
            gradients,
            step_ok,
            train_elicits,
            prior_sim,
            model_sim,
            target_quants,
            loss_components_expert,
            loss_components_training,
        ) = train_step()

        if bool(step_ok):
            n_skipped = 0
            # update trainable_variables using gradient info with adam
            # optimizer
            apply_update(gradients)
        else:
            n_skipped += 1
            n_skipped_total += 1
            _halve_learning_rate(sgd_optimizer)

        # time end of epoch
        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_time_start

        gradients_ep.append(gradients)
        method.record_epoch(res_dict, prior_sim, trainable_vars, parameters)

        # savings per epoch (independent from chosen method)
        time_per_epoch.append(epoch_time)
        total_losses.append(tf.squeeze(loss))
        penalties.append(tf.squeeze(penalty))
        component_losses.append(indiv_losses)

        loss_value = float(tf.squeeze(loss))
        bar.update(loss=loss_value, skipped=n_skipped_total)

        # the run cannot recover; stop after the epoch has been recorded
        if n_skipped >= MAX_SKIPPED_STEPS:
            msg = (
                f"Loss is not finite for {MAX_SKIPPED_STEPS} steps in a row."
                " Training stops."
            )
            # a warm-up run inside the initialization sets progress=0. It must
            # not print for each candidate.
            if progress == 1:
                print(msg)
            else:
                logger.info(msg)
            break

    bar.close()

    if n_skipped_total > 0:
        logger.info(
            f"{n_skipped_total} of {len(total_losses)} steps were skipped"
            " because the loss or a gradient was not finite. The learning rate"
            " was halved for each of them."
        )

    res_ep = {
        "loss": total_losses,
        "penalty": penalties,
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
