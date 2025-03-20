"""
Specification of target quantities and elicited statistics
"""

from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito.types import Target

tfd = tfp.distributions


def pearson_correlation(prior_samples: tf.Tensor) -> tf.Tensor:
    """
    Compute pearson correlation coefficient

    pearson correlation coefficient between model parameters

    Parameters
    ----------
    prior_samples
        Samples from prior distributions

    Returns
    -------
    correlation :
        Pearson correlation coefficient
    """
    corM = tfp.stats.correlation(prior_samples, sample_axis=1, event_axis=-1)
    tensor = tf.experimental.numpy.triu(corM, 1)
    tensor_mask = tf.experimental.numpy.triu(corM, 1) != 0.0

    cor = tf.boolean_mask(tensor, tensor_mask, axis=0)
    diag_elements = int((tensor.shape[-1] * (tensor.shape[-1] - 1)) / 2)
    return tf.reshape(cor, (prior_samples.shape[0], diag_elements))


# TODO: Update Custom Target Function
def use_custom_functions(simulations: dict[str, Any], custom_func: Any) -> Any:
    """
    Prepare user-defined functions

    Parameters
    ----------
    simulations
        Simulations from generative model

    custom_func
        User-defined function

    Returns
    -------
    custom_function :
        Custom function with keyword arguments
    """
    vars_from_func = custom_func.__code__.co_varnames
    res = {f"{var}": simulations[var] for var in vars_from_func if var in simulations}
    return custom_func(**res)


def computation_elicited_statistics(
    target_quantities: dict[str, tf.Tensor],  # shape=[B, num_samples, num_obs]
    targets: list[Target],
) -> dict[str, tf.Tensor]:  # shape=[B, num_stats]
    """
    Compute the elicited statistics

    elicited statistics from the target quantities by applying a
    prespecified elicitation technique.

    Parameters
    ----------
    target_quantities
        Simulated target quantities.

    targets
        List of target quantities

    Returns
    -------
    elicits_res :
        simulated elicited statistics.

    """
    # initialize dict for storing results
    elicits_res = dict()
    # loop over elicitation techniques
    for i in range(len(targets)):
        # use custom method if specified otherwise use built-in methods
        if targets[i]["query"]["name"] == "custom":
            elicited_statistic = use_custom_functions(
                target_quantities, targets[i]["query"]["value"]
            )
            elicits_res[f"{targets[i]['query']['func_name']}_{targets[i]['name']}"] = (
                elicited_statistic
            )

        if targets[i]["query"]["name"] == "identity":
            elicits_res[f"identity_{targets[i]['name']}"] = target_quantities[
                targets[i]["name"]
            ]

        if targets[i]["query"]["name"] == "pearson_correlation":
            # compute correlation between model parameters (used for
            # learning correlation structure of joint prior)
            elicited_statistic = pearson_correlation(
                target_quantities[targets[i]["name"]]
            )
            # save correlation in result dictionary
            elicits_res[f"cor_{targets[i]['name']}"] = elicited_statistic

        if targets[i]["query"]["name"] == "quantiles":
            quantiles = targets[i]["query"]["value"]

            # reshape target quantity
            target_tensor = target_quantities[targets[i]["name"]]
            tensor_rank = tf.rank(target_tensor)

            if (
                (tensor_rank == 3)  # noqa: PLR2004
                and (target_tensor.shape[1] is not None)
                and (target_tensor.shape[2] is not None)
            ):
                quan_reshaped = tf.reshape(
                    target_tensor,
                    shape=(
                        target_tensor.shape[0],
                        target_tensor.shape[1] * target_tensor.shape[2],
                    ),
                )
            elif tensor_rank == 2:  # noqa: PLR2004
                quan_reshaped = target_tensor
            else:
                msg = (
                    "rank of tensor of target quantity must be <=3,",
                    f"but got {tensor_rank}.",
                    f"for target quantity {targets[i]['name']}",
                )
                raise ValueError(msg)

            # compute quantiles
            computed_quantiles = tfp.stats.percentile(
                quan_reshaped, q=quantiles, axis=-1
            )
            # bring quantiles to the last dimension
            elicited_statistic = tf.einsum("ij...->ji...", computed_quantiles)
            elicits_res[f"quantiles_{targets[i]['name']}"] = elicited_statistic

    # return results
    return elicits_res


def computation_target_quantities(
    model_simulations: dict[str, tf.Tensor],
    prior_samples: tf.Tensor,  # shape=[B,rep,num_param]
    targets: list[Target],
) -> dict[str, tf.Tensor]:
    """
    Compute target quantities from model simulations.

    Parameters
    ----------
    model_simulations
        Simulations from generative model.

    prior_samples
        Samples from prior distributions of model parameters. Currently only
        needed if correlations between model parameters is used as elicitation
        technique.

    targets
        List of target quantities

    Returns
    -------
    targets_res :
        computed target quantities.
    """
    # initialize dict for storing results
    targets_res = dict()
    # loop over target quantities
    for i in range(len(targets)):
        tar = targets[i]
        # use correlation between model parameters
        if tar["query"]["name"] == "pearson_correlation":
            target_quantity = prior_samples
        # use custom target method
        elif tar["target_method"] is not None:
            target_quantity = use_custom_functions(
                model_simulations, tar["target_method"]
            )
        # target quantity equal to output of GenerativeModel
        else:
            target_quantity = model_simulations[tar["name"]]

        # save target quantities
        targets_res[tar["name"]] = target_quantity

    return targets_res
