"""
Built-in loss functions
"""

from typing import Any, Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito.types import Target

tfd = tfp.distributions


def preprocess(elicited_statistics: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """
    Preprocess elicited statistics

    Preprocess elicited statistics such that they have the required format for
    computing the individual losses between expert- and simulated statistics.

    Parameters
    ----------
    elicited_statistics
        Dictionary including the elicited statistics.

    Returns
    -------
    preprocessed_elicits :
        Dictionary including all preprocessed elicited statistics which will
        enter the loss function to compute the individual loss components.

    Raises
    ------
    AssertionError
        ``elicited_statistics`` can only have 2 dimensions (tensor rank = 2)

    """
    # extract names from elicited statistics
    name_elicits = list(elicited_statistics.keys())

    # prepare dictionary for storing results
    preprocessed_elicits = dict()
    # initialize some helpers for keeping track of target quantity
    target_control: list[str] = []
    i_target = 0
    eval_target = True
    # loop over elicited statistics
    for i, name in enumerate(name_elicits):
        # get name of target quantity
        target = name.split(sep="_")[-1]
        if i != 0:
            # check whether elicited statistic correspond to same target
            # quantity
            eval_target = target_control[-1] == target
        # append current target quantity
        target_control.append(target)
        # if target quantity changes go with index one up
        if not eval_target:
            i_target += 1
        # extract data
        tensor_elicit = elicited_statistics[name]

        if tf.rank(tensor_elicit) > 2:  # noqa: PLR2004
            msg = "elicited statistics can only have 2 dimensions."
            raise AssertionError(msg)

        if tf.rank(tensor_elicit) == 1:
            # add a last axis for loss computation
            prep_elicit = tf.expand_dims(tensor_elicit, axis=-1)
            # store result
            preprocessed_elicits[f"{name}_loss"] = prep_elicit
        else:
            preprocessed_elicits[f"{name}_loss_{i_target}"] = tensor_elicit

    return preprocessed_elicits


def indiv_loss(
    elicit_expert: dict[str, tf.Tensor],  # tensor shape: [1,num_stats]
    elicit_training: dict[str, tf.Tensor],  # tensor shape: [B,num_stats]
    targets: list[Target],
) -> list[float]:
    """
    Compute the individual loss between expert data and model-simulated data.

    Parameters
    ----------
    elicit_expert
        Dictionary including all preprocessed elicited statistics

    elicit_training
        Dictionary including all preprocessed model statistics

    targets
        Target quantities and specification of elicitation technique

    Returns
    -------
    indiv_losses :
        List of individual losses for each loss component

    """
    # create dictionary for storing results
    indiv_losses = []
    # extract expert loss components by name
    name_prep_elicits = list(elicit_expert.keys())
    # compute discrepancy
    for i, name in enumerate(name_prep_elicits):
        # import loss function
        loss_function = targets[i]["loss"]
        # broadcast expert loss to training data-shape
        elicit_expert_brdcst = tf.broadcast_to(
            elicit_expert[name],
            shape=(elicit_training[name].shape[0], elicit_expert[name].shape[1]),
        )
        # compute loss
        indiv_loss = loss_function(elicit_expert_brdcst, elicit_training[name])  # type: ignore [call-arg]
        indiv_losses.append(indiv_loss)

    return indiv_losses


def total_loss(
    elicit_training: dict[str, tf.Tensor],  # tensor shape: [B,num_stats]
    elicit_expert: dict[str, tf.Tensor],  # tensor shape: [1,num_stats]
    targets: list[Target],
) -> tuple[
    tf.Tensor,
    list[float],
    dict[str, tf.Tensor],
    dict[str, tf.Tensor],
]:  # shape: [B,num_stats]
    """
    Compute weighted average

    Compute the weighted average across all individual losses between expert
    data and model simulations.

    Parameters
    ----------
    elicit_training
        Elicited statistics simulated by the model.

    elicit_expert
        Elicited statistics as queried from the expert.

    targets
        Specification of target quantities and elicitation techniques.

    Returns
    -------
    loss :
        Weighted average across individual losses quantifying the discrepancy
        between expert data and model simulations.

    individual_losses :
        List of individual losses for each loss component.

    elicit_expert_prep :
        Dictionary including all preprocessed expert elicited statistics.

    elicit_training_prep :
        Dictionary including all preprocessed model-simulated elicited
        statistics.

    """
    # preprocess expert data and simulated data for usage in loss computation
    elicit_expert_prep = preprocess(elicit_expert)
    elicit_training_prep = preprocess(elicit_training)
    # compute individual losses for each loss component
    individual_losses = indiv_loss(elicit_expert_prep, elicit_training_prep, targets)
    # compute weighted average across individual losses to get the final loss
    # TODO: check whether order of loss_per_component and target quantities
    # is equivalent!
    loss: tf.Tensor = tf.zeros((1,))
    for i in range(len(targets)):
        loss += tf.multiply(individual_losses[i], targets[i]["weight"])

    return (loss, individual_losses, elicit_expert_prep, elicit_training_prep)


def L2(
    loss_component_expert: tf.Tensor,  # shape=[B, num_stats]
    loss_component_training: tf.Tensor,  # shape=[B, num_stats]
    axis: Optional[int] = None,
    ord: Union[str, int] = "euclidean",
) -> tf.Tensor:  # shape=[]
    """
    Compute norm of a difference

    compute the norm of the difference between two tensors along the specified axis.
    Used for the correlation loss when priors are assumed to be independent

    Parameters
    ----------
    loss_component_expert
        Preprocessed expert-elicited data

    loss_component_training
        Preprocessed model-simulated data

    axis
        Axis along which to compute the norm of the difference.

    ord
        Order of the norm. Supports 'euclidean' and other norms
        supported by tf.norm. Default is 'euclidean'.
    """
    difference = tf.subtract(loss_component_expert, loss_component_training)
    norm_values = tf.norm(difference, ord=ord, axis=axis)
    return tf.reduce_mean(norm_values)


class MMD2:
    """
    Maximum mean discrepancy loss
    """

    def __init__(self, kernel: str = "energy", **kwargs: dict[Any, Any]):
        """
        Compute the biased, squared maximum mean discrepancy

        Parameters
        ----------
        kernel
            Kernel type used for computing the MMD.
            When using a gaussian kernel an additional 'sigma' argument has to
            be passed.

        **kwargs
            additional keyword arguments that might be required by the
            different individual kernels

        Raises
        ------
        ValueError
            ``kernel`` must be either 'energy' or 'gaussian' kernel.

            ``sigma`` argument need to be passed if ``kernel = "gaussian"``

        Examples
        --------
        >>> el.losses.MMD2(kernel="energy")  # doctest: +SKIP

        >>> el.losses.MMD2(kernel="gaussian", sigma=1.0)  # doctest: +SKIP

        """
        # ensure that all additionally, required arguments are provided for
        # the respective kernel
        if (kernel == "gaussian") and ("sigma" not in list(kwargs.keys())):
            msg1: tuple[str, str] = (
                "You need to pass a 'sigma' argument when using a gaussian",
                "kernel in the MMD loss",
            )
            raise ValueError(msg1)

        # ensure correct kernel specification
        if kernel not in ["energy", "gaussian"]:
            msg2: tuple[str] = (
                "'kernel' must be either 'energy' or 'gaussian' kernel.",
            )
            raise ValueError(msg2)

        if kernel == "gaussian":
            self.sigma: Any = kwargs["sigma"]
        self.kernel_name = kernel

    def __call__(
        self,
        x: tf.Tensor,  # shape: [B, num_stats]
        y: tf.Tensor,  # shape: [B, num_stats]
    ) -> tf.Tensor:  # shape: []
        """
        Compute the biased, squared maximum mean discrepancy of two samples

        Parameters
        ----------
        x
            Preprocessed expert-elicited statistics.
            Preprocessing refers to broadcasting expert data to same shape as
            model-simulated data.

        y
            Model-simulated statistics corresponding to expert-elicited
            statistics

        Returns
        -------
        MMD2_mean :
            Average biased, squared maximum mean discrepancy between expert-
            elicited and model simulated data.

        """
        # treat samples as column vectors
        x = tf.expand_dims(x, -1)
        y = tf.expand_dims(y, -1)

        # Step 1
        # compute dot product between samples
        xx = tf.matmul(x, x, transpose_b=True)
        xy = tf.matmul(x, y, transpose_b=True)
        yy = tf.matmul(y, y, transpose_b=True)

        # compute squared difference
        u_xx = self.diag(xx)[:, :, None] - 2 * xx + self.diag(xx)[:, None, :]
        u_xy = self.diag(xx)[:, :, None] - 2 * xy + self.diag(yy)[:, None, :]
        u_yy = self.diag(yy)[:, :, None] - 2 * yy + self.diag(yy)[:, None, :]

        # apply kernel function to squared difference
        XX = self.kernel(u_xx, self.kernel_name)
        XY = self.kernel(u_xy, self.kernel_name)
        YY = self.kernel(u_yy, self.kernel_name)

        # Step 2
        # compute biased, squared MMD
        MMD2 = tf.reduce_mean(XX, (1, 2))
        MMD2 -= 2 * tf.reduce_mean(XY, (1, 2))
        MMD2 += tf.reduce_mean(YY, (1, 2))

        MMD2_mean = tf.reduce_mean(MMD2)

        return MMD2_mean

    def clip(
        self,
        u: tf.Tensor,  # shape: [B, num_stats, num_stats]
    ) -> Any:  # shape: [B, num_stats, num_stats]
        """
        Upper and lower clipping of value `u` to improve numerical stability

        Parameters
        ----------
        u
            result of prior computation.

        Returns
        -------
        u_clipped :
            clipped u value with ``min = 1e-8`` and ``max = 1e10``.

        """
        u_clipped = tf.clip_by_value(u, clip_value_min=1e-8, clip_value_max=int(1e10))
        return u_clipped

    def diag(
        self,
        xx: tf.Tensor,  # shape: [B, num_stats, num_stats]
    ) -> Any:  # shape: [B, num_stats]
        """
        Get diagonal elements of a matrix

        Get diagonale elements of a matrix, whereby the first tensor dimension
        are batches and should not be considered to get diagonale elements.

        Parameters
        ----------
        xx
            Similarity matrices with batch dimension in axis=0.

        Returns
        -------
        diag :
            diagonale elements of matrices per batch.

        """
        diag = tf.experimental.numpy.diagonal(xx, axis1=1, axis2=2)
        return diag

    def kernel(
        self,
        u: tf.Tensor,
        kernel: str,  # shape: [B, num_stats, num_stats]
    ) -> Any:  # shape: [B, num_stats, num_stats]
        """
        Kernel used in MMD to compute discrepancy between samples.

        Parameters
        ----------
        u
            squared distance between samples.

        kernel
            name of kernel used for computing discrepancy.

        Returns
        -------
        d :
            discrepancy between samples.

        """
        if kernel == "energy":
            # clipping for numerical stability reasons
            d = -tf.math.sqrt(self.clip(u))
        if kernel == "gaussian":
            d = tf.exp(-0.5 * tf.divide(u, self.sigma))
        return d
