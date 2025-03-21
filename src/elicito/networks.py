"""
setup network block in Elicit object
"""

from typing import Any, Callable

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito.types import NFDict

tfd = tfp.distributions


def NF(
    inference_network: Callable[[Any], Any],
    network_specs: dict[str, Any],
    base_distribution: Callable[[Any], Any],
) -> NFDict:
    """
    Specify normalizing flow used from BayesFlow library

    Parameters
    ----------
    inference_network
        type of inference network as specified by bayesflow.inference_networks.

    network_specs
        specification of normalizing flow architecture. Arguments are inherited
        from chosen bayesflow.inference_networks.

    base_distribution
        Base distribution from which should be sampled during learning.
        Normally the base distribution is a multivariate normal.

    Returns
    -------
    nf_dict :
        dictionary specifying the normalizing flow settings.

    """
    nf_dict: NFDict = dict(
        inference_network=inference_network,
        network_specs=network_specs,
        base_distribution=base_distribution,
    )

    return nf_dict


class BaseNormal:
    """
    standard normal base distribution for normalizing flow
    """

    def __call__(self, num_params: int) -> Any:
        """
        Multivariate standard normal distribution

        distribution has as many dimensions as parameters in the generative model.

        Parameters
        ----------
        num_params
            number of model parameters.

        Returns
        -------
        :
            tfp.distributions object.

        """
        base_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(num_params), scale_diag=tf.ones(num_params)
        )
        return base_dist


# initialized instance of the BaseNormal class
base_normal = BaseNormal()
