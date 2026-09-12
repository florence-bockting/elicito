"""
Joint priors learned with a normalizing flow
"""

from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito.parameters import networks
from elicito.parameters._base import numpy_seed, seed_pair
from elicito.types import (
    Initializer,
    NFDict,
    Parameter,
    PriorMethods,
)


class DeepPrior:
    """Joint non-parametric prior via a normalizing flow."""

    name = PriorMethods.deep_prior.value

    def build(
        self,
        parameters: list[Parameter],
        network: NFDict | None,
        init_matrix_slice: dict[str, tf.Tensor] | None,
        seed: int,
    ) -> Any:
        """Create the trainable prior object."""
        # for more information see BayesFlow documentation
        # https://bayesflow.org/api/bayesflow.inference_networks.html
        if network is not None:
            INN = network["inference_network"]

            # The permutation layers draw from the numpy global generator,
            # see `numpy_seed`.
            with numpy_seed(seed):
                invertible_neural_network = INN(**network["network_specs"])  # type: ignore [call-arg]

            # save initialized priors
            init_prior = invertible_neural_network

            # build network
            # initialize base distribution
            base_dist = network["base_distribution"](num_params=len(parameters))  # type: ignore
            # the base distribution never changes during training;
            # keep it on the prior object instead of rebuilding it per epoch
            init_prior.base_distribution = base_dist
            # sample from base distribution
            u = base_dist.sample((128, 200))
            init_prior(u, None)
        return init_prior

    def sample(  # noqa: D102, PLR0913
        self,
        initialized_priors: Any,
        parameters: list[Parameter],
        network: NFDict | None,
        B: int,
        num_samples: int,
        seed: int,
    ) -> Any:
        # reuse the base distribution built in `build`. The seed is stateless,
        # see `ParametricPrior.sample`.
        u = initialized_priors.base_distribution.sample(
            (B, num_samples),
            seed=tfp.random.split_seed(seed_pair(seed), n=1, salt="base")[0],
        )
        # apply transformation function to samples from base distr.
        (unconstr_priors, _) = initialized_priors(u, condition=None, inverse=False)
        # apply parameter constraints if specified
        constr_priors = []
        for j in range(len(parameters)):
            constr = parameters[j]["constraint"]
            constr_priors.append(constr(unconstr_priors[:, :, j]))
        prior_samples = tf.stack(constr_priors, axis=-1)
        return prior_samples

    def trainable_variables(self, prior_model: Any) -> Any:
        """Return the variables the optimizer updates."""
        return prior_model.init_priors.trainable_variables

    def new_history(
        self, prior_model: Any, parameters: list[Parameter]
    ) -> dict[str, Any]:
        """Create the per-epoch record, seeded with the initial values."""
        return {"means": [], "stds": []}

    def record_epoch(
        self,
        history: dict[str, Any],
        prior_sim: Any,
        trainable_vars: Any,
        parameters: list[Parameter],
    ) -> None:
        """Create the per-epoch record, seeded with the initial values."""
        history["means"].append(tf.reduce_mean(prior_sim, (0, 1)))
        history["stds"].append(tf.reduce_mean(tf.math.reduce_std(prior_sim, 1), 0))

    def finalize(
        self,
        res_ep: dict[str, Any],
        output_res: dict[str, Any],
        gradients_ep: Any,
        trainable_vars: Any,
    ) -> None:
        """Add the method-specific entries to the results."""
        output_res["num_NN_weights"] = [v.shape for v in trainable_vars]
        output_res["learned_weights"] = {
            f"weight_{i}": v.numpy().copy() for i, v in enumerate(trainable_vars)
        }

    def check(
        self,
        parameters: list[Parameter],
        network: NFDict | None,
        initializer: Initializer | None,
    ) -> None:
        """Raise if the sections are not valid for this method."""
        if network is None:
            msg = "If method is 'deep prior',  the section 'network' can't be None."
            raise ValueError(msg)

        if initializer is not None:
            msg = (
                "For method 'deep_prior' the "
                "'initializer' is not used and should be set to None."
            )
            raise ValueError(msg)

        if network["network_specs"]["num_params"] != len(parameters):
            msg = (
                "The number of model parameters as "
                "specified in the parameters section, must match the "
                "number of parameters specified in the network."
                f"Expected {len(parameters)} but got "
                f"{network['network_specs']['num_params']}"
            )
            raise ValueError(msg)

        if network["base_distribution"].__class__ != networks.BaseNormal:
            msg = (
                "Currently only the standard normal distribution "
                "is implemented as base distribution. "
                "See GitHub issue #35."
            )
            raise NotImplementedError(msg)
