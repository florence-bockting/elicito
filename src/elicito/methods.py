"""
Strategy objects for the prior-learning methods
"""

from typing import Any, Protocol

import tensorflow as tf

from elicito.types import NFDict, Parameter, PriorMethods


class PriorMethod(Protocol):
    """Behaviour that differs between the prior-learning methods."""

    name: str

    def build(
        self,
        parameters: list[Parameter],
        network: NFDict | None,
        init_matrix_slice: dict[str, tf.Tensor] | None,
        seed: int,
    ) -> Any:
        """Create the trainable prior object."""
        ...


class ParametricPrior:
    """Independent parametric priors."""

    name = PriorMethods.parametric_prior.value

    def build(
        self,
        parameters: list[Parameter],
        network: NFDict | None,
        init_matrix_slice: dict[str, tf.Tensor] | None,
        seed: int,
    ) -> Any:
        """Create the trainable prior object."""
        # create dict with all hyperparameters
        hyp_dict = dict()
        hp_keys = list()
        param_names = list()
        hp_names = list()
        initialized_hyperparam: dict[str, Any] = dict()

        for i in range(len(parameters)):
            hyperparameter = parameters[i]["hyperparams"]
            if hyperparameter is not None:
                num_hyperpar = len(hyperparameter)

                hyp_dict[f"param{i}"] = hyperparameter
                param_names += [parameters[i]["name"]] * num_hyperpar
                hp_keys += list(hyperparameter.keys())
                for j in range(num_hyperpar):
                    current_key = list(hyperparameter.keys())[j]
                    hp_names.append(hyperparameter[current_key]["name"])

        checked_params = list()
        for j, (i, hp_n, hp_k) in enumerate(
            zip(tf.unique(param_names).idx, hp_names, hp_keys)
        ):
            if parameters[i]["hyperparams"] is not None:
                hp_dict = parameters[i]["hyperparams"][hp_k]

            if hp_dict is not None:
                if hp_dict["shared"] and hp_dict["name"] in checked_params:
                    pass
                else:
                    # get initial value
                    if init_matrix_slice is not None:
                        initial_value: Any = init_matrix_slice[hp_n]
                    # initialize hyperparameter
                    initialized_hyperparam[f"{hp_k}_{hp_n}"] = tf.Variable(
                        initial_value=initial_value,
                        trainable=True,
                        name=f"{hp_dict['constraint_name']}.{hp_n}",
                    )

                    # save initialized priors
                    init_prior = initialized_hyperparam

                if hp_dict["shared"]:
                    checked_params.append(hp_n)
        return init_prior


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

            invertible_neural_network = INN(**network["network_specs"])  # type: ignore [call-arg]

            # save initialized priors
            init_prior = invertible_neural_network

            # build network
            # initialize base distribution
            base_dist = network["base_distribution"](num_params=len(parameters))  # type: ignore
            # sample from base distribution
            u = base_dist.sample((128, 200))
            init_prior(u, None)
        return init_prior


_METHODS: dict[str, PriorMethod] = {
    ParametricPrior.name: ParametricPrior(),
    DeepPrior.name: DeepPrior(),
}


def get_method(name: str) -> PriorMethod:
    """Return the strategy for a ``trainer["method"]`` string."""
    try:
        return _METHODS[name]
    except KeyError:
        msg = f"Unknown method {name!r}. Valid: {sorted(_METHODS)}."
        raise ValueError(msg) from None
