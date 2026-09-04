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

    def sample(
        self,
        initialized_priors: Any,
        parameters: list[Parameter],
        network: NFDict | None,
        B: int,
        num_samples: int,
    ) -> Any:
        """Draw prior samples of shape (B, num_samples, num_params)."""
        ...

    def trainable_variables(self, prior_model: Any) -> Any:
        """Return the variables the optimizer updates."""
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

    def sample(
        self,
        initialized_priors: Any,
        parameters: list[Parameter],
        network: NFDict | None,
        B: int,
        num_samples: int,
    ) -> Any:
        priors = []
        for i in range(len(parameters)):
            # get the prior distribution family as specified by the user
            prior_family = parameters[i]["family"]

            hp_k = list(parameters[i]["hyperparams"].keys())
            init_dict = {}
            for k in hp_k:
                hp_n = parameters[i]["hyperparams"][k]["name"]
                hp_constraint = parameters[i]["hyperparams"][k]["constraint"]
                init_key = f"{k}_{hp_n}"
                # init_dict[f"{k}"]=initialized_priors[init_key]
                init_dict[f"{k}"] = hp_constraint(initialized_priors[init_key])
            # sample from the prior distribution
            priors.append(prior_family(**init_dict).sample((B, num_samples)))
        # stack all prior distributions into one tf.Tensor of
        # shape (B, S, num_parameters)
        if len(priors[0].shape) < 3:  # noqa: PLR2004
            prior_samples = tf.stack(priors, axis=-1)
        else:
            prior_samples = tf.concat(priors, axis=-1)
        return prior_samples

    def trainable_variables(self, prior_model: Any) -> Any:
        """Return the variables the optimizer updates."""
        return prior_model.trainable_variables


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

    def sample(
        self,
        initialized_priors: Any,
        parameters: list[Parameter],
        network: NFDict | None,
        B: int,
        num_samples: int,
    ) -> Any:
        # initialize base distribution
        base_dist = network["base_distribution"](num_params=len(parameters))  # type: ignore
        # sample from base distribution
        u = base_dist.sample((B, num_samples))
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
