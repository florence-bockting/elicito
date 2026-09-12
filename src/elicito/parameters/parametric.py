"""
Independent parametric priors
"""

from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito.parameters._base import _constraints, seed_pair
from elicito.types import (
    Initializer,
    NFDict,
    Parameter,
    PriorMethods,
)


class ParametricPrior:
    """Independent parametric priors."""

    name = PriorMethods.parametric_prior.value

    def __init__(self) -> None:
        # cache filled by `new_history`; the map is constant during training
        self._constraints: dict[str, Any] | None = None

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

    def sample(  # noqa: D102, PLR0913
        self,
        initialized_priors: Any,
        parameters: list[Parameter],
        network: NFDict | None,
        B: int,
        num_samples: int,
        seed: int,
    ) -> Any:
        # One stateless seed per parameter. A stateless seed repeats the draws
        # without `tf.random.set_seed`, which a compiled forward pass cannot
        # afford: it clears the kernel caches of the whole graph. One seed for
        # every parameter would correlate the draws, so the seed is split.
        seeds = tfp.random.split_seed(seed_pair(seed), n=len(parameters), salt="priors")
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
            priors.append(
                prior_family(**init_dict).sample((B, num_samples), seed=seeds[i])
            )
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

    def new_history(
        self, prior_model: Any, parameters: list[Parameter]
    ) -> dict[str, Any]:
        """Create the per-epoch record, seeded with the initial values."""
        constraints = _constraints(parameters)
        self._constraints = constraints
        history: dict[str, Any] = {}
        for var in prior_model.trainable_variables:
            name = var.name[:-2].split(".")[1]
            history.setdefault(name, []).append(
                float(constraints[name](var.numpy().copy()))
            )
        return history

    def record_epoch(
        self,
        history: dict[str, Any],
        prior_sim: Any,
        trainable_vars: Any,
        parameters: list[Parameter],
    ) -> None:
        """Create the per-epoch record, seeded with the initial values."""
        constraints = self._constraints
        if constraints is None:
            constraints = _constraints(parameters)
        for var in trainable_vars:
            name = var.name[:-2].split(".")[1]
            history[name].append(float(constraints[name](var.numpy().copy())))

    def finalize(
        self,
        res_ep: dict[str, Any],
        output_res: dict[str, Any],
        gradients_ep: Any,
        trainable_vars: Any,
    ) -> None:
        """Add the method-specific entries to the results."""
        res_ep["hyperparameter_gradient"] = gradients_ep
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
        if initializer is None:
            msg = (
                "If method is 'parametric_prior', "
                " the section 'initializer' can't be None."
            )
            raise ValueError(msg)

        if network is not None:
            msg = (
                "If method is 'parametric prior' "
                "the 'network' is not used and should be set to None."
            )
            raise ValueError(msg)

        # check that hyperparameter names are not redundant
        hyp_names = []
        hyp_shared = []
        for i in range(len(parameters)):
            if parameters[i]["hyperparams"] is None:
                msg = (
                    "When using method='parametric_prior', the argument "
                    "'hyperparams' of el.parameter "
                    "cannot be None."
                )
                raise ValueError(msg)

            hyp_names.append(
                [
                    parameters[i]["hyperparams"][key]["name"]
                    for key in parameters[i]["hyperparams"].keys()
                ]
            )
            hyp_shared.append(
                [
                    parameters[i]["hyperparams"][key]["shared"]
                    for key in parameters[i]["hyperparams"].keys()
                ]
            )
        # flatten nested list
        hyp_names_flat = sum(hyp_names, [])  # noqa: RUF017
        hyp_shared_flat = sum(hyp_shared, [])  # noqa: RUF017

        hyperparams = initializer["hyperparams"]
        # exact values win over a method string, as resolve_init_method does
        if hyperparams is not None:
            for k in hyperparams:
                if k not in hyp_names_flat:
                    msg = (
                        f"Hyperparameter name '{k}' doesn't "
                        "match any name specified in the parameters "
                        "section. Have you misspelled the name?"
                    )
                    raise ValueError(msg)

        seen = []
        duplicate = []
        share = []
        for n, s in zip(hyp_names_flat, hyp_shared_flat):
            if n not in seen:
                seen.append(n)
            elif s:
                share.append(n)
            else:
                duplicate.append(n)

        if len(duplicate) != 0:
            msg = (
                "The following hyperparameter have the same "
                f"name but are not shared: {duplicate}. \n"
                "Have you forgot to set shared=True?"
            )
            raise ValueError(msg)
