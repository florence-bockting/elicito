"""
Strategy objects for the prior-learning methods
"""

from typing import Any, Protocol

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

import elicito as el
from elicito import networks
from elicito.types import (
    ExpertDict,
    Initializer,
    NFDict,
    Parameter,
    PriorMethods,
    Target,
    Trainer,
)


def _constraints(parameters: list[Parameter]) -> dict[str, Any]:
    """Map each hyperparameter name to its constraint function."""
    constraints: dict[str, Any] = {}
    for param in parameters:
        hyperparams = param["hyperparams"]
        if hyperparams is None:
            continue
        for hyp in hyperparams:
            constraints[hyperparams[hyp]["name"]] = hyperparams[hyp]["constraint"]
    return constraints


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

    def new_history(
        self, prior_model: Any, parameters: list[Parameter]
    ) -> dict[str, Any]:
        """Create the per-epoch record, seeded with the initial values."""
        ...

    def record_epoch(
        self,
        history: dict[str, Any],
        prior_sim: Any,
        trainable_vars: Any,
        parameters: list[Parameter],
    ) -> None:
        """Append this epoch's values to the record."""
        ...

    def finalize(
        self,
        res_ep: dict[str, Any],
        output_res: dict[str, Any],
        gradients_ep: Any,
        trainable_vars: Any,
    ) -> None:
        """Add the method-specific entries to the results."""
        ...

    def check(
        self,
        parameters: list[Parameter],
        network: NFDict | None,
        initializer: Initializer | None,
    ) -> None:
        """Raise if the sections are not valid for this method."""
        ...

    def initialize(  # noqa: PLR0913
        self,
        expert_elicited_statistics: dict[str, tf.Tensor],
        initializer: Initializer | None,
        parameters: list[Parameter],
        trainer: Trainer,
        model: dict[str, Any],
        targets: list[Target],
        network: NFDict | None,
        expert: ExpertDict,
        seed: int,
        progress: int,
    ) -> tuple[Any, Any, Any, Any]:
        """Build the prior model used to start the training."""
        ...

    def init_matrix_slice(
        self,
        initializer: Initializer,
        parameters: list[Parameter],
        trainer: Trainer,
    ) -> Any:
        """Return the initial hyperparameter slice for a dry run."""
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

    def sample(  # noqa: D102
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

    def new_history(
        self, prior_model: Any, parameters: list[Parameter]
    ) -> dict[str, Any]:
        """Create the per-epoch record, seeded with the initial values."""
        constraints = _constraints(parameters)
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
        if initializer["method"] is None and hyperparams is not None:
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

    def initialize(  # noqa: PLR0913
        self,
        expert_elicited_statistics: dict[str, tf.Tensor],
        initializer: Initializer | None,
        parameters: list[Parameter],
        trainer: Trainer,
        model: dict[str, Any],
        targets: list[Target],
        network: NFDict | None,
        expert: ExpertDict,
        seed: int,
        progress: int,
    ) -> tuple[Any, Any, Any, Any]:
        """Build the prior model used to start the training."""
        if initializer is None:
            # check() rejects this earlier; the guard narrows the type
            msg = "If method is 'parametric_prior', 'initializer' can't be None."
            raise ValueError(msg)

        if initializer["hyperparams"] is not None:
            # prepare generative model
            init_prior_model = el.simulations.Priors(
                ground_truth=False,
                init_matrix_slice=initializer["hyperparams"],
                trainer=trainer,
                parameters=parameters,
                network=None,
                expert=expert,
                seed=seed,
            )
            return init_prior_model, None, None, None

        loss_list, init_prior, init_matrix = el.initialization.init_runs(
            expert_elicited_statistics=expert_elicited_statistics,
            initializer=initializer,
            parameters=parameters,
            trainer=trainer,
            model=model,
            targets=targets,
            network=None,
            expert=expert,
            seed=seed,
            progress=progress,
        )

        # extract pre-specified quantile loss out of all runs
        # get corresponding set of initial values
        loss_quantile = initializer["loss_quantile"]

        boolean_mask = tf.math.equal(
            loss_list, tfp.stats.percentile(loss_list, loss_quantile)
        )
        idx = tf.where(tf.squeeze(boolean_mask, 1))

        init_prior_model = init_prior[int(tf.squeeze(idx))]
        return init_prior_model, loss_list, init_prior, init_matrix

    def init_matrix_slice(  # noqa: D102
        self,
        initializer: Initializer,
        parameters: list[Parameter],
        trainer: Trainer,
    ) -> Any:
        if initializer["distribution"] is None:
            return initializer["hyperparams"]

        init_matrix = el.initialization.uniform_samples(
            seed=trainer["seed"],
            hyppar=initializer["distribution"]["hyper"],  # type: ignore [arg-type]
            n_samples=initializer["iterations"],  # type: ignore [arg-type]
            method=initializer["method"],  # type: ignore [arg-type]
            mean=initializer["distribution"]["mean"],
            radius=initializer["distribution"]["radius"],
            parameters=parameters,
        )
        return {f"{key}": init_matrix[key][0] for key in init_matrix}


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
            # the base distribution never changes during training;
            # keep it on the prior object instead of rebuilding it per epoch
            init_prior.base_distribution = base_dist
            # sample from base distribution
            u = base_dist.sample((128, 200))
            init_prior(u, None)
        return init_prior

    def sample(  # noqa: D102
        self,
        initialized_priors: Any,
        parameters: list[Parameter],
        network: NFDict | None,
        B: int,
        num_samples: int,
    ) -> Any:
        # reuse the base distribution built in `build`
        u = initialized_priors.base_distribution.sample((B, num_samples))
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

    def initialize(  # noqa: PLR0913
        self,
        expert_elicited_statistics: dict[str, tf.Tensor],
        initializer: Initializer | None,
        parameters: list[Parameter],
        trainer: Trainer,
        model: dict[str, Any],
        targets: list[Target],
        network: NFDict | None,
        expert: ExpertDict,
        seed: int,
        progress: int,
    ) -> tuple[Any, Any, Any, Any]:
        """Build the prior model used to start the training."""
        # prepare generative model
        init_prior_model = el.simulations.Priors(
            ground_truth=False,
            init_matrix_slice=None,
            trainer=trainer,
            parameters=parameters,
            network=network,
            expert=expert,
            seed=seed,
        )
        # loss_list, init_prior and init_matrix stay empty for this method
        return init_prior_model, None, None, None

    def init_matrix_slice(  # noqa: D102
        self,
        initializer: Initializer,
        parameters: list[Parameter],
        trainer: Trainer,
    ) -> Any:
        return None


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
