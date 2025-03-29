"""
Simulations from prior and model
"""

from typing import Any, Callable, Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito.types import ExpertDict, NFDict, Parameter, Trainer

tfd = tfp.distributions


# initalize generator model
class Priors(tf.Module):
    """
    Initialize the hyperparameters (i.e., trainable variables)

    Parameters
    ----------
    ground_truth
        True if expert data are simulated from a given ground truth (oracle)

    init_matrix_slice
        Samples drawn from the initialization distribution to initialize
        the hyperparameter of the parametric prior distributions
        Only required for `method = "parametric_prior"` otherwise None.
    trainer
        Specification of training settings

    parameters
        List of model parameters

    network
        Specification of neural network
        Only required for ``deep_prior`` method.
        For ``parametric_prior`` use ``None``.

    expert
        Provide input data from expert or simulate data from oracle with
        either the ``data`` or ``simulator`` method

    seed
        Seed used for learning.
    """

    def __init__(  # noqa: PLR0913
        self,
        ground_truth: bool,
        init_matrix_slice: Optional[dict[str, tf.Tensor]],
        trainer: Trainer,
        parameters: list[Parameter],
        network: Optional[NFDict],
        expert: ExpertDict,
        seed: int,
    ):
        self.ground_truth = ground_truth
        self.init_matrix_slice = init_matrix_slice
        self.trainer = trainer
        self.parameters = parameters
        self.network = network
        self.expert = expert
        # initialize new attribute
        self.init_priors: Union[None, dict[str, tf.Tensor]]
        # set seed
        tf.random.set_seed(seed)
        # initialize hyperparameter for learning (if true hyperparameter
        # are given, no initialization is needed)
        if not self.ground_truth:
            self.init_priors = intialize_priors(
                self.init_matrix_slice,
                self.trainer["method"],
                seed,
                self.parameters,
                self.network,
            )
        else:
            self.init_priors = None

    def __call__(self) -> Any:  # shape=[B,num_samples,num_params]
        """
        Sample from the initialized prior distribution(s).

        Returns
        -------
        prior_samples
            Samples from prior distribution(s).

        """
        prior_samples = sample_from_priors(
            initialized_priors=self.init_priors,
            ground_truth=self.ground_truth,
            num_samples=self.trainer["num_samples"],
            B=self.trainer["B"],
            seed=self.trainer["seed"],
            method=self.trainer["method"],
            parameters=self.parameters,
            network=self.network,
            expert=self.expert,
        )

        return prior_samples


def intialize_priors(  # noqa: PLR0912
    init_matrix_slice: Optional[dict[str, tf.Tensor]],
    method: str,
    seed: int,
    parameters: list[Parameter],
    network: Optional[NFDict],
) -> dict[str, tf.Tensor]:
    """
    Initialize prior distributions.

    Parameters
    ----------
    init_matrix_slice
        Samples drawn from the initialization distribution to initialize
        the hyperparameter of the parametric prior distributions
        Only for method="parametric_prior", otherwise None.

    method
        Parametric_prior or deep_prior method

    seed
        Seed of current workflow run

    parameters
        List of model parameter

    network
        specification of neural network
        Only required for ``deep_prior`` method. For ``parametric_prior``
        use ``None``.

    Returns
    -------
    init_prior :
        returns initialized prior distributions ready for prior sampling.

    """
    # set seed
    tf.random.set_seed(seed)

    if method == "parametric_prior":
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
                hp_dict = parameters[i]["hyperparams"][hp_k]  # type: ignore [index]

            if hp_dict is not None:
                if hp_dict["shared"] and hp_dict["name"] in checked_params:
                    pass
                else:
                    # get initial value
                    if init_matrix_slice is not None:
                        initial_value: Any = float(init_matrix_slice[hp_n])
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

    if method == "deep_prior":
        # for more information see BayesFlow documentation
        # https://bayesflow.org/api/bayesflow.inference_networks.html
        if network is not None:
            INN = network["inference_network"]

            invertible_neural_network = INN(**network["network_specs"])  # type: ignore [call-arg]

            # save initialized priors
            init_prior = invertible_neural_network

    return init_prior


def sample_from_priors(  # noqa: PLR0913
    initialized_priors: Union[None, dict[str, tf.Tensor], Callable[[Any], Any]],
    ground_truth: bool,
    num_samples: int,
    B: int,
    seed: int,
    method: str,
    parameters: list[Parameter],
    network: Optional[NFDict],
    expert: ExpertDict,
) -> Any:  # shape=[B,num_samples,num_params]
    """
    Sample from initialized prior distributions.

    Parameters
    ----------
    initialized_priors
        Initialized prior distributions ready for prior sampling.

    ground_truth
        True if expert data is simulated from ground truth.

    num_samples
        Number of samples from the prior(s).

    B
        Batch size.

    seed
        Seed used for learning.

    method
        Parametric_prior or deep_prior method

    parameters
        List of model parameters

    network
        Specification of neural network
        Only required for ``deep_prior`` method. For ``parametric_prior``
        use ``None``.

    expert
        Provide input data from expert or simulate data from oracle with
        either the ``data`` or ``simulator`` method

    Returns
    -------
    prior_samples :
        Samples from prior distributions.

    """
    # set seed
    tf.random.set_seed(seed)
    if ground_truth:
        # number of samples for ground truth
        rep_true = expert["num_samples"]
        priors = []

        for pr in list(expert["ground_truth"].values()):
            # sample from the prior distribution
            prior_sample = pr.sample((1, rep_true))
            # ensure that all samples have the same shape
            if len(prior_sample.shape) < 3:  # noqa: PLR2004
                prior = tf.expand_dims(prior_sample, -1)
            else:
                prior = prior_sample
            priors.append(prior)
        # concatenate all prior samples into one tensor
        prior_samples = tf.concat(priors, axis=-1)

    if (method == "parametric_prior") and (not ground_truth):
        priors = []

        for i in range(len(parameters)):
            # get the prior distribution family as specified by the user
            prior_family = parameters[i]["family"]

            hp_k = list(parameters[i]["hyperparams"].keys())  # type: ignore [union-attr]
            init_dict = {}
            for k in hp_k:
                hp_n = parameters[i]["hyperparams"][k]["name"]  # type: ignore [index]
                hp_constraint = parameters[i]["hyperparams"][k]["constraint"]  # type: ignore [index]
                init_key = f"{k}_{hp_n}"
                # init_dict[f"{k}"]=initialized_priors[init_key]
                init_dict[f"{k}"] = hp_constraint(initialized_priors[init_key])  # type: ignore
            # sample from the prior distribution
            priors.append(prior_family(**init_dict).sample((B, num_samples)))
        # stack all prior distributions into one tf.Tensor of
        # shape (B, S, num_parameters)
        if len(priors[0].shape) < 3:  # noqa: PLR2004
            prior_samples = tf.stack(priors, axis=-1)
        else:
            prior_samples = tf.concat(priors, axis=-1)

    if (
        (method == "deep_prior")
        and (not ground_truth)
        and initialized_priors is not None
    ):
        # initialize base distribution
        base_dist = network["base_distribution"](num_params=len(parameters))  # type: ignore
        # sample from base distribution
        u = base_dist.sample((B, num_samples))
        # apply transformation function to samples from base distr.
        (unconstr_priors, _) = initialized_priors(u, condition=None, inverse=False)  # type: ignore
        # apply parameter constraints if specified
        constr_priors = []
        for j in range(len(parameters)):
            constr = parameters[j]["constraint"]
            constr_priors.append(constr(unconstr_priors[:, :, j]))
        prior_samples = tf.stack(constr_priors, axis=-1)
    return prior_samples


def simulate_from_generator(
    prior_samples: tf.Tensor,
    seed: int,
    model: dict[str, Any],  # shape=[B,num_samples,num_params]
) -> Any:
    """
    Simulate data from the specified generative model.

    Parameters
    ----------
    prior_samples
        Samples from prior distributions.

    seed
        Seed used for learning. Specification in :func:`elicit.elicit.trainer`.

    model
        Specification of generative model using :func:`elicit.elicit.model`.

    Returns
    -------
    model_simulations :
        simulated data from generative model.

    """
    # set seed
    tf.random.set_seed(seed)
    # get model and initialize generative model
    GenerativeModel = model["obj"]
    generative_model = GenerativeModel()
    # get model specific arguments (that are not prior samples)
    add_model_args = model.copy()
    add_model_args.pop("obj")
    # simulate from generator
    if len(add_model_args) < 1:
        model_simulations = generative_model(prior_samples)
    else:
        model_simulations = generative_model(prior_samples, **add_model_args)

    return model_simulations
