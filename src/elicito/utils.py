"""
helper functions for setting up the Elicit object
"""

import os
import pickle
import warnings
from typing import Any, Optional

import cloudpickle  # type: ignore
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

import elicito as el
from elicito.exceptions import MissingOptionalDependencyError
from elicito.simulations import Priors, simulate_from_generator
from elicito.targets import (
    computation_elicited_statistics,
    computation_target_quantities,
)
from elicito.types import (
    ExpertDict,
    NFDict,
    Parallel,
    Parameter,
    SaveHist,
    SaveResults,
    Target,
    Trainer,
)

tfd = tfp.distributions


def save_as_pkl(obj: Any, save_dir: str) -> None:
    """
    Save file as pickle.

    Parameters
    ----------
    obj
        Variable that needs to be saved.

    save_dir
        Path indicating the file location.

    Examples
    --------
    >>> save_as_pkl(obj, "results/file.pkl")  # doctest: +SKIP

    """
    # if directory does not exist, create it
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    # save obj to location as pickle
    serialized_obj = cloudpickle.dumps(obj)
    with open(save_dir, "wb") as file:
        pickle.dump(serialized_obj, file=file)


def identity(x: float) -> Any:
    """
    Identity function. Returns the input

    Parameters
    ----------
    x
        Input x

    Returns
    -------
    x :
        input x without transformation.

    """
    return x


class DoubleBound:
    """
    constrain double-bounded distributions
    """

    def __init__(self, lower: float, upper: float):
        """
        Constrain double-bounded distribution

        A variable constrained to be in the open interval
        (``lower``, ``upper``) is transformed to an unconstrained variable Y
        via a scaled and translated log-odds transform.

        Basis for the here used constraints, is the
        `constraint transforms implementation in [Stan](https://mc-stan.org/docs/reference-manual/transforms.html).

        Parameters
        ----------
        lower
            Lower bound of variable x.

        upper
            Upper bound of variable x.

        """
        self.lower = lower
        self.upper = upper

    def logit(self, u: tf.Tensor) -> tf.Tensor:
        r"""
        Implement the logit transformation for :math:`u \in (0,1)`:

        .. math::

            logit(u) = \log\left(\frac{u}{1-u}\right)

        Parameters
        ----------
        u
            Variable in open unit interval.

        Returns
        -------
        v
            Log-odds of u.

        """
        # log-odds definition
        v = tf.math.log(u) - tf.math.log(1 - u)
        # cast v into correct dtype
        v = tf.cast(v, dtype=tf.float32)
        return v

    def inv_logit(self, v: float) -> tf.Tensor:
        r"""
        Implement the inverse-logit transformation

        The inverse-logit transformation is the logistic
        sigmoid for :math:`v \in (-\infty,+\infty)`:

        .. math::

            logit^{-1}(v) = \frac{1}{1+\exp(-v)}

        Parameters
        ----------
        v
            Unconstrained variable

        Returns
        -------
        u
            Logistic sigmoid of the unconstrained variable

        """
        # logistic sigmoid transform
        u = tf.divide(1.0, (1.0 + tf.exp(v)))
        # cast v to correct dtype
        u = tf.cast(u, dtype=tf.float32)
        return u

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Scale and translate logit transformed variable

        transform variable x with ``lower`` and ``upper`` bound
        into an unconstrained variable y.

        .. math::

            Y = logit\left(\frac{X - lower}{upper - lower}\right)

        Parameters
        ----------
        x
            Variable with lower and upper bound.

        Returns
        -------
        y
            Unconstrained variable.

        """
        # scaled and translated logit transform
        y = self.logit(tf.divide((x - self.lower), (self.upper - self.lower)))
        # cast y to correct dtype
        y = tf.cast(y, dtype=tf.float32)
        return y

    def inverse(self, y: float) -> tf.Tensor:
        r"""
        Apply inverse of the log-odds transform

        unconstrained variable y is transformed into a constrained variable x
        with ``lower`` and ``upper`` bound.

        .. math::

            X = lower + (upper - lower) \cdot logit^{-1}(Y)

        Parameters
        ----------
        y
            Unconstrained variable

        Returns
        -------
        x :
            Constrained variable with lower and upper bound

        """
        # inverse of log-odds transform
        x = self.lower + (self.upper - self.lower) * self.inv_logit(y)
        # cast x to correct dtype
        x = tf.cast(x, dtype=tf.float32)
        return x


class LowerBound:
    """
    constrain lower-bounded distributions
    """

    def __init__(self, lower: float):
        """
        Transform ``lower`` bound variable to unconstrained variable Y

        use inverse-softplus transform.

        References
        ----------
        - [Stan](https://mc-stan.org/docs/reference-manual/transforms.html)

        Parameters
        ----------
        lower
            Lower bound of variable X.

        """
        self.lower = lower

    def forward(self, x: float) -> Any:
        r"""
        Transform ``lower``-bounded x via inverse-softplus into an unconstrained y.

        .. math::

            Y = softplus^{-1}(X - lower)

        Parameters
        ----------
        x
            Variable with a lower bound.

        Returns
        -------
        y :
            Unconstrained variable.

        """
        # inverse softplus transform
        y = tfp.math.softplus_inverse(x - self.lower)
        # cast y into correct type
        y = tf.cast(y, dtype=tf.float32)
        return y

    def inverse(self, y: float) -> tf.Tensor:
        r"""
        Apply softplus to unconstrained y to get ``lower``-bounded x

        .. math::

            X = softplus(Y) + lower

        Parameters
        ----------
        y
            Unconstrained variable.

        Returns
        -------
        x :
            Variable with a lower bound.

        """
        # softplus transform
        x = tf.math.softplus(y) + self.lower
        # cast x into correct dtype
        x = tf.cast(x, dtype=tf.float32)
        return x


class UpperBound:
    """
    transform ``upper`` bounded distribution
    """

    def __init__(self, upper: float):
        """
        Transform ``upper`` bounded x into unconstrained y

        use inverse-softplus transform.

        Parameters
        ----------
        upper
            Upper bound of variable X.

        References
        ----------
        + [Stan](https://mc-stan.org/docs/reference-manual/transforms.html)

        """
        self.upper = upper

    def forward(self, x: float) -> Any:
        r"""
        Transform upper-bouned into unconstarined variable

        use inverse-softplus transform

        .. math::

            Y = softplus^{-1}(upper - X)

        Parameters
        ----------
        x
            Variable with an upper bound.

        Returns
        -------
        y :
            Unconstrained variable.

        """
        # logarithmic transform
        y = tfp.math.softplus_inverse(self.upper - x)
        # cast y into correct dtype
        y = tf.cast(y, dtype=tf.float32)
        return y

    def inverse(self, y: float) -> tf.Tensor:
        r"""
        Transform uncstrained into lower-bounded variable

        use softplus transform

        .. math::

            X = upper - softplus(Y)

        Parameters
        ----------
        y
            Unconstrained variable.

        Returns
        -------
        x :
            Variable with an upper bound.

        """
        # exponential transform
        x = self.upper - tf.math.softplus(y)
        # cast x into correct dtype
        x = tf.cast(x, dtype=tf.float32)
        return x


def one_forward_simulation(
    prior_model: Priors, model: dict[str, Any], targets: list[Target], seed: int
) -> tuple[dict[Any, Any], tf.Tensor, dict[Any, Any], dict[Any, Any]]:
    """
    Run one forward simulation from prior samples to elicited statistics.

    Parameters
    ----------
    prior_model
        Initialized prior distributions which can be used for sampling.

    model
        Specification of generative model

    targets
        List of target quantities

    seed
        Random seed.

    Returns
    -------
    elicited_statistics :
        Dictionary containing the elicited statistics that can be used to
        compute the loss components

    prior_samples :
        Samples from prior distributions

    model_simulations :
        Samples from the generative model (likelihood) given the prior samples
        for the model parameters

    target_quantities :
        Target quantities as a function of the model simulations.

    """
    # set seed
    tf.random.set_seed(seed)
    # generate samples from initialized prior
    prior_samples = prior_model()
    # simulate prior predictive distribution based on prior samples
    # and generative model
    model_simulations = simulate_from_generator(prior_samples, seed, model)
    # compute the target quantities
    target_quantities = computation_target_quantities(
        model_simulations, prior_samples, targets
    )
    # compute the elicited statistics by applying a specific elicitation
    # method on the target quantities
    elicited_statistics = computation_elicited_statistics(target_quantities, targets)
    return (elicited_statistics, prior_samples, model_simulations, target_quantities)


def get_expert_data(  # noqa: PLR0913
    trainer: Trainer,
    model: dict[str, Any],
    targets: list[Target],
    expert: ExpertDict,
    parameters: list[Parameter],
    network: Optional[NFDict],
    seed: int,
) -> tuple[Any, ...]:
    """
    Load the training data

    data can be expert data or data simulations using a pre-defined ground truth.

    Parameters
    ----------
    trainer
        Specification of training settings and meta-information for
        workflow

    model
        Specification of generative model

    targets
        List of target quantities

    expert
        Provide input data from expert or simulate data from oracle with
        either the ``data`` or ``simulator`` method

    parameters
        List of model parameters specified with :func:`elicit.elicit.parameter`.

    network
        Specification of neural network
        Only required for ``deep_prior`` method. For ``parametric_prior``
        use ``None``.

    seed
        Internal seed for reproducible results

    Returns
    -------
    expert_data :
        dictionary containing the training data. Must have same form as the
        model-simulated elicited statistics. Correct specification of
        keys can be checked using :func:`elicit.utils.get_expert_datformat`

    expert_prior :
        samples from ground truth. Exists only if expert data are simulated
        from an oracle. Otherwise this output is ``None``

    """
    try:
        expert["data"]
    except KeyError:
        oracle = True
    else:
        oracle = False

    if oracle:
        # set seed
        tf.random.set_seed(seed)
        # sample from true priors
        prior_model = Priors(
            ground_truth=True,
            init_matrix_slice=None,
            trainer=trainer,
            parameters=parameters,
            network=network,
            expert=expert,
            seed=seed,
        )
        # compute elicited statistics and target quantities
        expert_data, expert_prior, *_ = one_forward_simulation(
            prior_model=prior_model, model=model, targets=targets, seed=seed
        )
        return tuple((expert_data, expert_prior))
    else:
        # load expert data from file
        expert_data = expert["data"]
        return tuple((expert_data, None))


def save(
    eliobj: Any,
    name: Optional[str] = None,
    file: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Save the eliobj as pickle.

    Parameters
    ----------
    eliobj
        Instance of the :func:`elicit.elicit.Elicit` class.

    name
        Name of the saved .pkl file.
        File is saved as .results/{method}/{name}_{seed}.pkl

    file
        Path to file, including file name,
        e.g. file="res" (saved as res.pkl) or
        file="method1/res" (saved as method1/res.pkl)

    overwrite
        Whether to overwrite existing file.

    """
    # either name or file must be specified
    if not ((name is None) ^ (file is None)):
        msg = (
            "Name and file cannot be both None or specified.",
            "Either one has to be None.",
        )
        raise AssertionError(msg)

    if (name is not None) and (file is None):
        if name.endswith(".pkl"):
            name = name.removesuffix(".pkl")
        # create saving path
        path = f"./results/{eliobj.trainer['method']}/{name}_{eliobj.trainer['seed']}"

    if (name is None) and (file is not None):
        # postprocess file (or name) to avoid file.pkl.pkl
        if file.endswith(".pkl"):
            file = file.removesuffix(".pkl")
        path = "./" + file

    # check whether saving path is already used
    if os.path.isfile(path + ".pkl") and not overwrite:
        user_ans = input(
            "In provided directory exists already a file with"
            + " identical name. Do you want to overwrite it?"
            + " Press 'y' for overwriting and 'n' for abording."
        )
        while user_ans not in ["n", "y"]:
            user_ans = input(
                "Please press either 'y' for overwriting or 'n'"
                + "for abording the process."
            )

        if user_ans == "n":
            overwrite = False
            print("Process aborded. File is not overwritten.")

    if not os.path.isfile(path + ".pkl") or overwrite:
        storage = dict()
        # user inputs
        storage["model"] = eliobj.model
        storage["parameters"] = eliobj.parameters
        storage["targets"] = eliobj.targets
        storage["expert"] = eliobj.expert
        storage["optimizer"] = eliobj.optimizer
        storage["trainer"] = eliobj.trainer
        storage["initializer"] = eliobj.initializer
        storage["network"] = eliobj.network
        # results
        storage["results"] = eliobj.results
        storage["history"] = eliobj.history

        save_as_pkl(storage, path + ".pkl")
        print(f"saved in: {path}.pkl")


def load(file: str) -> Any:
    """
    Load a saved ``eliobj`` from specified path.

    Parameters
    ----------
    file
        path where ``eliobj`` object is saved.

    Returns
    -------
    eliobj :
        loaded ``eliobj`` object.

    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "data_wrangling", requirement="pandas"
        ) from exc

    obj_pickled = pd.read_pickle(file)  # noqa: S301
    obj = pickle.loads(obj_pickled)  # noqa: S301

    eliobj = el.Elicit(
        model=obj["model"],
        parameters=obj["parameters"],
        targets=obj["targets"],
        expert=obj["expert"],
        optimizer=obj["optimizer"],
        trainer=obj["trainer"],
        initializer=obj["initializer"],
        network=obj["network"],
    )

    # add results if already fitted
    eliobj.history = obj["history"]
    eliobj.results = obj["results"]

    return eliobj


def parallel(
    runs: int = 4, cores: Optional[int] = None, seeds: Optional[list[int]] = None
) -> Parallel:
    """
    Specify parallelization

    Specification for parallelizing training by running multiple training
    instances with different seeds simultaneously.

    Parameters
    ----------
    runs
        Number of replication.

    cores
        Number of cores that should be used.

    seeds
        A list of seeds. If ``None`` seeds are drawn from a Uniform(0,999999)
        distribution. The seed information corresponding to each chain is
        stored in ``eliobj.results``.

    Returns
    -------
    parallel_dict :
        dictionary containing the parallelization settings.

    """
    parallel_dict: Parallel = dict(runs=runs, cores=cores, seeds=seeds)  # type: ignore

    if cores is None:
        parallel_dict["cores"] = runs

    return parallel_dict


def save_history(
    loss: bool = True,
    loss_component: bool = True,
    time: bool = True,
    hyperparameter: bool = True,
    hyperparameter_gradient: bool = True,
) -> SaveHist:
    """
    Control sub-results of the history object

    define which results should be included or excluded.
    Results are saved across epochs.
    By default all sub-results are included.

    Parameters
    ----------
    loss
        Total loss per epoch.

    loss_component
        Loss per loss-component per epoch.

    time
        Time in sec per epoch.

    hyperparameter
        'parametric_prior' method: Trainable hyperparameters of parametric
        prior distributions.
        'deep_prior' method: Mean and standard deviation of each marginal
        from the joint prior.

    hyperparameter_gradient
        Gradients of the hyperparameter. Only for 'parametric_prior' method.

    Returns
    -------
    save_hist_dict :
        dictionary with inclusion/exclusion settings for each sub-result in
        history object.

    Warnings
    --------
    if ``loss_component`` or ``loss`` are excluded, :func:`elicit.plots.loss`
    can't be used as it requires this information.

    if ``hyperparameter`` is excluded, :func:`elicit.plots.hyperparameter`
    can't be used as it requires this information.
    Only parametric_prior method.

    if ``hyperparameter`` is excluded, :func:`elicit.plots.marginals`
    can't be used as it requires this information.
    Only deep_prior method.

    """
    if not loss or not loss_component:
        warnings.warn(
            "el.plots.loss() requires information about "
            + "'loss' and 'loss_component'. If you don't save this information "
            + "el.plot.loss() can't be used."
        )
    if not hyperparameter:
        warnings.warn(
            "el.plots.hyperparameter() and el.plots.marginals() require"
            + " information about 'hyperparameter'. If you don't save this"
            + " information these plots can't be used."
        )

    save_hist_dict: SaveHist = dict(
        loss=loss,
        time=time,
        loss_component=loss_component,
        hyperparameter=hyperparameter,
        hyperparameter_gradient=hyperparameter_gradient,
    )
    return save_hist_dict


def save_results(  # noqa: PLR0913
    target_quantities: bool = True,
    elicited_statistics: bool = True,
    prior_samples: bool = True,
    model_samples: bool = True,
    expert_elicited_statistics: bool = True,
    expert_prior_samples: bool = True,
    init_loss_list: bool = True,
    init_prior: bool = True,
    init_matrix: bool = True,
    loss_tensor_expert: bool = True,
    loss_tensor_model: bool = True,
) -> SaveResults:
    """
    Control sub-results of the result object

    Specify whether results should be included or excluded in the final result file.
    Results are based on the computation of the last epoch.
    By default all sub-results are included.

    Parameters
    ----------
    target_quantities
        simulation-based target quantities.

    elicited_statistics
        simulation-based elicited statistics.

    prior_samples
        samples from simulation-based prior distributions.

    model_samples
        output variables from the simulation-based generative model.

    expert_elicited_statistics
        expert-elicited statistics.

    expert_prior_samples
        if oracle is used: samples from the true prior distribution,
        otherwise it is None.

    init_loss_list
        initialization phase: Losses related to the samples drawn from the
        initialization distribution.
        Only included for method 'parametric_prior'.

    init_prior
        initialized elicit model object including the trainable variables.
        Only included for method 'parametric_prior'.

    init_matrix
        initialization phase: samples drawn from the initialization
        distribution for each hyperparameter.
        Only included for method 'parametric_prior'.

    loss_tensor_expert
        expert term in loss component for computing the discrepancy.

    loss_tensor_model
        simulation-based term in loss component for computing the
        discrepancy.

    Returns
    -------
    save_res_dict :
        dictionary with inclusion/exclusion settings for each sub-result
        in results object.

    Warnings
    --------
    if ``elicited_statistics`` is excluded :func:`elicit.plots.loss` can't be
    used as it requires this information.

    if ``init_matrix`` is excluded :func:`elicit.plots.initialization` can't be
    used as it requires this information.

    if ``prior_samples`` is excluded :func:`elicit.plots.prior_joint` can't be
    used as it requires this information.

    if ``target_quantities`` is excluded :func:`elicit.plots.priorpredictive`
    can't be used as it requires this information.

    if ``expert_elicited_statistics`` or ``elicited_statistics`` is
    excluded :func:`elicit.plots.elicits` can't be used as it requires this
    information.

    """
    if not elicited_statistics:
        warnings.warn(
            "el.plots.loss() requires information about "
            + "'elicited_statistics'. If you don't save this information "
            + "el.plot.loss() can't be used."
        )
    if not init_matrix:
        warnings.warn(
            "el.plots.initialization() requires information about "
            + "'init_matrix'. If you don't save this information "
            + "this plotting function can't be used."
        )
    if not prior_samples:
        warnings.warn(
            "el.plots.priors() requires information about "
            + "'prior_samples'. If you don't save this information "
            + "this plotting function can't be used."
        )
    if not target_quantities:
        warnings.warn(
            "el.plots.priorpredictive() requires information about "
            + "'target_quantities'. If you don't save this information "
            + "this plotting function can't be used."
        )
    if (not expert_elicited_statistics) or (not elicited_statistics):
        warnings.warn(
            "el.plots.elicits() requires information about "
            + "'expert_elicited_statistics' and 'elicited_statistics'. "
            + "If you don't save this information this plotting function"
            + " can't be used."
        )

    save_res_dict: SaveResults = dict(
        target_quantities=target_quantities,
        elicited_statistics=elicited_statistics,
        prior_samples=prior_samples,
        model_samples=model_samples,
        expert_elicited_statistics=expert_elicited_statistics,
        expert_prior_samples=expert_prior_samples,
        init_loss_list=init_loss_list,
        init_prior=init_prior,
        init_matrix=init_matrix,
        loss_tensor_expert=loss_tensor_expert,
        loss_tensor_model=loss_tensor_model,
    )
    return save_res_dict


def clean_savings(
    history: dict[str, Any],
    results: dict[str, Any],
    save_history: SaveHist,
    save_results: SaveResults,
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """
    clean-up result dictionaries

    Parameters
    ----------
    history
        Results that are saved across epochs including among others loss,
        loss_component, time, and hyperparameter.
        See [`save_history`][elicito.utils.save_history] for complete list.

    results
        Results that are saved for the last epoch only including prior_samples,
        elicited_statistics, target_quantities, etc.
        See [`save_results`][elicito.utils.save_results] for complete list.

    save_history
        Exclude or include sub-results in the final result file.
        In the ``history`` object are all results that are saved across epochs.

    save_results
        Exclude or include sub-results in the final result file.
        In the ``results`` object are all results that are saved for the last
        epoch only.

    Returns
    -------
    results, history :
        final results taking in consideration exclusion criteria as specified
        in `save_history` and `save_results`.

    """
    for key_hist in save_history:
        if not save_history[key_hist]:  # type: ignore
            history.pop(key_hist)

    for key_res in save_results:
        if not save_results[key_res]:  # type: ignore
            results.pop(key_res)
    return results, history


def get_expert_datformat(targets: list[Target]) -> dict[str, list[Any]]:
    """
    Inspect which data format for the expert data is expected by the method.

    Parameters
    ----------
    targets
        list of target quantities

    Returns
    -------
    elicit_dict :
        expected format of expert data.

    """
    elicit_dict: dict[str, Any] = dict()
    for tar in targets:
        query = tar["query"]["name"]
        if query == "custom":
            query = tar["query"]["func_name"]
        target = tar["name"]
        key = query + "_" + target
        elicit_dict[key] = list()

    return elicit_dict


def softmax_gumbel_trick(likelihood: Any, upper_thres: float, temp: float = 1.6) -> Any:
    """
    Apply softmax-gumble trick

    The softmax-gumbel trick computes a continuous approximation of ypred from
    a discrete likelihood and thus allows for the computation of gradients for
    discrete random variables.

    Currently, this approach is only implemented for models without upper
    boundary (e.g., Poisson model).

    References
    ----------
    - Maddison, C. J., Mnih, A. & Teh, Y. W. The concrete distribution:
      A continuous relaxation of discrete random variables in International
      Conference on Learning Representations (2017).
      https://doi.org/10.48550/arXiv.1611.00712
    - Jang, E., Gu, S. & Poole, B. Categorical reparameterization with
      gumbel-softmax in International Conference on Learning Representations
      (2017). https://openreview.net/forum?id=rkE3y85ee.
    - Joo, W., Kim, D., Shin, S. & Moon, I.-C. Generalized gumbel-softmax
      gradient estimator for generic discrete random variables. Preprint
      at https://doi.org/10.48550/arXiv.2003.01847 (2020).

    Parameters
    ----------
    likelihood
        shape = [B, num_samples, num_obs, 1]
        likelihood function used in the generative model.
        Must be a tfp.distributions object.

    upper_thres
        upper threshold at which the distribution of the outcome variable is
        truncated. For double-bounded distribution (e.g. Binomial) this is
        simply the "total count" information. Lower-bounded distribution
        (e.g. Poisson) must be truncated to create an artificial
        double-boundedness.

    temp
        temperature hyperparameter of softmax function. A temperature going
        towards zero yields approximates a categorical distribution, while
        a temperature >> 0 approximates a continuous distribution.

    Returns
    -------
    ypred :
        continuously approximated ypred from the discrete likelihood.

    Raise
    -----
    ValueError
        if rank of ``likelihood`` is not 4. The shape of the likelihood obj
        must have an extra final dimension, i.e., (B, num_samples, num_obs, 1),
        for the softmax-gumbel computation. Use for example
        ``tf.expand_dims(mu,-1)`` for expanding the batch-shape of the
        likelihood.

        if likelihood is not in tfp.distributions module. The likelihood
        must be a tfp.distributions object.

    """
    # check rank of likelihood object
    if len(likelihood.batch_shape) != 4:  # noqa: PLR2004
        msg = (
            "The 'likelihood' in the generative model must have",
            " batch_shape = (B, num_samples, num_obs, 1).",
            " The additional final axis is required by the softmax-gumbel",
            " computation. Use for example `tf.expand_dims(mu,-1)` for",
            " expanding the batch-shape of the likelihood.",
        )
        raise ValueError(msg)

    # check value/type of likelihood object
    if likelihood.name.lower() not in dir(tfd):
        msg1: str = "Likelihood in generative model must be a tfp.distribution object."
        raise ValueError(msg1)

    # set seed
    tf.random.set_seed(el.SEED)
    # get batch size, num_samples, num_observations
    B, S, number_obs, _ = likelihood.batch_shape
    # constant outcome vector (including zero outcome)
    thres = upper_thres
    c = tf.range(thres + 1, delta=1, dtype=tf.float32)
    # broadcast to shape (B, rep, outcome-length)
    c_brct = tf.broadcast_to(c[None, None, None, :], shape=(B, S, number_obs, len(c)))
    # compute pmf value
    pi = likelihood.prob(c_brct)
    # prevent underflow
    pi = tf.where(pi < 1.8 * 10 ** (-30), 1.8 * 10 ** (-30), pi)
    # sample from uniform
    u = tfd.Uniform(0, 1).sample((B, S, number_obs, len(c)))
    # generate a gumbel sample from uniform sample
    g = -tf.math.log(-tf.math.log(u))
    # softmax gumbel trick
    w = tf.nn.softmax(
        tf.math.divide(
            tf.math.add(tf.math.log(pi), g),
            temp,
        )
    )
    # reparameterization/linear transformation
    ypred = tf.reduce_sum(tf.multiply(w, c), axis=-1)
    return ypred
