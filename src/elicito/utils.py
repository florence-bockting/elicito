"""
helper functions for setting up the Elicit object
"""

import logging
from typing import Any, Callable, Optional

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

# the names with noqa are re-exported, so that el.utils.<name> stays valid
from elicito.models import (
    all_finite,  # noqa: F401
    nonfinite_fraction,  # noqa: F401
    one_forward_simulation,
    simulate_and_elicit,  # noqa: F401
)
from elicito.parameters.priors import Priors
from elicito.types import (
    ExpertDict,
    NFDict,
    Parallel,
    Parameter,
    Target,
    Trainer,
)

tfd = tfp.distributions
logger = logging.getLogger(__name__)

# Seed of the current run. Elicit sets it before a run, and
# gumbel_softmax_trick reads it.
SEED = 0


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


def add_derived(samples: Any, **derived: Callable[[Any], Any]) -> None:
    """
    Add derived parameters to the prior samples

    A derived parameter is a function of the model parameters, computed in the
    generative model and not sampled. It is therefore not in the prior group
    of the samples. This function computes it from the prior samples and
    stores it there. The plotting functions can then select it by name.

    Parameters
    ----------
    samples
        result of :func:`elicito.elicit.Elicit.sample`. The prior group is changed
        in place.

    **derived
        one function per derived parameter, named by the argument. Each
        function gets the prior samples as an ``xarray.Dataset`` and returns
        the derived samples.

    Examples
    --------
    >>> samples = eliobj.sample()  # doctest: +SKIP
    >>> el.utils.add_derived(  # doctest: +SKIP
    ...     samples,
    ...     h1=lambda prior: prior["hts"] + prior["dh"],
    ... )
    >>> el.plots.prior_marginals(  # doctest: +SKIP
    ...     eliobj, params=["h1", "hts"], samples=samples
    ... )

    Raises
    ------
    KeyError
        Can't find 'prior' in the samples.

    ValueError
        A name in ``derived`` is already a model parameter.

    """
    try:
        prior = samples["prior"]
    except KeyError:
        raise KeyError(  # noqa: TRY003
            "No 'prior' group found. Pass the result of 'eliobj.sample()'."
        )

    model_params = prior.attrs["model_parameters"]
    # `to_dataset` copies the samples out of the DataTree node, so the new
    # variables are written back with the setter below
    prior_samples = prior.to_dataset()

    for name, function in derived.items():
        if name in model_params:
            raise ValueError(
                f"'{name}' is a model parameter. A derived parameter needs"
                + " a name of its own."
            )
        if name in prior_samples.data_vars:
            logger.info(f"Replacing the derived parameter '{name}'.")
        prior_samples[name] = function(prior_samples)

    samples["prior"].dataset = prior_samples


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
        if query == "pearson_correlation":
            key = "cor_" + target
        else:
            key = query + "_" + target
        elicit_dict[key] = list()

    return elicit_dict


def gumbel_softmax_trick(likelihood: Any, upper_thres: float, temp: float = 1.6) -> Any:
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
            "The 'likelihood' in the generative model must have "
            "batch_shape = (B, num_samples, num_obs, 1). "
            "The additional final axis is required by the softmax-gumbel "
            "computation. Use for example `tf.expand_dims(mu,-1)` for "
            "expanding the batch-shape of the likelihood."
        )
        raise ValueError(msg)

    # set seed
    tf.random.set_seed(SEED)
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


def compute_num_weights(num_NN_weights: list[tf.TensorShape]) -> int:
    """
    Compute number of weights of a tf.keras model.

    Parameters
    ----------
    num_NN_weights :
        list of tf.TensorShape objects of each layer in the model.

    Returns
    -------
    :
        number of weights of the model (incl. biases)
    """
    return sum(
        int(tf.reduce_prod([d if d is not None else 1 for d in shape]))
        for shape in num_NN_weights
    )
