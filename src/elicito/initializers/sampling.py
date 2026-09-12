"""
Initialization by sampling candidates from a box
"""

import logging
from collections.abc import Iterable
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito import models
from elicito._progress import ProgressTable
from elicito.exceptions import MissingOptionalDependencyError
from elicito.initializers._base import InitResult, _check_box, _select_candidate
from elicito.losses import total_loss
from elicito.optimizers import sgd
from elicito.parameters.priors import Priors
from elicito.types import (
    ExpertDict,
    Initializer,
    NFDict,
    Parameter,
    Target,
    Trainer,
    Uniform,
)

tfd = tfp.distributions
logger = logging.getLogger(__name__)


class BoxSample:
    """Draw candidates from a box and keep that with minimum loss."""

    name = "box"
    default_iterations = 32

    def check(self, initializer: Initializer) -> None:
        """Reject an input this method cannot use."""
        _check_box(initializer)

    def skips_search(self, optimizer: dict[str, Any]) -> bool:
        """Whether the training repeats this search, so it can be dropped."""
        return False

    def propose(  # noqa: PLR0913
        self,
        expert_elicited_statistics: dict[str, tf.Tensor],
        initializer: Initializer,
        parameters: list[Parameter],
        trainer: Trainer,
        optimizer: dict[str, Any],
        model: dict[str, Any],
        targets: list[Target],
        network: Optional[NFDict],
        expert: ExpertDict,
        seed: int,
        progress: int,
    ) -> InitResult:
        """Score every candidate, then keep one."""
        loss_list, init_var_list, init_matrix = init_runs(
            expert_elicited_statistics=expert_elicited_statistics,
            initializer=initializer,
            parameters=parameters,
            trainer=trainer,
            optimizer=optimizer,
            model=model,
            targets=targets,
            network=network,
            expert=expert,
            seed=seed,
            progress=progress,
        )
        idx = _select_candidate(loss_list, initializer)
        return InitResult(
            prior_model=init_var_list[idx],
            candidates=init_matrix,
            losses=loss_list,
        )

    def dry_run_slice(
        self,
        initializer: Initializer,
        parameters: list[Parameter],
        trainer: Trainer,
    ) -> Any:
        """Return a slice of the right shape for ``Elicit.__init__``."""
        init_matrix = uniform_samples(
            seed=trainer["seed"],
            hyppar=initializer["distribution"]["hyper"],  # type: ignore [index, arg-type]
            n_samples=initializer["iterations"],  # type: ignore [arg-type]
            method=initializer["method"],  # type: ignore [arg-type]
            mean=initializer["distribution"]["mean"],  # type: ignore [index]
            radius=initializer["distribution"]["radius"],  # type: ignore [index]
            parameters=parameters,
        )
        return {f"{key}": init_matrix[key][0] for key in init_matrix}


def uniform_samples(  # noqa: PLR0913, PLR0912, PLR0915
    seed: int,
    hyppar: list[str],
    n_samples: int,
    method: str,
    mean: Union[float, Iterable[float]],
    radius: Union[float, Iterable[float]],
    parameters: list[Parameter],
) -> dict[str, Any]:
    """
    Sample from uniform distribution for each hyperparameter.

    Parameters
    ----------
    seed
        User-specified seed as defined in [`trainer`][elicito.specs.trainer].

    hyppar
        List of hyperparameter names (strings) declaring the order for the
        list of **means** and **radius**.
        If **means** and **radius** are each a float, then this number is
        applied to all hyperparameter such that no order of hyperparameter
        needs to be specified. In this case ``hyppar = None``

    n_samples
        Number of samples from the uniform distribution for each
        hyperparameter.

    method
        Name of sampling method used for drawing samples from uniform.
        Currently implemented are "random", "lhs", and "sobol".

    mean
        Specification of the uniform distribution. The uniform distribution
        ranges from (`mean - radius`) to (`mean + radius`).

    radius
        Specification of the uniform distribution. The uniform distribution
        ranges from (`mean - radius`) to (`mean + radius`).

    parameters
        List including dictionary with all information about the (hyper-)parameters.
        Can be retrieved as attribute from the initialized
        [`Elicit`][elicito.elicit.Elicit] obj (i.e., `eliobj.parameters`)

    Raises
    ------
    ValueError
        ``method`` must be either "sobol", "lhs", or "random".
        ``n_samples`` must be a positive integer
    TypeError
        arises if ``method`` is not a string.

    Returns
    -------
    res_dict :
        dictionary with *keys* being the hyperparameters and *values* the
        samples from the uniform distribution.

    """
    # set seed
    tf.random.set_seed(seed)

    # Validate n_samples
    if not isinstance(n_samples, int) or n_samples <= 0:
        msg = "n_samples must be a positive integer."
        raise ValueError(msg)

    # Validate method
    if not isinstance(method, str):
        msg = "method must be a string."  # type: ignore [unreachable]
        raise TypeError(msg)

    if method not in ["sobol", "lhs", "random"]:
        msg = "Unsupported method. Choose from 'sobol', 'lhs', or 'random'."
        raise ValueError(msg)

    if method in ("sobol", "lhs"):
        try:
            from scipy.stats import qmc
        except ImportError as exc:
            raise MissingOptionalDependencyError("scipy", requirement="scipy") from exc

    # counter number of hyperparameters
    n_hypparam = 0
    name_hyper: list[str] = []
    res_dict = dict()

    if hyppar is None:
        if type(mean) is list:  # type: ignore [unreachable]
            msg = (
                "If different mean values should be specified per "
                "hyperparameter, the hyppar argument cannot be None."
            )
            raise ValueError(msg)
        if type(radius) is list:
            msg = (
                "If different radius values should be specified per "
                "hyperparameter, the hyppar argument cannot be None."
            )
            raise ValueError(msg)
        for i in range(len(parameters)):
            for hyperparam in parameters[i]["hyperparams"]:
                dim = parameters[i]["hyperparams"][hyperparam]["dim"]
                name = parameters[i]["hyperparams"][hyperparam]["name"]
                n_hypparam += dim
                for j in range(dim):
                    name_hyper.append(name)

        # make sure type is correct
        mean = tf.cast(mean, tf.float32)
        radius = tf.cast(radius, tf.float32)

        sampler: Union[qmc.LatinHypercube, qmc.Sobol]
        # Generate samples based on the chosen method
        if method == "sobol":
            sampler = qmc.Sobol(d=n_hypparam, seed=seed)
            sample_data = sampler.random(n=n_samples)
        elif method == "lhs":
            sampler = qmc.LatinHypercube(d=n_hypparam, seed=seed)
            sample_data = sampler.random(n=n_samples)
        elif method == "random":
            uniform_samples = tfd.Uniform(
                tf.subtract(mean, radius), tf.add(mean, radius)
            ).sample((n_samples, n_hypparam))
        # Inverse transform
        if method in ("sobol", "lhs"):
            sample_dat = tf.cast(tf.convert_to_tensor(sample_data), tf.float32)
            uniform_samples = tfd.Uniform(
                tf.subtract(mean, radius), tf.add(mean, radius)
            ).quantile(sample_dat)
        # store initialization results per hyperparameter
        for j, name in zip(range(n_hypparam), name_hyper):
            res_dict[name] = uniform_samples[:, j]
    else:
        if (type(mean) is not list) or (type(radius) is not list):
            msg = (
                "mean and radius arguments of function uniform_samples "
                "must be of type list."
            )
            raise ValueError(msg)

        # One design over all hyperparameters, as in the branch above. A
        # separate one-dimensional sequence per hyperparameter runs in nearly
        # the same order for each of them, which correlates the columns and
        # leaves the candidates on a diagonal of the box.
        if method == "sobol":
            sampler = qmc.Sobol(d=len(hyppar), seed=seed)
            sample_data = sampler.random(n=n_samples)
        elif method == "lhs":
            sampler = qmc.LatinHypercube(d=len(hyppar), seed=seed)
            sample_data = sampler.random(n=n_samples)

        for column, (i, j, n) in enumerate(zip(mean, radius, hyppar)):
            i_casted = tf.cast(i, tf.float32)
            j_casted = tf.cast(j, tf.float32)

            if method == "random":
                uniform_samples = tfd.Uniform(
                    tf.subtract(i_casted, j_casted),
                    tf.add(i_casted, j_casted),
                ).sample((n_samples,))
            else:
                tensor_data = tf.convert_to_tensor(sample_data[:, column])
                # Inverse transform
                sample_dat = tf.cast(tensor_data, tf.float32)
                uniform_samples = tfd.Uniform(
                    tf.subtract(i_casted, j_casted),
                    tf.add(i_casted, j_casted),
                ).quantile(sample_dat)

            res_dict[n] = uniform_samples
    return res_dict


def init_runs(  # noqa: PLR0913
    expert_elicited_statistics: dict[str, tf.Tensor],
    initializer: Initializer,
    parameters: list[Parameter],
    trainer: Trainer,
    optimizer: dict[str, Any],
    model: dict[str, Any],
    targets: list[Target],
    network: Optional[NFDict],
    expert: ExpertDict,
    seed: int,
    progress: int,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """
    Compute the discrepancy between expert data and simulated data

    Discrepancy for multiple hyperparameter initialization values.

    Parameters
    ----------
    expert_elicited_statistics
        User-specified expert data as provided by [`Elicit`][elicito.specs.Expert].

    initializer
        User-input from [`initializer`][elicito.specs.initializer].

    parameters
        User-input from [`parameter`][elicito.specs.parameter].

    trainer
        User-input from [`trainer`][elicito.specs.trainer].

    optimizer
        User-input from [`optimizer`][elicito.specs.optimizer]. Used to run
        the warm-up epochs of a candidate.

    model
        User-input from [`model`][elicito.specs.model].

    targets
        User-input from [`target`][elicito.specs.target].

    network
        User-input from one of the methods implemented in the
        [`networks`][elicito.networks] module.

    expert
        User-input from [`Expert`][elicito.specs.Expert].

    seed
        internal seed for reproducible results

    progress
        progress is muted if `progress=0`.
        progress is printed if `progress=1`

    Returns
    -------
    loss_list :
        list with all losses computed for each initialization run.

    init_var_list :
        list with initializer prior model for each run.

    init_matrix :
        dictionary with *keys* being the hyperparameter names and *values*
        being the drawn initial values per run.

    """
    # create a copy of the seed variable for incremental increase of seed
    # for each initialization run. It stays a Python integer: a tensor seed
    # reaches `tf.random.set_seed`, and the graph of a compiled forward pass
    # then cannot read the global seed.
    seed_copy = int(seed)
    # set seed
    tf.random.set_seed(seed)
    # initialize saving of results
    loss_list = []
    init_var_list = []
    save_prior = []

    # sample initial values
    distribution: Any = initializer["distribution"]
    if distribution is not None:
        init_matrix = uniform_samples(
            seed=seed,
            hyppar=distribution["hyper"],
            n_samples=initializer["iterations"],  # type: ignore [arg-type]
            method=initializer["method"],  # type: ignore [arg-type]
            mean=distribution["mean"],
            radius=distribution["radius"],
            parameters=parameters,
        )

    epochs = range(initializer["iterations"])  # type: ignore [arg-type]
    bar = ProgressTable(
        "Initialization",
        total=initializer["iterations"],  # type: ignore [arg-type]
        disable=progress != 1,
        loss=float("nan"),
    )

    # a candidate is scored by its loss after `warmup_epochs` training epochs.
    # `0` scores it at epoch 0, which is the previous behaviour.
    warmup_epochs = int(initializer.get("warmup_epochs", 0) or 0)

    for i in epochs:
        # update seed
        seed_copy = seed_copy + 1
        # extract initial hyperparameter value for each run
        init_matrix_slice = {f"{key}": init_matrix[key][i] for key in init_matrix}
        # initialize prior distributions based on initial hyperparameters
        prior_model = Priors(
            ground_truth=False,
            init_matrix_slice=init_matrix_slice,
            trainer=trainer,
            parameters=parameters,
            network=network,
            expert=expert,
            seed=seed_copy,
        )

        if warmup_epochs > 0:
            # a low loss at epoch 0 does not show whether the trajectory is
            # stable. The candidate keeps its trained values; init_matrix
            # records the drawn values from before the warm-up.
            warmup_trainer = trainer.copy()
            warmup_trainer["epochs"] = warmup_epochs
            warmup_trainer["progress"] = 0

            history, _ = sgd.sgd_training(
                expert_elicited_statistics=expert_elicited_statistics,
                prior_model_init=prior_model,
                trainer=warmup_trainer,
                optimizer=optimizer,
                model=model,
                targets=targets,
                parameters=parameters,
                seed=seed_copy,
                progress=0,
            )
            # sgd_training stores a scalar; the epoch-0 branch and
            # _outputs.create_init_group both expect shape (1,)
            loss = tf.reshape(history["loss"][-1], (1,))
        else:
            # simulate from priors and generative model and compute the
            # elicited statistics corresponding to the initial hyperparameters
            (training_elicited_statistics, _, _, target_quantities) = (
                models.one_forward_simulation(
                    prior_model=prior_model, model=model, targets=targets, seed=seed
                )
            )

            # compute discrepancy between expert elicited statistics and
            # simulated data corresponding to initial hyperparameter values
            (loss, *_) = total_loss(
                elicit_training=training_elicited_statistics,
                elicit_expert=expert_elicited_statistics,
                targets=targets,
            )

            # A quantile query hides an overflow: the 95% quantile of a sample
            # with a few infinite draws is still finite. A candidate that
            # overflows must not be selected, so mark it as failed here.
            if not models.all_finite(target_quantities):
                loss = tf.fill(tf.shape(loss), tf.constant(np.nan, loss.dtype))
        # save loss value, initial hyperparameter values and initialized prior
        # model for each run
        init_var_list.append(prior_model)
        save_prior.append(prior_model.trainable_variables)
        loss_list.append(loss.numpy())
        bar.update(loss=float(tf.squeeze(loss)))
    bar.close()

    # A candidate with a non-finite loss cannot be used as a start value. It
    # is kept in the list, so that loss_list stays aligned with init_matrix
    # for the initialization plot. Selection skips it.
    n_failed = int(np.sum(~np.isfinite(np.asarray(loss_list, dtype=np.float64))))
    if n_failed > 0:
        logger.info(
            f"{n_failed} of {len(loss_list)} initialization candidates yield a"
            " non-finite loss. They are excluded from the selection of the"
            " start value."
        )

    return loss_list, init_var_list, init_matrix


def uniform(
    radius: Union[float, list[float]] = 1.0,
    mean: Union[float, list[float]] = 0.0,
    hyper: Optional[list[str]] = None,
) -> Uniform:
    """
    Specify uniform initialization distribution

    specify uniform used for drawing initial values for each hyperparameter.
    Initial values are drawn from a uniform distribution
    ranging from ``mean - radius`` to ``mean + radius``.

    Parameters
    ----------
    radius
        Initial values are drawn from a uniform distribution ranging from
        ``mean - radius`` to ``mean + radius``.
        If a ``float`` is provided the same setting will be used for all
        hyperparameters.
        If different settings per hyperparameter are required, a ``list`` of
        length equal to the number of hyperparameters should be provided.
        The order of values should be equivalent to the order of hyperparameter
        names provided in **hyper**.
        The default is ``1.``.

    mean
        Initial values are drawn from a uniform distribution ranging from
        ``mean - radius`` to ``mean + radius``.
        If a ``float`` is provided the same setting will be used for all
        hyperparameters.
        If different settings per hyperparameter are required, a ``list`` of
        length equal to the number of hyperparameters should be provided.
        The order of values should be equivalent to the order of hyperparameter
        names provided in **hyper**.
        The default is ``0.``.

    hyper
        List of hyperparameter names as specified in [`hyper`][elicito.specs.hyper].
        The values provided in **radius** and **mean** should follow the order
        of hyperparameters indicated in this list.
        If a float is passed to **radius** and **mean** this argument is not
        necessary.

    Raises
    ------
    AssertionError
        ``hyper``, ``mean``, and ``radius`` must have the same length.

    Returns
    -------
    init_dict :
        Dictionary with all seetings of the uniform distribution used for
        initializing the hyperparameter values.

    """
    if hyper is not None:
        if len(hyper) != len(mean):  # type: ignore [arg-type]
            msg = "`hyper`, `mean`, and `radius` must have the same length."
            raise AssertionError(msg)

    init_dict = Uniform(radius=radius, mean=mean, hyper=hyper)

    return init_dict
