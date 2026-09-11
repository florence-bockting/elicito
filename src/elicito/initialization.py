"""
Hyperparameter initialization for parametric prior
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore
from tqdm import tqdm

import elicito as el
from elicito.exceptions import MissingOptionalDependencyError
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

# Argument names that carry a shape, not a magnitude. A shape does not live on
# the scale of the elicited data, so the pooled spread says nothing about it.
SHAPE_ARGS = frozenset(
    {
        "concentration",
        "concentration0",
        "concentration1",
        "df",
        "power",
        "skewness",
        "tailweight",
    }
)

# Natural range of a shape hyperparameter that the box covers. Below one, a
# Weibull or a Gamma is so heavy-tailed that single draws overflow.
SHAPE_LOW = 1.0
SHAPE_HIGH = 5.0


@dataclass
class InitResult:
    """What an initialization method returns to ``initialize``."""

    prior_model: Any
    candidates: Optional[dict[str, Any]] = None
    losses: Optional[list[Any]] = None


class InitMethod(Protocol):
    """Behaviour that differs between the initialization methods."""

    name: str
    default_iterations: int

    def check(self, initializer: Initializer) -> None:
        """Reject an input this method cannot use."""
        ...

    def skips_search(self, optimizer: dict[str, Any]) -> bool:
        """Whether the training repeats this search, so it can be dropped."""
        ...

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
        """Pick the hyperparameters that start the training."""
        ...

    def dry_run_slice(
        self,
        initializer: Initializer,
        parameters: list[Parameter],
        trainer: Trainer,
    ) -> Any:
        """Return a slice of the right shape for ``Elicit.__init__``."""
        ...


_INIT_METHODS: dict[str, type[InitMethod]] = {}


def get_init_method(name: str) -> InitMethod:
    """Return a new strategy object for an ``initializer["method"]`` string."""
    try:
        method_cls = _INIT_METHODS[name]
    except KeyError:
        msg = (
            "Currently implemented initialization methods are "
            f"{', '.join(repr(key) for key in sorted(_INIT_METHODS))}, but got "
            f"method={name!r} as input."
        )
        raise ValueError(msg) from None
    return method_cls()


def resolve_init_method(initializer: Initializer) -> InitMethod:
    """Return the initialization method that ``initializer`` asks for."""
    # exact values are chosen by their presence, not by a method string
    if initializer["hyperparams"] is not None:
        return ExactValues()

    name = initializer["method"]
    if name is None:
        msg = (
            "Either 'method' or 'hyperparams' has"
            "to be specified. Use method for sampling from an"
            "initialization distribution and 'hyperparams' for"
            "specifying exact initial values per hyperparameter."
        )
        raise ValueError(msg)
    return get_init_method(name)


class ExactValues:
    """Start from hyperparameter values the user supplied."""

    name = "exact"
    default_iterations = 0  # nothing is drawn

    def check(self, initializer: Initializer) -> None:
        """Reject an input this method cannot use."""
        if initializer["hyperparams"] is None:
            msg = "Method 'exact' needs 'hyperparams'."
            raise ValueError(msg)

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
        """Build the prior model from the given values."""
        prior_model = el.simulations.Priors(
            ground_truth=False,
            init_matrix_slice=initializer["hyperparams"],
            trainer=trainer,
            parameters=parameters,
            network=None,
            expert=expert,
            seed=seed,
        )
        return InitResult(prior_model=prior_model)

    def dry_run_slice(
        self,
        initializer: Initializer,
        parameters: list[Parameter],
        trainer: Trainer,
    ) -> Any:
        """Return a slice of the right shape for ``Elicit.__init__``."""
        return initializer["hyperparams"]


def _select_candidate(losses: list[Any], initializer: Initializer) -> int:
    """Return the index of the candidate at the requested loss quantile."""
    # elicit.initializer always sets this; the default keeps the best candidate
    loss_quantile = initializer["loss_quantile"]
    if loss_quantile is None:
        loss_quantile = 0.0

    # A candidate whose loss is not finite must not take part in the
    # selection. Without this, a single NAN makes the percentile NAN, no
    # candidate matches, and the index lookup fails with an error that does
    # not name the cause.
    values = np.asarray(losses, dtype=np.float64).reshape(-1)
    finite = np.flatnonzero(np.isfinite(values))

    if finite.size == 0:
        dist = initializer["distribution"]
        detail = (
            f"The initialization distribution is centred at {dist['mean']} "
            f"with radius {dist['radius']}, on the unconstrained scale. "
            "Re-centre it on the expected hyperparameter values, or "
            "reduce its radius."
            if dist is not None
            else "No initialization distribution is set."
        )
        msg = (
            f"All {values.size} initialization candidates yield a "
            f"non-finite loss, so no start value can be selected. {detail}"
        )
        raise ValueError(msg)

    # pick the candidate closest to the requested quantile of the finite
    # losses. argmin also settles a tie, which an equality test could not.
    target = np.percentile(values[finite], loss_quantile)
    return int(finite[int(np.argmin(np.abs(values[finite] - target)))])


def _check_box(initializer: Initializer) -> None:
    """Reject a box method that has no box to draw from."""
    for name in ("distribution", "iterations"):
        if initializer[name] is None:
            msg = f"If '{name}' is None, then 'method' must also be None."
            raise ValueError(msg)


class BoxSample:
    """Draw candidates from a box and keep one, by its loss quantile."""

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


for _sampler in ("sobol", "lhs", "random"):
    _INIT_METHODS[_sampler] = BoxSample


class _SearchStart:
    """A start value that a derivative-free search picks out of the box."""

    name: str
    default_iterations: int

    def check(self, initializer: Initializer) -> None:
        """Reject an input this method cannot use."""
        _check_box(initializer)

    def skips_search(self, optimizer: dict[str, Any]) -> bool:
        """Whether the training repeats this search, so it can be dropped."""
        return False

    def search(  # noqa: PLR0913
        self,
        expert_elicited_statistics: dict[str, tf.Tensor],
        parameters: list[Parameter],
        trainer: Trainer,
        model: dict[str, Any],
        targets: list[Target],
        expert: ExpertDict,
        distribution: dict[str, Any],
        max_evals: int,
        seed: int,
    ) -> dict[str, Any]:
        """Return one value per hyperparameter, on the unconstrained scale."""
        raise NotImplementedError

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
        """Search for the start values, then build the prior model."""
        distribution = initializer["distribution"]
        iterations = initializer["iterations"]
        if distribution is None or iterations is None:
            # check() rejects this earlier; the guard narrows the type
            msg = f"Method {self.name!r} needs 'distribution' and 'iterations'."
            raise ValueError(msg)

        # a derivative-free search needs no gradient, so it cannot diverge.
        # The copy keeps the user's Elicit object unchanged.
        initializer = dict(initializer)  # type: ignore [assignment]
        if self.skips_search(optimizer):
            logger.info(
                f"{self.name}: the training runs the same search, so the "
                "initialization only reads the box. 'iterations' is not used."
            )
            names = hyper_names(parameters)
            box = build_box(dict(distribution), expert_elicited_statistics, parameters)
            centre = el.warmstart._box_vector(box, names, "mean")
            initializer["hyperparams"] = dict(zip(names, centre))
        else:
            initializer["hyperparams"] = self.search(
                expert_elicited_statistics=expert_elicited_statistics,
                parameters=parameters,
                trainer=trainer,
                model=model,
                targets=targets,
                expert=expert,
                # dict() satisfies the signature; a TypedDict is invariant
                distribution=dict(distribution),
                max_evals=iterations,
                seed=seed,
            )
        return ExactValues().propose(
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

    def dry_run_slice(
        self,
        initializer: Initializer,
        parameters: list[Parameter],
        trainer: Trainer,
    ) -> Any:
        """Return a slice of the right shape for ``Elicit.__init__``."""
        # the dry run only needs a slice of the right shape. The search
        # looks for the real values during `fit`.
        initializer = dict(initializer)  # type: ignore [assignment]
        initializer["method"] = "random"
        return BoxSample().dry_run_slice(initializer, parameters, trainer)


class WarmStart(_SearchStart):
    """Search for a start point with Nelder-Mead, from the box centre."""

    name = "warmstart"
    default_iterations = 100  # objective evaluations, not candidates

    def search(  # noqa: PLR0913
        self,
        expert_elicited_statistics: dict[str, tf.Tensor],
        parameters: list[Parameter],
        trainer: Trainer,
        model: dict[str, Any],
        targets: list[Target],
        expert: ExpertDict,
        distribution: dict[str, Any],
        max_evals: int,
        seed: int,
    ) -> dict[str, Any]:
        """Return one value per hyperparameter, on the unconstrained scale."""
        return el.warmstart.warm_start(
            expert_elicited_statistics=expert_elicited_statistics,
            parameters=parameters,
            trainer=trainer,
            model=model,
            targets=targets,
            expert=expert,
            distribution=distribution,
            max_evals=max_evals,
            seed=seed,
        )


_INIT_METHODS[WarmStart.name] = WarmStart


class CmaEs(_SearchStart):
    """Search for a start point with CMA-ES, over the whole box."""

    name = "cmaes"
    # objective evaluations, not candidates. A global search needs more of
    # them than the local warm start: it spends the first generations on
    # where the good region is, not on the value inside it.
    default_iterations = 500

    def skips_search(self, optimizer: dict[str, Any]) -> bool:
        """Whether the training repeats this search, so it can be dropped."""
        # `optimizer="cmaes"` runs this search again, from the point that
        # this search returns. The second run starts with a new covariance
        # matrix, so it drops what the first one learned. One run over the
        # whole budget is then better, and the box gives it its start point
        # and its step size.
        return bool(optimizer["optimizer"] == el.cmaes.CMAES)

    def search(  # noqa: PLR0913
        self,
        expert_elicited_statistics: dict[str, tf.Tensor],
        parameters: list[Parameter],
        trainer: Trainer,
        model: dict[str, Any],
        targets: list[Target],
        expert: ExpertDict,
        distribution: dict[str, Any],
        max_evals: int,
        seed: int,
    ) -> dict[str, Any]:
        """Return one value per hyperparameter, on the unconstrained scale."""
        return el.cmaes.cma_search(
            expert_elicited_statistics=expert_elicited_statistics,
            parameters=parameters,
            trainer=trainer,
            model=model,
            targets=targets,
            expert=expert,
            distribution=distribution,
            max_evals=max_evals,
            seed=seed,
        )


_INIT_METHODS[CmaEs.name] = CmaEs


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
        User-specified seed as defined in [`trainer`][elicito.elicit.trainer].

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
        [`Elicit`][elicito.Elicit] obj (i.e., `eliobj.parameters`)

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
        User-specified expert data as provided by [`Elicit`][elicito.elicit.Expert].

    initializer
        User-input from [`initializer`][elicito.elicit.initializer].

    parameters
        User-input from [`parameter`][elicito.elicit.parameter].

    trainer
        User-input from [`trainer`][elicito.elicit.trainer].

    optimizer
        User-input from [`optimizer`][elicito.elicit.optimizer]. Used to run
        the warm-up epochs of a candidate.

    model
        User-input from [`model`][elicito.elicit.model].

    targets
        User-input from [`target`][elicito.elicit.target].

    network
        User-input from one of the methods implemented in the
        [`networks`][elicito.networks] module.

    expert
        User-input from [`Expert`][elicito.elicit.Expert].

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
        distribution = build_box(distribution, expert_elicited_statistics, parameters)
        init_matrix = uniform_samples(
            seed=seed,
            hyppar=distribution["hyper"],
            n_samples=initializer["iterations"],  # type: ignore [arg-type]
            method=initializer["method"],  # type: ignore [arg-type]
            mean=distribution["mean"],
            radius=distribution["radius"],
            parameters=parameters,
        )

    epochs: Any
    if progress == 1:
        print("Initialization")
        epochs = tqdm(range(initializer["iterations"]))  # type: ignore [arg-type]
    else:
        epochs = range(initializer["iterations"])  # type: ignore [arg-type]

    # a candidate is scored by its loss after `warmup_epochs` training epochs.
    # `0` scores it at epoch 0, which is the previous behaviour.
    warmup_epochs = int(initializer.get("warmup_epochs", 0) or 0)

    for i in epochs:
        # update seed
        seed_copy = seed_copy + 1
        # extract initial hyperparameter value for each run
        init_matrix_slice = {f"{key}": init_matrix[key][i] for key in init_matrix}
        # initialize prior distributions based on initial hyperparameters
        prior_model = el.simulations.Priors(
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

            history, _ = el.optimization.sgd_training(
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
                el.utils.one_forward_simulation(
                    prior_model=prior_model, model=model, targets=targets, seed=seed
                )
            )

            # compute discrepancy between expert elicited statistics and
            # simulated data corresponding to initial hyperparameter values
            (loss, *_) = el.losses.total_loss(
                elicit_training=training_elicited_statistics,
                elicit_expert=expert_elicited_statistics,
                targets=targets,
            )

            # A quantile query hides an overflow: the 95% quantile of a sample
            # with a few infinite draws is still finite. A candidate that
            # overflows must not be selected, so mark it as failed here.
            if not el.utils.all_finite(target_quantities):
                loss = tf.fill(tf.shape(loss), tf.constant(np.nan, loss.dtype))
        # save loss value, initial hyperparameter values and initialized prior
        # model for each run
        init_var_list.append(prior_model)
        save_prior.append(prior_model.trainable_variables)
        loss_list.append(loss.numpy())
    if progress == 1:
        print(" ")

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


def init_prior(  # noqa: PLR0913
    expert_elicited_statistics: dict[str, tf.Tensor],
    initializer: Optional[Initializer],
    parameters: list[Parameter],
    trainer: Trainer,
    optimizer: dict[str, Any],
    model: dict[str, Any],
    targets: list[Target],
    network: Optional[NFDict],
    expert: ExpertDict,
    seed: int,
    progress: int,
) -> tuple[Any, Any, Any]:
    """
    Extract target loss and initialize prior model

    Parameters
    ----------
    expert_elicited_statistics
        Expert-elicited statistics

    initializer
        Initialization of hyperparameter values

    parameters
        Specification of model parameters

    trainer
        Specification of trainer settings for the optimization process

    optimizer
        User-input from [`optimizer`][elicito.elicit.optimizer]. Used to run
        the warm-up epochs of a candidate.

    model
        Generative model

    targets
        Elicitation techniques and target quantities

    network
        Generative model for learning non-parametric priors

    expert
        Expert specification

    seed
        Internally used seed for reproducible results

    progress
        whether progress should be printed or muted

    Returns
    -------
    init_prior_model :
        initialized priors that will be used for the training phase.

    loss_list :
        list with all losses computed for each initialization run.

    init_matrix :
        dictionary with *keys* being the hyperparameter names and *values*
        being the drawn initial values per run.

    """
    return el.methods.get_method(trainer["method"]).initialize(
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


def uniform(
    radius: Union[float, list[float]] = 1.0,
    mean: Union[float, list[float]] = 0.0,
    hyper: Optional[list[str]] = None,
) -> dict[Any, Any]:
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
        List of hyperparameter names as specified in [`hyper`][elicito.elicit.hyper].
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

    init_dict = dict(radius=radius, mean=mean, hyper=hyper)

    return init_dict


def hyper_names(parameters: list[Parameter]) -> list[str]:
    """
    List the hyperparameter names in the order the initializer uses

    Parameters
    ----------
    parameters
        List including dictionary with all information about the
        (hyper-)parameters.

    Returns
    -------
    names :
        Hyperparameter names, in the order of ``parameters``.

    """
    names: list[str] = []
    for param in parameters:
        hyperparams = param["hyperparams"]
        if hyperparams is None:
            continue
        for hyp in hyperparams:
            names.append(hyperparams[hyp]["name"])
    return names


def build_box(
    distribution: dict[str, Any],
    expert_elicited_statistics: dict[str, Any],
    parameters: list[Parameter],
) -> dict[str, Any]:
    """
    Return the concrete initialization box

    A box from [`from_elicits`][elicito.initialization.from_elicits] is
    deferred, and is built here from the expert data. Any other box is
    returned unchanged.

    Parameters
    ----------
    distribution
        Initialization box, as stored in the initializer.

    expert_elicited_statistics
        Elicited statistics of the expert.

    parameters
        List including dictionary with all information about the
        (hyper-)parameters.

    Returns
    -------
    box :
        Box with a concrete ``mean``, ``radius`` and ``hyper``.

    """
    if distribution.get("from_elicits", False):
        return _from_elicits_box(
            expert_elicited_statistics, parameters, distribution["factor"]
        )
    return distribution


def _from_elicits_box(
    expert_elicited_statistics: dict[str, Any],
    parameters: list[Parameter],
    factor: float = 2.0,
) -> dict[Any, Any]:
    """
    Derive a uniform initialization box from the expert data

    Pools all elicited statistics into one location and one spread. The box
    of a hyperparameter then follows its role:

    - An unbounded hyperparameter is a location. It is centred at the pooled
      median, with radius ``factor * spread``.
    - A lower-bounded hyperparameter whose name is in ``SHAPE_ARGS`` is a
      shape. A shape has no relation to the scale of the data, so its box
      covers the natural range ``SHAPE_LOW`` to ``SHAPE_HIGH``.
    - Any other lower-bounded hyperparameter is a magnitude. Its box spans
      from ``spread / 100`` up to the pooled 95% quantile, because it can be
      a small prior scale or a scale as large as the elicited data.

    Pooling all targets is crude. The box is correct in order of
    magnitude only. That is enough to avoid a start value that is wrong
    by a factor of ten.

    Parameters
    ----------
    expert_elicited_statistics
        Elicited statistics of the expert, as passed to
        [`init_prior`][elicito.initialization.init_prior].

    parameters
        List including dictionary with all information about the
        (hyper-)parameters.

    factor
        Multiplier of the pooled spread. The default is ``2.``.

    Returns
    -------
    init_dict :
        Dictionary with all settings of the uniform distribution, as
        returned by [`uniform`][elicito.initialization.uniform].

    """
    pooled = np.concatenate(
        [
            np.reshape(np.asarray(v, dtype=np.float32), -1)
            for v in expert_elicited_statistics.values()
        ]
    )
    q25, median, q75, q95 = np.percentile(pooled, [25.0, 50.0, 75.0, 95.0])
    spread = float(max((q75 - q25) / 1.35, 1e-3))
    upper = float(max(q95, spread))

    forward = el.utils.LowerBound(lower=0.0).forward
    shape_low = float(forward(SHAPE_LOW))
    shape_high = float(forward(SHAPE_HIGH))
    magnitude_low = float(forward(max(spread / 100.0, 1e-3)))
    magnitude_high = float(forward(upper))

    hyper = hyper_names(parameters)
    mean: list[float] = []
    radius: list[float] = []
    for param in parameters:
        hyperparams = param["hyperparams"]
        if hyperparams is None:
            continue
        for hyp in hyperparams:
            if hyperparams[hyp]["constraint_name"] != "softplusL":
                mean.append(float(median))
                radius.append(factor * spread)
            elif hyp in SHAPE_ARGS:
                mean.append((shape_low + shape_high) / 2.0)
                radius.append((shape_high - shape_low) / 2.0)
            else:
                mean.append((magnitude_low + magnitude_high) / 2.0)
                radius.append((magnitude_high - magnitude_low) / 2.0)

    return uniform(radius=radius, mean=mean, hyper=hyper)


def from_elicits(factor: float = 2.0) -> Uniform:
    """
    Derive the initialization box from the expert data

    The box cannot be built before ``fit``, because the expert statistics
    of [`expert.simulator`][elicito.elicit.expert] do not exist yet. This
    function only records the request. [`init_prior`]
    [elicito.initialization.init_prior] builds the box.

    Parameters
    ----------
    factor
        Multiplier of the pooled spread. The default is ``2.``.

    Returns
    -------
    init_dict :
        Dictionary marking the initialization box as deferred.

    """
    return dict(radius=0.0, mean=0.0, hyper=None, from_elicits=True, factor=factor)
