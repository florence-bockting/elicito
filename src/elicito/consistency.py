"""
Check whether the expert information is consistent

All targets come from the same prior and the same generative model. The
information that an expert gives about one target can therefore contradict
the information about another target. No prior then fits every target, and
the fit is a compromise between them.

The functions run on a fitted eliobj:

- `detect` tests, per target, whether the fit is worse than the sampling
  error explains.
- `slack` refits with a free shift per target, and shows which targets need
  a shift to fit, and how large it is.
- `calibrate_penalty` finds the penalty of `slack` that keeps every shift
  of a consistent expert at zero.
- `best_replication` finds the replication with the lowest loss. Both checks
  need a fit at the best optimum: a local optimum also fits badly.
"""

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorflow as tf
import xarray as xr

from elicito import losses, models, utils

if TYPE_CHECKING:
    from elicito.elicit import Elicit

# number of prior samples of a pseudo-expert, if the expert gives data
EXPERT_NUM_SAMPLES = 10_000


def _check_fitted(eliobj: "Elicit") -> None:
    if not hasattr(eliobj, "results"):
        msg = "No results found. Run 'eliobj.fit()' first."
        raise AttributeError(msg)
    if eliobj.trainer["method"] != "parametric_prior":
        msg = "Only method='parametric_prior' is supported."
        raise ValueError(msg)


def _replication_seed(eliobj: "Elicit", replication: int) -> int:
    return int(eliobj.results.history_stats.seed_replication.values[replication])


def _target_losses(
    eliobj: "Elicit", training: dict[str, tf.Tensor], expert: dict[str, tf.Tensor]
) -> list[tf.Tensor]:
    """Loss of each target, multiplied by its weight"""
    _, individual, _, _ = losses.total_loss(
        elicit_training=training, elicit_expert=expert, targets=eliobj.targets
    )
    return [loss * target["weight"] for loss, target in zip(individual, eliobj.targets)]


def _learned_prior(eliobj: "Elicit", replication: int) -> tuple[Any, int]:
    """Prior with the learned hyperparameters, and the seed of its replication"""
    seed = _replication_seed(eliobj, replication)
    trainer = {**eliobj.trainer, "seed": seed}
    return eliobj._learned_prior_model(replication, trainer, seed), seed


def _training_statistics(eliobj: "Elicit", replication: int) -> dict[str, tf.Tensor]:
    """Elicited statistics of the learned prior, with the draws of the fit"""
    prior_model, seed = _learned_prior(eliobj, replication)
    training, *_ = models.one_forward_simulation(
        prior_model, eliobj.model, eliobj.targets, seed
    )
    return training


def _expert_statistics(eliobj: "Elicit") -> dict[str, tf.Tensor]:
    expert, _ = utils.get_expert_data(
        eliobj.trainer,
        eliobj.model,
        eliobj.targets,
        eliobj.expert,
        eliobj.parameters,
        eliobj.network,
        eliobj.trainer["seed"],
    )
    return expert  # type: ignore [no-any-return]


def _pseudo_experts(
    eliobj: "Elicit", replication: int, num_draws: int, num_samples: int, seed: int
) -> Iterator[dict[str, tf.Tensor]]:
    """Elicited statistics of pseudo-experts that simulate from the learned prior"""
    # one prior model for all pseudo-experts; the prior reads the seed from
    # its trainer on every call
    trainer = {**eliobj.trainer, "B": 1, "num_samples": num_samples, "seed": seed}
    pseudo_prior = eliobj._learned_prior_model(replication, trainer, seed)
    for m in range(num_draws):
        pseudo_prior.trainer["seed"] = seed + m
        pseudo_expert, *_ = models.one_forward_simulation(
            pseudo_prior, eliobj.model, eliobj.targets, seed + m
        )
        yield pseudo_expert


def _expert_ranges(expert: dict[str, tf.Tensor]) -> dict[str, float]:
    """Range of the expert statistics of each target, the unit of a shift"""
    ranges = {}
    for name, value in expert.items():
        spread = float(tf.reduce_max(value) - tf.reduce_min(value))
        ranges[name] = spread if spread > 0 else 1.0
    return ranges


def best_replication(eliobj: "Elicit") -> int:
    """
    Find the replication with the lowest loss

    The loss is computed at the stored hyperparameters of each replication,
    with the draws of its fit. A replication whose loss is not finite is
    never the best.

    Parameters
    ----------
    eliobj
        fitted eliobj with ``method="parametric_prior"``

    Returns
    -------
    :
        index of the replication

    Raises
    ------
    AttributeError
        eliobj has not been fitted yet.

    ValueError
        The method is not ``"parametric_prior"``.

    Examples
    --------
    >>> el.consistency.best_replication(eliobj)  # doctest: +SKIP
    """
    _check_fitted(eliobj)
    expert = _expert_statistics(eliobj)
    num_replications = len(eliobj.results.history_stats.seed_replication.values)
    total = np.full(num_replications, np.inf)
    for i in range(num_replications):
        loss, *_ = losses.total_loss(
            elicit_training=_training_statistics(eliobj, i),
            elicit_expert=expert,
            targets=eliobj.targets,
        )
        value = float(tf.squeeze(loss))
        if np.isfinite(value):
            total[i] = value
    return int(np.argmin(total))


def detect(
    eliobj: "Elicit",
    num_draws: int = 100,
    num_samples: int | None = None,
    seed: int = 0,
    replication: int | None = None,
) -> xr.Dataset:
    """
    Test, per target, whether the learned prior fits the expert information

    The test compares the loss of each target with its loss under consistent
    information. The learned prior itself gives consistent information: a
    pseudo-expert simulates the elicited statistics from it. Each of the
    **num_draws** pseudo-experts uses a new seed. The loss against the
    pseudo-experts shows how large the loss of a target is from sampling
    error alone.

    A small `p_value` means that the target fits worse than the sampling
    error explains. Three causes are possible: the expert information on that
    target contradicts the information on the other targets, the model cannot
    express it, or the fit stopped in a local optimum. Run the test on the
    best of several replications to make the last cause unlikely.

    Parameters
    ----------
    eliobj
        fitted eliobj with ``method="parametric_prior"``

    num_draws
        number of pseudo-experts

    num_samples
        number of prior samples of a pseudo-expert. Default is the
        ``num_samples`` of a simulated expert, else 10 000.

    seed
        seed of the first pseudo-expert. Pseudo-expert ``m`` uses
        ``seed + m``.

    replication
        index of the replication. Default is the replication with the lowest
        loss, see `best_replication`.

    Returns
    -------
    :
        xr.Dataset with the dimension ``target`` and the variables
        ``observed`` (loss against the expert), ``null_mean`` and ``null_q95``
        (mean and 95% quantile of the loss against the pseudo-experts), and
        ``p_value``. All losses include the weight of the target.

    Raises
    ------
    AttributeError
        eliobj has not been fitted yet.

    ValueError
        The method is not ``"parametric_prior"``.

    Examples
    --------
    >>> el.consistency.detect(eliobj)  # doctest: +SKIP
    """
    _check_fitted(eliobj)
    if replication is None:
        replication = best_replication(eliobj)
    if num_samples is None:
        num_samples = eliobj.expert.get("num_samples", EXPERT_NUM_SAMPLES)

    training = _training_statistics(eliobj, replication)
    expert = _expert_statistics(eliobj)
    observed = np.array(
        [float(loss) for loss in _target_losses(eliobj, training, expert)]
    )

    null = np.zeros((num_draws, len(eliobj.targets)))
    pseudo_experts = _pseudo_experts(eliobj, replication, num_draws, num_samples, seed)
    for m, pseudo_expert in enumerate(pseudo_experts):
        null[m] = [
            float(loss) for loss in _target_losses(eliobj, training, pseudo_expert)
        ]

    p_value = (1 + (null >= observed).sum(axis=0)) / (1 + num_draws)
    names = [target["name"] for target in eliobj.targets]
    return xr.Dataset(
        data_vars=dict(
            observed=("target", observed),
            null_mean=("target", null.mean(axis=0)),
            null_q95=("target", np.quantile(null, 0.95, axis=0)),
            p_value=("target", p_value),
        ),
        coords=dict(target=names),
        attrs=dict(
            num_draws=num_draws,
            num_samples=num_samples,
            seed=seed,
            replication=replication,
        ),
    )


def calibrate_penalty(  # noqa: PLR0913
    eliobj: "Elicit",
    num_draws: int = 100,
    num_samples: int | None = None,
    seed: int = 0,
    quantile: float = 0.95,
    replication: int | None = None,
) -> float:
    """
    Find the smallest penalty that keeps every shift of a consistent expert at zero

    In `slack`, a shift stays at zero exactly when the gradient of the loss
    with respect to it is shorter than the penalty. The function computes
    this gradient at the learned hyperparameters for pseudo-experts that
    simulate from the learned prior, as in `detect`. Their information is
    consistent by construction. The penalty is the **quantile** of the
    longest gradient over the targets. A consistent expert then keeps every
    shift at zero with this probability, at the start of the refit.

    Parameters
    ----------
    eliobj
        fitted eliobj with ``method="parametric_prior"``

    num_draws
        number of pseudo-experts

    num_samples
        number of prior samples of a pseudo-expert. Default is the
        ``num_samples`` of a simulated expert, else 10 000.

    seed
        seed of the first pseudo-expert. Pseudo-expert ``m`` uses
        ``seed + m``.

    quantile
        probability that a consistent expert keeps every shift at zero

    replication
        index of the replication. Default is the replication with the lowest
        loss, see `best_replication`.

    Returns
    -------
    :
        weight of the penalty for `slack`

    Raises
    ------
    AttributeError
        eliobj has not been fitted yet.

    ValueError
        The method is not ``"parametric_prior"``.

    Examples
    --------
    >>> el.consistency.calibrate_penalty(eliobj)  # doctest: +SKIP
    """
    _check_fitted(eliobj)
    if replication is None:
        replication = best_replication(eliobj)
    if num_samples is None:
        num_samples = eliobj.expert.get("num_samples", EXPERT_NUM_SAMPLES)

    training = _training_statistics(eliobj, replication)
    ranges = _expert_ranges(_expert_statistics(eliobj))
    longest = []
    for pseudo_expert in _pseudo_experts(
        eliobj, replication, num_draws, num_samples, seed
    ):
        shifts = {name: tf.zeros_like(value[0]) for name, value in training.items()}
        with tf.GradientTape() as tape:
            tape.watch(list(shifts.values()))
            shifted = {
                name: value + ranges[name] * shifts[name]
                for name, value in training.items()
            }
            loss, *_ = losses.total_loss(
                elicit_training=shifted,
                elicit_expert=pseudo_expert,
                targets=eliobj.targets,
            )
        gradients = tape.gradient(loss, list(shifts.values()))
        longest.append(max(float(tf.norm(gradient)) for gradient in gradients))
    return float(np.quantile(longest, quantile))


def slack(  # noqa: PLR0913
    eliobj: "Elicit",
    penalty: Sequence[float] | None = None,
    epochs: int = 300,
    learning_rate: float = 0.01,
    step_size: float = 0.05,
    replication: int | None = None,
) -> xr.Dataset:
    """
    Refit with a free shift per target, and show which targets need one

    Each target gets a shift, one value per elicited statistic. The shift is
    added to the simulated statistics before the loss. A shift lets a target
    fit whatever the other targets demand, so the refit always has a prior
    that fits.

    A penalty keeps the shifts at zero unless a shift lowers the loss by more
    than it costs. The penalty is the sum, over the targets, of the length of
    the shift vector, measured in units of the range of the expert
    statistics of that target. A target with consistent information keeps a
    shift of exactly zero. A target that contradicts the others gets a shift,
    in the units of the data.

    The refit starts at the learned hyperparameters. It uses Adam for the
    hyperparameters and a proximal gradient step for the shifts. A shift
    stays at zero exactly when the gradient of the loss with respect to it,
    in units of the expert range, is shorter than the penalty. Each value
    of **penalty** gives one refit. A conflict that appears for every value
    is a stable result.

    Two targets that contradict each other cannot both be right. The penalty
    then shifts the target whose shift is cheaper, which is not necessarily
    the target that is wrong.

    Parameters
    ----------
    eliobj
        fitted eliobj with ``method="parametric_prior"``

    penalty
        weights of the penalty, one refit each. A larger weight gives fewer
        shifts. Default is the single weight of `calibrate_penalty`.

    epochs
        number of steps of each refit

    learning_rate
        learning rate of Adam for the hyperparameters

    step_size
        largest move of a shift in one step, in units of the range of the
        expert statistics

    replication
        index of the replication to start from. Default is the replication
        with the lowest loss, see `best_replication`.

    Returns
    -------
    :
        xr.Dataset with the variables ``size`` (dimensions ``penalty`` and
        ``target``), the root mean square of the shift in units of the range
        of the expert statistics; ``loss`` (dimension ``penalty``), the loss
        of the refit with the shifts and without the penalty; ``skipped``
        (dimension ``penalty``), the number of steps skipped because the loss
        or a gradient was not finite; and one
        variable per elicited statistic with the shift in the units of the
        data.

    Raises
    ------
    AttributeError
        eliobj has not been fitted yet.

    ValueError
        The method is not ``"parametric_prior"``.

    Examples
    --------
    >>> el.consistency.slack(eliobj, penalty=[0.01, 0.1])  # doctest: +SKIP
    """
    _check_fitted(eliobj)
    if replication is None:
        replication = best_replication(eliobj)
    if penalty is None:
        penalty = [calibrate_penalty(eliobj, replication=replication)]
    expert = _expert_statistics(eliobj)
    ranges = _expert_ranges(expert)

    sizes = np.zeros((len(penalty), len(expert)))
    refit_loss = np.zeros(len(penalty))
    skipped = np.zeros(len(penalty), dtype=int)
    shifts_data: dict[str, list[np.ndarray]] = {name: [] for name in expert}
    for p, weight in enumerate(penalty):
        prior_model, seed = _learned_prior(eliobj, replication)
        variables = prior_model.trainable_variables
        # shift in units of the range of the expert statistics
        shifts = {
            name: tf.Variable(tf.zeros_like(value[0])) for name, value in expert.items()
        }
        shift_list = list(shifts.values())
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # compiled like the training step: a `tf.random.set_seed` in the loop
        # would clear the kernel caches, see `models.simulate_and_elicit`
        @tf.function(reduce_retracing=True)  # type: ignore [misc]
        def gradient_step() -> Any:
            with tf.GradientTape() as tape:
                training, *_ = models.simulate_and_elicit(
                    prior_model, eliobj.model, eliobj.targets, seed
                )
                shifted = {
                    name: value + ranges[name] * shifts[name]
                    for name, value in training.items()
                }
                loss, *_ = losses.total_loss(
                    elicit_training=shifted,
                    elicit_expert=expert,
                    targets=eliobj.targets,
                )
            gradients = tape.gradient(loss, [*variables, *shift_list])
            step_ok = tf.math.is_finite(tf.squeeze(loss))
            for gradient in gradients:
                if gradient is not None:
                    step_ok = tf.logical_and(
                        step_ok, tf.reduce_all(tf.math.is_finite(gradient))
                    )
            return loss, gradients, step_ok

        @tf.function(reduce_retracing=True)  # type: ignore [misc]
        def apply_update(gradients: Any, weight: Any) -> None:
            optimizer.apply_gradients(zip(gradients[: len(variables)], variables))
            for shift, gradient in zip(shift_list, gradients[len(variables) :]):
                # the step length is clipped, so that a step moves a shift by
                # at most `step_size` expert ranges. A proximal step has the
                # same fixed points for every step length, so the clipping
                # does not change the result.
                eta = step_size / tf.maximum(1.0, tf.norm(gradient))
                shift.assign_sub(eta * gradient)
                # group soft-thresholding: the proximal step of the penalty
                length = tf.norm(shift)
                shift.assign(
                    shift * tf.maximum(0.0, 1.0 - eta * weight / (length + 1e-12))
                )

        tf.random.set_seed(seed)
        last_loss = float("nan")
        for _ in range(epochs):
            loss, gradients, step_ok = gradient_step()
            # a step that is not finite is skipped, as in the training
            if bool(step_ok):
                last_loss = float(tf.squeeze(loss))
                apply_update(gradients, tf.constant(weight, dtype=tf.float32))
            else:
                skipped[p] += 1

        refit_loss[p] = last_loss
        for k, (name, shift) in enumerate(shifts.items()):
            sizes[p, k] = float(tf.sqrt(tf.reduce_mean(tf.square(shift))))
            shifts_data[name].append(shift.numpy() * ranges[name])

    names = [target["name"] for target in eliobj.targets]
    data_vars: dict[str, Any] = dict(
        size=(("penalty", "target"), sizes),
        loss=("penalty", refit_loss),
        skipped=("penalty", skipped),
    )
    for name, values in shifts_data.items():
        data_vars[name] = (("penalty", f"{name}_dim"), np.stack(values))
    return xr.Dataset(
        data_vars=data_vars,
        coords=dict(penalty=list(penalty), target=names),
        attrs=dict(replication=replication, epochs=epochs),
    )
