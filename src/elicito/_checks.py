"""
Check user input of Elicit object
"""

import inspect
from collections.abc import Callable
from typing import Any

from elicito.optimizers import cmaes
from elicito.parameters import methods
from elicito.types import Target


def check_elicit(  # type: ignore  # noqa: PLR0913
    model,
    parameters,
    targets,
    trainer,
    optimizer,
    network,
    initializer,
) -> None:
    """
    Check whether input values to Elicit() are valid
    """
    check_model(model)
    check_target_names(targets)

    # CMA-ES keeps a full covariance matrix of the search space. The weights
    # of a normalizing flow are thousands of dimensions, which it cannot search.
    if optimizer["optimizer"] == cmaes.CMAES and trainer["method"] == "deep_prior":
        msg = (
            f"optimizer='{cmaes.CMAES}' cannot learn a deep prior. A network "
            "has too many trainable variables for a derivative-free search. "
            "Use optimizer='auto' or a tf.keras optimizer instead."
        )
        raise ValueError(msg)

    methods.get_method(trainer["method"]).check(parameters, network, initializer)


def check_model(model: Callable[..., Any]) -> None:
    """
    Check that the generative model is callable with the prior samples
    """
    if isinstance(model, type) or not callable(model):
        msg = (
            "The model must be a PyMC model or a callable, such as an "
            f"instance of a class with __call__. Got {model!r}."
        )
        raise TypeError(msg)
    params = inspect.signature(model).parameters
    if "prior_samples" not in params:
        msg = "The model must take the argument 'prior_samples'."
        raise ValueError(msg)
    # elicito passes a stateless seed to a model that names `seed`
    missing = [
        name
        for name, p in params.items()
        if name not in {"prior_samples", "seed"} and p.default is p.empty
    ]
    if missing:
        msg = (
            f"The model arguments {missing} have no value. "
            "Bind them with functools.partial."
        )
        raise ValueError(msg)


def check_expert(expert: Any, targets: list[Target], parameters: Any) -> None:
    """
    Check the answers of an expert, or the ground truth of an oracle
    """
    # check expert data
    try:
        expert["ground_truth"]
    except KeyError:
        check_answers(expert["data"], targets)

    else:
        # oracle: ensure ground truth has same dim as number of model param
        expected_params = [param["name"] for param in parameters]
        num_params = 0
        if expert["ground_truth"] is None:
            pass
        else:
            for k in expert["ground_truth"]:
                # sample once; .sample() is not free and the shape is all we need
                ground_truth_sample = expert["ground_truth"][k].sample(1)
                # type list can result in cases where a tfd.Sequential/
                # Jointdistribution is used
                if type(ground_truth_sample) is list:
                    num_params += sum(
                        [param.shape[-1] for param in ground_truth_sample]
                    )
                else:
                    num_params += ground_truth_sample.shape[-1]

        if len(expected_params) != num_params:
            msg = (
                "Dimensionality of ground truth in "
                "'expert' is not the same  as number of model "
                f"parameters. Got {num_params=}, expected "
                f"{len(expected_params)}."
            )
            raise AssertionError(msg)


def check_answers(answers: dict[str, Any], targets: list[Target]) -> None:
    """
    Check the expert answers against the targets

    Parameters
    ----------
    answers
        Expert answers, keyed by target name.

    targets
        List of target quantities.

    Raises
    ------
    ValueError
        An answer is missing, has no target, or has the wrong length.
    """
    names = [tar["name"] for tar in targets]
    missing = [name for name in names if name not in answers]
    unknown = [key for key in answers if key not in names]
    if missing or unknown:
        msg = (
            "The expert answers do not match the targets. "
            f"Missing answers: {missing}. Answers without a target: {unknown}."
        )
        raise ValueError(msg)

    for tar in targets:
        if tar["query"]["name"] != "quantiles":
            continue
        n_probs = len(tar["query"]["value"])
        n_values = answers[tar["name"]].shape[-1]
        if n_values != n_probs:
            msg = (
                f"The answer for target {tar['name']!r} has {n_values} values, "
                f"but the target asks for {n_probs} quantiles."
            )
            raise ValueError(msg)


# names of variables in the results that a target name must not take
RESERVED_NAMES = ("prior", "total_loss", "penalty")


def check_target_names(targets: list[Target]) -> None:
    """
    Check that the target names are unique and not reserved

    Parameters
    ----------
    targets
        List of target quantities.

    Raises
    ------
    ValueError
        Two targets have the same name, or a name is reserved.
    """
    names = [tar["name"] for tar in targets]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        msg = f"Target names must be unique. Duplicates: {duplicates}."
        raise ValueError(msg)
    reserved = [name for name in names if name in RESERVED_NAMES]
    if reserved:
        msg = f"Target names {reserved} are reserved. Use another name."
        raise ValueError(msg)
