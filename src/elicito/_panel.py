"""
Expand the setup of one expert into the setup of a panel of experts

The panel is an ordinary setup: its parameters, targets and model carry the
index of the expert as a suffix, for example ``beta0__e1``. The training does
not know about experts.
"""

import inspect
from collections.abc import Callable
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito.parameters._base import seed_pair
from elicito.specs import hyper
from elicito.types import Hyper, Parameter, Target

OFFSET = "__offset"


def suffix(name: str, expert: int) -> str:
    """Return the name of a quantity of one expert in the panel"""
    return f"{name}__e{expert}"


def _shifted(family: Callable[..., Any], over: dict[str, Hyper]) -> Callable[..., Any]:
    """Return the family of one expert: the panel value plus an offset"""

    def family_of_expert(**hyperparams: Any) -> Any:
        for key, hyp in over.items():
            offset = hyperparams.pop(key + OFFSET)
            # the offset lives on the unconstrained scale, so a bound holds
            panel = hyp["unconstrain"](hyperparams[key])
            hyperparams[key] = hyp["constraint"](panel + offset)
        return family(**hyperparams)

    return family_of_expert


def parameters(
    params: list[Parameter], n_experts: int, over: tuple[str, ...]
) -> list[Parameter]:
    """
    Copy the parameters for each expert, in the order of the experts

    A hyperparameter in ``over`` gets one offset for each expert. All other
    hyperparameters are shared by the experts.
    """
    panel = []
    for e in range(n_experts):
        for par in params:
            hyperparams: dict[str, Any] = {}
            shifted: dict[str, Hyper] = {}
            for key, hyp in par["hyperparams"].items():
                hyperparams[key] = {**hyp, "shared": True}
                if hyp["name"] in over:
                    shifted[key] = hyp
                    hyperparams[key + OFFSET] = hyper(suffix(hyp["name"], e))
            panel.append(
                Parameter(
                    **{
                        **par,
                        "name": suffix(par["name"], e),
                        "family": _shifted(par["family"], shifted),
                        "hyperparams": hyperparams,
                    }
                )
            )
    return panel


def _renamed(func: Callable[..., Any], expert: int) -> Callable[..., Any]:
    """Return a function that takes the arguments of one expert by name"""
    names = list(inspect.signature(func).parameters)

    def func_of_expert(**kwargs: Any) -> Any:
        return func(**{name: kwargs[suffix(name, expert)] for name in names})

    func_of_expert.__signature__ = inspect.Signature(  # type: ignore [attr-defined]
        [
            inspect.Parameter(suffix(name, expert), inspect.Parameter.KEYWORD_ONLY)
            for name in names
        ]
    )
    return func_of_expert


def targets(tars: list[Target], n_experts: int) -> list[Target]:
    """Copy the targets for each expert"""
    panel = []
    for e in range(n_experts):
        for tar in tars:
            output = tar["output"]
            query = tar["query"]
            if query["name"] == "custom":
                query = {**query, "value": _renamed(query["value"], e)}
            panel.append(
                Target(
                    **{
                        **tar,
                        "name": suffix(tar["name"], e),
                        "query": query,
                        "output": _renamed(output, e)
                        if callable(output)
                        else suffix(output, e),
                    }
                )
            )
    return panel


class Model:
    """
    Run the model of one expert for each expert in the panel

    With ``pooled=False``, the prior samples hold the parameters of every
    expert, in the order of the experts. With ``pooled=True``, all experts
    share one prior, so the model runs once.
    """

    def __init__(
        self,
        model: Callable[..., Any],
        n_experts: int,
        pooled: bool,
        correlations: list[str],
    ) -> None:
        self.model = model
        self.n_experts = n_experts
        self.pooled = pooled
        # outputs of the correlation targets: the prior samples of an expert
        self.correlations = correlations
        self._takes_seed = "seed" in inspect.signature(model).parameters

    def __call__(self, prior_samples: Any, seed: Any = None) -> dict[str, Any]:
        """Return the outputs of every expert, keyed by suffixed name"""
        n_runs = 1 if self.pooled else self.n_experts
        samples = tf.split(prior_samples, n_runs, axis=-1)
        seeds = (
            tfp.random.split_seed(seed_pair(0) if seed is None else seed, n=n_runs)
            if self._takes_seed
            else [None] * n_runs
        )
        runs = []
        for run in range(n_runs):
            kwargs = {"seed": seeds[run]} if self._takes_seed else {}
            outputs = dict(self.model(samples[run], **kwargs))
            outputs.update({name: samples[run] for name in self.correlations})
            runs.append(outputs)

        panel = {}
        for e in range(self.n_experts):
            outputs = runs[0 if self.pooled else e]
            panel.update({suffix(name, e): value for name, value in outputs.items()})
        return panel


def answers(experts: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten the answers of every expert into one dict, keyed by suffixed name"""
    return {
        suffix(name, e): value
        for e, answer in enumerate(experts)
        for name, value in answer.items()
    }
