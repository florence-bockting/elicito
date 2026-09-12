"""
specification of custom types
"""

from collections.abc import Callable
from enum import Enum
from typing import Any, TypedDict

import tensorflow as tf


class Hyper(TypedDict):
    """
    Typed dictionary for specification of [`hyper`][elicito.specs.hyper]
    """

    name: str
    constraint: Callable[[float], tf.Tensor]
    constraint_name: str
    vtype: Callable[[Any], Any]
    dim: int
    shared: bool


class Parameter(dict[str, Any]):
    """Class for specification of a parameter, inheriting from `dict`."""

    def __init__(
        self,
        name: str,
        family: Any,
        hyperparams: dict[str, Hyper] | None,
        constraint_name: str,
        constraint: Callable[[float], float],
    ):
        super().__init__(
            name=name,
            family=family,
            hyperparams=hyperparams,
            constraint_name=constraint_name,
            constraint=constraint,
        )
        self.name: str = name
        self.family: Any = family
        self.hyperparams: dict[str, Hyper] | None = hyperparams
        self.constraint_name: str = constraint_name
        self.constraint: Callable[[float], float] = constraint

    def __str__(self) -> str:
        """Return a readable summary of the object."""
        if self.family is None:
            family_name = "Unknown"
        else:
            family_name = self.family.__name__

        if self.hyperparams is None:
            hypers = ""
        else:
            hypers = ", ".join(
                [f"{k}: {v['name']}" for k, v in self.hyperparams.items()]
            )
        return f"{self.name} ~ {family_name}({hypers})"

    def __repr__(self) -> str:
        """Return a readable summary of the object."""
        return self.__str__()


class QueriesDict(TypedDict, total=False):
    """
    Typed dictionary for specification of [`queries`][elicito.specs.Queries]
    """

    name: str
    value: Any | None
    func_name: str


class Target(dict[str, Any]):
    """Class for specification of a target, inheriting from `dict`."""

    def __init__(
        self,
        name: str,
        query: QueriesDict,
        target_method: Callable[[Any], Any] | None,
        loss: Callable[[Any], float],
        weight: float,
    ):
        super().__init__(
            name=name,
            query=query,
            target_method=target_method,
            loss=loss,
            weight=weight,
        )
        self.name: str = name
        self.query: QueriesDict = query
        self.target_method: Callable[[Any], Any] | None = target_method
        self.loss: Callable[[Any], float] = loss
        self.weight: float = weight

    def __str__(self) -> str:
        """Return a readable summary of the object."""
        return (
            f"Target(name={self.name!r}, query={self.query['name']}, "
            f"loss={self.loss.__class__.__name__}, weight={self.weight})"
        )

    def __repr__(self) -> str:
        """Return a readable summary of the object."""
        return self.__str__()


class ExpertDict(TypedDict, total=False):
    """
    typed dictionary of specification of [`expert`][elicito.specs.Expert]
    """

    ground_truth: dict[str, Any]
    num_samples: int
    data: dict[str, list[Any]]


class Uniform(TypedDict):
    """
    typed dictionary for specification of initialization distribution

    See [`uniform`][elicito.initializers.sampling.uniform]

    """

    radius: float | list[float | int]
    mean: float | list[float | int]
    hyper: list[str] | None


class Initializer(TypedDict):
    """
    typed dictionary for specification of initialization method
    """

    method: str | None
    distribution: Uniform | None
    iterations: int | None
    warmup_epochs: int
    hyperparams: dict[str, Any] | None


class Trainer(TypedDict, total=False):
    """
    typed dictionary for specification of [`trainer`][elicito.specs.trainer]
    """

    method: str
    seed: int
    B: int
    num_samples: int
    epochs: int
    seed_chain: int
    progress: int
    kappa: float


class NFDict(TypedDict):
    """
    Typed dictionary for specification of normalizing flow

    See [`network`][elicito.networks.NF]

    """

    inference_network: Callable[[Any], Any]
    network_specs: dict[str, Any]
    base_distribution: Callable[[Any], Any]


class Parallel(TypedDict):
    """
    Typed dictionary for specification of parallelization `parallel`

    See [`parallel`][elicito.utils.parallel]
    """

    runs: int
    cores: int
    seeds: list[int] | None


class MetaSettings(TypedDict):
    """
    Typed dictionary for specification of meta settings in `Elicit`

    See [`Elicit`][elicito.elicit.Elicit]
    """

    dry_run: bool


class PriorMethods(str, Enum):
    """Method used for learning prior distribution"""

    parametric_prior = "parametric_prior"
    deep_prior = "deep_prior"


class ProgressMethod(int, Enum):
    """Printing status of optimization"""

    HIDE_PROGRESS = 0
    SHOW_PROGRESS = 1


class SamplingMethod(str, Enum):
    """Sampling method used for initialization"""

    sobol = "sobol"
    random = "random"
    lhs = "lhs"
    warmstart = "warmstart"
    cmaes = "cmaes"


class VariableType(str, Enum):
    """Type of variable"""

    real = "real"
    array = "array"
    cov = "cov"
    cov2tril = "cov2tril"
