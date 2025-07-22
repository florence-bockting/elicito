"""
specification of custom types
"""

from typing import Any, Callable, Optional, TypedDict, Union

import tensorflow as tf


class Hyper(TypedDict):
    """
    Typed dictionary for specification of [`hyper`][elicito.elicit.hyper]
    """

    name: str
    constraint: Callable[[float], tf.Tensor]
    constraint_name: str
    vtype: Callable[[Any], Any]
    dim: int
    shared: bool


class Parameter(dict):
    """Class for specification of a parameter, inheriting from `dict`."""

    def __init__(self, name, family, hyperparams, constraint_name, constraint):
        super().__init__(
            name=name,
            family=family,
            hyperparams=hyperparams,
            constraint_name=constraint_name,
            constraint=constraint,
        )
        self.name: str = name
        self.family: Any = family
        self.hyperparams: Optional[dict[str, Hyper]] = hyperparams
        self.constraint_name: str = constraint_name
        self.constraint: Callable[[float], float] = constraint

    def __str__(self):
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
        return (
            f"{self.name} ~ {family_name}({hypers})"
            # f"constraint_name={self.constraint_name!r}, "
            # f"constraint={self.constraint})"
        )

    def __repr__(self):
        """Return a readable summary of the object."""
        return self.__str__()


class QueriesDict(TypedDict, total=False):
    """
    Typed dictionary for specification of [`queries`][elicito.elicit.Queries]
    """

    name: str
    value: Optional[Any]
    func_name: str


class Target(dict):
    """Class for specification of a target, inheriting from `dict`."""

    def __init__(self, name, query, target_method, loss, weight):
        super().__init__(
            name=name,
            query=query,
            target_method=target_method,
            loss=loss,
            weight=weight,
        )
        self.name: str = name
        self.query: QueriesDict = query
        self.target_method: Optional[Callable[[Any], Any]] = target_method
        self.loss: Callable[[Any], float] = loss
        self.weight: float = weight

    def __str__(self):
        """Return a readable summary of the object."""
        return (
            f"Target(name={self.name!r}, query={self.query['name']}, "
            f"loss={self.loss.__class__.__name__}, weight={self.weight})"
        )

    def __repr__(self):
        """Return a readable summary of the object."""
        return self.__str__()


class ExpertDict(TypedDict, total=False):
    """
    typed dictionary of specification of [`expert`][elicito.elicit.Expert]
    """

    ground_truth: dict[str, Any]
    num_samples: int
    data: dict[str, list[Any]]


class Uniform(TypedDict):
    """
    typed dictionary for specification of initialization distribution

    See [`uniform`][elicito.initialization.uniform]

    """

    radius: Union[float, list[Union[float, int]]]
    mean: Union[float, list[Union[float, int]]]
    hyper: Optional[list[str]]


class Initializer(TypedDict):
    """
    typed dictionary for specification of initialization method
    """

    method: Optional[str]
    distribution: Optional[Uniform]
    loss_quantile: Optional[float]
    iterations: Optional[int]
    hyperparams: Optional[dict[str, Any]]


class Trainer(TypedDict, total=False):
    """
    typed dictionary for specification of [`trainer`][elicito.elicit.trainer]
    """

    method: str
    seed: int
    B: int
    num_samples: int
    epochs: int
    seed_chain: int
    progress: int


class NFDict(TypedDict):
    """
    Typed dictionary for specification of normalizing flow

    See [`network`][elicito.networks.NF]

    """

    inference_network: Callable[[Any], Any]
    network_specs: dict[str, Any]
    base_distribution: Callable[[Any], Any]


class SaveHist(TypedDict):
    """
    Typed dictionary for specification of saving `history` results

    See [`save_history`][elicito.utils.save_history]
    """

    loss: bool
    time: bool
    loss_component: bool
    hyperparameter: bool
    hyperparameter_gradient: bool


class SaveResults(TypedDict):
    """
    Typed dictionary for specification of saving `results`

    See [`save_results`][elicito.utils.save_results]
    """

    target_quantities: bool
    elicited_statistics: bool
    prior_samples: bool
    model_samples: bool
    expert_elicited_statistics: bool
    expert_prior_samples: bool
    init_loss_list: bool
    init_prior: bool
    init_matrix: bool
    loss_tensor_expert: bool
    loss_tensor_model: bool


class Parallel(TypedDict):
    """
    Typed dictionary for specification of parallelization `parallel`

    See [`parallel`][elicito.utils.parallel]
    """

    runs: int
    cores: int
    seeds: Optional[list[int]]
