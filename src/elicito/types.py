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


class Parameter(TypedDict, total=False):
    """
    Typed dictionary for specification of [`parameter`][elicito.elicit.parameter]
    """

    name: str
    family: Any
    hyperparams: Optional[dict[str, Hyper]]
    constraint_name: str
    constraint: Callable[[float], float]


class QueriesDict(TypedDict, total=False):
    """
    Typed dictionary for specification of [`queries`][elicito.elicit.Queries]
    """

    name: str
    value: Optional[Any]
    func_name: str


class Target(TypedDict):
    """
    typed dictionary for specification of [`target`][elicito.elicit.target]
    """

    name: str
    query: QueriesDict
    target_method: Optional[Callable[[Any], Any]]
    loss: Callable[[Any], float]
    weight: float


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
