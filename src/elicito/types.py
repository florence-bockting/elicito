"""
specification of custom types
"""

from typing import Callable, Optional, TypedDict, Union

import tensorflow_probability as tfp


class Hyper(TypedDict):
    """
    Typed dictionary for specification of `hyper` (set correct link, TODO)
    """

    name: str
    constraint: Callable
    constraint_name: str
    vtype: Callable
    dim: int
    shared: bool


class Parameter(TypedDict, total=False):
    """
    Typed dictionary for specification of `parameter` (set correct link, TODO)
    """

    name: str
    family: tfp.distributions.Distribution
    hyperparams: Optional[dict[str, Hyper]]
    constraint_name: str
    constraint: Callable


class QueriesDict(TypedDict, total=False):
    """
    Typed dictionary for specification of `queries` (set correct link, TODO)
    """

    name: str
    value: Optional[Union[Callable, tuple]]
    func_name: str


class Target(TypedDict):
    """
    typed dictionary for specification of `target` (set correct link, TODO)
    """

    name: str
    query: QueriesDict
    target_method: Optional[Callable]
    loss: Callable
    weight: float


class ExpertDict(TypedDict, total=False):
    """
    typed dictionary of specification of `expert` (set correct link, TODO)
    """

    ground_truth: dict
    num_samples: int
    data: dict[str, list]


class Uniform(TypedDict):
    """
    typed dictionary for specification of initialization distribution `uniform`

    (set correct link, TODO)
    """

    radius: Union[float, list]
    mean: Union[float, list]
    hyper: Optional[list]


class Initializer(TypedDict):
    """
    typed dictionary for specification of initialization method (set correct link, TODO)
    """

    method: Optional[str]
    distribution: Optional[Uniform]
    loss_quantile: Optional[float]
    iterations: Optional[int]
    hyperparams: Optional[dict]


class Trainer(TypedDict, total=False):
    """
    typed dictionary for specification of `trainer` (set correct link; TODO)
    """

    method: str
    seed: int
    B: int
    num_samples: int
    epochs: int
    seed_chain: int


class NFDict(TypedDict):
    """
    typed dictionary for specification of normalizing flow in `network`

    (set correct link, TODO)
    """

    inference_network: Callable
    network_specs: dict
    base_distribution: Callable


class SaveHist(TypedDict):
    """
    typed dictionary for specification of saving `history` results
    """

    loss: bool
    time: bool
    loss_component: bool
    hyperparameter: bool
    hyperparameter_gradient: bool


class SaveResults(TypedDict):
    """
    typed dictionary for specification of saving `results`
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
    typed dictionary for specification of parallelization `parallel`
    """

    runs: int
    cores: int
    seeds: Optional[list]
