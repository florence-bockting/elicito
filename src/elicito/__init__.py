"""
A Python package for learning prior distributions based on expert knowledge
"""

import importlib.metadata

import tensorflow as tf

from elicito import (
    initializers,
    losses,
    models,
    optimizers,
    parameters,
    plots,
    specs,
    targets,
    types,
    utils,
)
from elicito.elicit import Elicit
from elicito.initializers import initializer
from elicito.parameters import networks
from elicito.specs import (
    correlation,
    custom,
    expert,
    hyper,
    identity,
    parameter,
    partial,
    quantiles,
    queries,
)

tf.get_logger().setLevel("ERROR")

__version__ = importlib.metadata.version("elicito")

__all__ = [
    "Elicit",
    "correlation",
    "custom",
    "expert",
    "hyper",
    "identity",
    "initializer",
    "initializers",
    "losses",
    "models",
    "networks",
    "optimizers",
    "parameter",
    "parameters",
    "partial",
    "plots",
    "quantiles",
    "queries",
    "specs",
    "targets",
    "types",
    "utils",
]
