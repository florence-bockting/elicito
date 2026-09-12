"""
A Python package for learning prior distributions based on expert knowledge
"""

import importlib.metadata

import tensorflow as tf

from elicito import (
    initializers,
    losses,
    models,
    networks,
    optimizers,
    parameters,
    plots,
    specs,
    targets,
    types,
    utils,
)
from elicito.elicit import Elicit
from elicito.specs import (
    expert,
    hyper,
    initializer,
    meta_settings,
    model,
    optimizer,
    parameter,
    queries,
    target,
    trainer,
)

tf.get_logger().setLevel("ERROR")

__version__ = importlib.metadata.version("elicito")

__all__ = [
    "Elicit",
    "expert",
    "hyper",
    "initializer",
    "initializers",
    "losses",
    "meta_settings",
    "model",
    "models",
    "networks",
    "optimizer",
    "optimizers",
    "parameter",
    "parameters",
    "plots",
    "queries",
    "specs",
    "target",
    "targets",
    "trainer",
    "types",
    "utils",
]
