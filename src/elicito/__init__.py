"""
A Python package for learning prior distributions based on expert knowledge
"""

import importlib.metadata

from elicito import (
    initialization,
    losses,
    networks,
    optimization,
    plots,
    simulations,
    targets,
    types,
    utils,
)
from elicito.elicit import (
    Elicit,
    expert,
    hyper,
    initializer,
    model,
    optimizer,
    parameter,
    queries,
    target,
    trainer,
)

__version__ = importlib.metadata.version("elicito")

__all__ = [
    "Elicit",
    "expert",
    "hyper",
    "initialization",
    "initializer",
    "losses",
    "model",
    "networks",
    "optimization",
    "optimizer",
    "parameter",
    "plots",
    "queries",
    "simulations",
    "target",
    "targets",
    "trainer",
    "types",
    "utils",
]
