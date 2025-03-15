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
from elicito.elicit import *

__version__ = importlib.metadata.version("elicito")

__all__ = [
    "Elicit",
    "initialization",
    "losses",
    "networks",
    "optimization",
    "parameter",
    "plots",
    "simulations",
    "target",
    "targets",
    "types",
    "utils",
]
