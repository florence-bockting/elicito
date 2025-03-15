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

__version__ = importlib.metadata.version("elicito")

__all__ = [
    "initialization",
    "losses",
    "networks",
    "optimization",
    "plots",
    "simulations",
    "targets",
    "types",
    "utils",
]
