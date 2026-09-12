"""
Result type and candidate selection shared by the initialization methods
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from elicito.types import (
    Initializer,
)


@dataclass
class InitResult:
    """What an initialization method returns to ``initialize``."""

    prior_model: Any
    candidates: Optional[dict[str, Any]] = None
    losses: Optional[list[Any]] = None


def _select_candidate(losses: list[Any], initializer: Initializer) -> int:
    """Return the index of the candidate with minimum loss"""
    values = np.asarray(losses, dtype=np.float64).reshape(-1)
    finite = np.flatnonzero(np.isfinite(values))

    if finite.size == 0:
        dist = initializer["distribution"]
        detail = (
            f"The initialization distribution is centred at {dist['mean']} "
            f"with radius {dist['radius']}, on the unconstrained scale. "
            "Re-centre it on the expected hyperparameter values, or "
            "reduce its radius."
            if dist is not None
            else "No initialization distribution is set."
        )
        msg = (
            f"All {values.size} initialization candidates yield a "
            f"non-finite loss, so no start value can be selected. {detail}"
        )
        raise ValueError(msg)

    return int(finite[int(np.argmin(values[finite]))])


def _check_box(initializer: Initializer) -> None:
    """Reject a box method that has no box to draw from."""
    for name in ("distribution", "iterations"):
        if initializer[name] is None:
            msg = f"If '{name}' is None, then 'method' must also be None."
            raise ValueError(msg)
