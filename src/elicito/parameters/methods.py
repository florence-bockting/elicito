"""
Protocol of the prior-learning methods, and their registry
"""

from typing import Any, Protocol

import tensorflow as tf

from elicito.parameters.deep import DeepPrior
from elicito.parameters.parametric import ParametricPrior
from elicito.types import (
    Initializer,
    NFDict,
    Parameter,
)


class PriorMethod(Protocol):
    """Behaviour that differs between the prior-learning methods."""

    name: str

    def build(
        self,
        parameters: list[Parameter],
        network: NFDict | None,
        init_matrix_slice: dict[str, tf.Tensor] | None,
        seed: int,
    ) -> Any:
        """Create the trainable prior object."""
        ...

    def sample(  # noqa: PLR0913
        self,
        initialized_priors: Any,
        parameters: list[Parameter],
        network: NFDict | None,
        B: int,
        num_samples: int,
        seed: int,
    ) -> Any:
        """Draw prior samples of shape (B, num_samples, num_params)."""
        ...

    def trainable_variables(self, prior_model: Any) -> Any:
        """Return the variables the optimizer updates."""
        ...

    def new_history(
        self, prior_model: Any, parameters: list[Parameter]
    ) -> dict[str, Any]:
        """Create the per-epoch record, seeded with the initial values."""
        ...

    def record_epoch(
        self,
        history: dict[str, Any],
        prior_sim: Any,
        trainable_vars: Any,
        parameters: list[Parameter],
    ) -> None:
        """Append this epoch's values to the record."""
        ...

    def finalize(
        self,
        res_ep: dict[str, Any],
        output_res: dict[str, Any],
        gradients_ep: Any,
        trainable_vars: Any,
    ) -> None:
        """Add the method-specific entries to the results."""
        ...

    def check(
        self,
        parameters: list[Parameter],
        network: NFDict | None,
        initializer: Initializer | None,
    ) -> None:
        """Raise if the sections are not valid for this method."""
        ...


_METHODS: dict[str, type[PriorMethod]] = {
    ParametricPrior.name: ParametricPrior,
    DeepPrior.name: DeepPrior,
}


def get_method(name: str) -> PriorMethod:
    """Return a new strategy object for a ``trainer["method"]`` string."""
    try:
        method_cls = _METHODS[name]
    except KeyError:
        msg = f"Unknown method {name!r}. Valid: {sorted(_METHODS)}."
        raise ValueError(msg) from None
    return method_cls()
