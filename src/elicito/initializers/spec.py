"""
The user specification of the initialization method
"""

from typing import Any, Optional

from elicito.initializers.methods import get_init_method, resolve_init_method
from elicito.initializers.sampling import uniform
from elicito.types import Initializer, SamplingMethod, Uniform


def initializer(
    method: Optional[SamplingMethod] = SamplingMethod.warmstart,
    distribution: Optional[Uniform] = None,
    iterations: Optional[int] = None,
    warmup_epochs: int = 0,
    hyperparams: Optional[dict[str, Any]] = None,
) -> Initializer:
    """
    Initialize hyperparameter values

    Only necessary for method ``parametric_prior``.
    Two approaches are currently possible:

    1. Specify specific initial values for each hyperparameter.
    2. Use one of the implemented sampling approaches to draw initial
       values from one of the provided initialization distributions

    In (2) initial values for each hyperparameter are drawn from a uniform
    distribution ranging from ``mean - radius`` to ``mean + radius``.

    Parameters
    ----------
    method
        Name of initialization method. The default is "warmstart".
        Currently supported are "random", "lhs", "sobol" and "warmstart".
        The first three draw candidates from **distribution**. "warmstart"
        instead runs a Nelder-Mead search from the centre of **distribution**,
        before the first gradient step.
        "warmstart" rescues a badly placed box. It does not beat sampling from
        a well-placed one: the loss at the start does not predict the loss
        after training.

    distribution
        Specification of initialization distribution.
        The default is [`uniform`][elicito.initializers.sampling.uniform] with its
        defaults, ``mean=0`` and ``radius=1``, on the unconstrained scale.
        Set a ``mean`` and a ``radius`` that match the scale of the
        hyperparameters. A box that is wrong by a factor of ten gives a bad
        start value, and for some prior families a non-finite loss.

    iterations
        Number of samples drawn from the initialization distribution.
        For method "warmstart" it is the number of objective evaluations of
        the search, not a number of candidates. The default comes from the
        method: 100 for "warmstart", 32 for the three samplers.

    warmup_epochs
        Number of training epochs to run for each candidate before it is
        scored. With ``0`` a candidate is scored by its loss at epoch 0, which
        does not show whether its trajectory is stable. A value of about 10
        rejects a candidate that diverges early. Cost is
        ``iterations * warmup_epochs`` extra epochs; ``iterations=32,
        warmup_epochs=10`` costs 320 epochs against a 500-epoch run.

    hyperparams
        Dictionary with specific initial values per hyperparameter.
        **Note:** Initial values are considered to be on the *unconstrained
        scale*. Use the ``forward`` method of
        [`LowerBound`][elicito.parameters.bijections.LowerBound],
        [`UpperBound`][elicito.parameters.bijections.UpperBound] and
        [`DoubleBound`][elicito.parameters.bijections.DoubleBound]
        for transforming a constrained hyperparameter into an
        unconstrained one. In hyperparams dictionary, *keys* refer to
        hyperparameter names, as specified in [`hyper`][elicito.specs.hyper]
        and *values* to the respective initial values.

    Returns
    -------
    init_dict :
        Dictionary specifying the initialization method.

    Raises
    ------
    ValueError
        ``method`` can only take the values "random", "sobol", or "lhs"

        Either ``method`` or ``hyperparams`` has to be specified.

    Examples
    --------
    >>> el.initializer(  # doctest: +SKIP
    >>>     method="lhs",  # doctest: +SKIP
    >>>     iterations=32,  # doctest: +SKIP
    >>>     distribution=el.initializers.uniform(  # doctest: +SKIP
    >>>         radius=1,  # doctest: +SKIP
    >>>         mean=0   # doctest: +SKIP
    >>>         )  # doctest: +SKIP
    >>>     )  # doctest: +SKIP

    >>> el.initializer(  # doctest: +SKIP
    >>>     hyperparams = dict(  # doctest: +SKIP
    >>>         mu0=0.,  # doctest: +SKIP
    >>>         sigma0=el.parameters.LowerBound(lower=0).forward(0.3),  # doctest: +SKIP
    >>>         mu1=1.,  # doctest: +SKIP
    >>>         sigma1=el.parameters.LowerBound(lower=0).forward(0.5),  # doctest: +SKIP
    >>>         sigma2=el.parameters.LowerBound(lower=0).forward(0.4)  # doctest: +SKIP
    >>>         )  # doctest: +SKIP
    >>>     )  # doctest: +SKIP
    """
    # check that method is implemented

    if method is None:
        args = {"distribution": distribution, "iterations": iterations}

        for name, value in args.items():
            if value is not None:
                raise ValueError(f"If method is None, '{name}' must also be None.")  # noqa: TRY003

        if hyperparams is None:
            msg = (
                "Either 'method' or 'hyperparams' has"
                "to be specified. Use method for sampling from an"
                "initialization distribution and 'hyperparams' for"
                "specifying exact initial values per hyperparameter."
            )
            raise ValueError(msg)

    else:
        if distribution is None:
            distribution = uniform()
        if iterations is None:
            iterations = get_init_method(method).default_iterations

        # ensure that iterations is an integer
        if iterations is not None:
            iterations = int(iterations)

    init_dict: Initializer = dict(
        method=method,
        distribution=distribution,
        iterations=iterations,
        warmup_epochs=int(warmup_epochs),
        hyperparams=hyperparams,
    )

    # each initialization method rejects the input it cannot use
    resolve_init_method(init_dict).check(init_dict)

    return init_dict
