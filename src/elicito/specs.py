"""
Constructors for the building blocks of an elicitation method
"""

import inspect
from typing import Any, Callable, Optional

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito import losses
from elicito.parameters.bijections import (
    DoubleBound,
    LowerBound,
    UpperBound,
)
from elicito.parameters.bijections import identity as identity_bijection
from elicito.types import (
    ExpertDict,
    Hyper,
    Parameter,
    Partial,
    QueriesDict,
    Target,
    VariableType,
)

tfd = tfp.distributions


class Dtype:
    """
    Create a tensorflow scalar or array depending on the vtype attribute.

    Attributes
    ----------
    vtype
        Type of input x.

    dim
        Dimensionality of input x.
        Scalar: `dim = 1`; Vector: `dim > 1`

    Returns
    -------
    :
        Tensor of shape depending on `vtype` and `dim`.
    """

    def __init__(self, vtype: VariableType, dim: int):
        """
        Initialize Dtype

        Parameters
        ----------
        vtype
            Type of input x.
            either real, array, cov, or cov2tril

        dim
            Dimensionality of input
        """
        self.vtype = vtype
        self.dim = dim

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply data type to input x

        Parameters
        ----------
        x
            input x

        Returns
        -------
        :
            input x with correct type
        """
        if self.vtype == VariableType.real:
            dtype_dim = tf.cast(x, dtype=tf.float32)
        if self.vtype == VariableType.array:
            dtype_dim = tf.constant(x, dtype=tf.float32, shape=(self.dim,))
        if self.vtype == VariableType.cov:
            dtype_dim = tf.constant(x, dtype=tf.float32, shape=(self.dim, self.dim))
        if self.vtype == VariableType.cov2tril:
            dtype_dim = tf.linalg.cholesky(
                tf.constant(x, dtype=tf.float32, shape=(self.dim, self.dim))
            )
        return dtype_dim


def hyper(  # noqa: PLR0913
    name: str,
    lower: float = float("-inf"),
    upper: float = float("inf"),
    vtype: VariableType = VariableType.real,
    dim: int = 1,
    shared: bool = False,
) -> Hyper:
    """
    Specify prior hyperparameters.

    Parameters
    ----------
    name
        Custom name of hyperparameter.

    lower
        Lower bound of hyperparameter.

    upper
        Upper bound of hyperparameter.

    vtype
        Hyperparameter type. Either "real", "array",
        "cov" or "cov2tril" (lower triangular of
        covariance matrix realised via cholesky(cov))

    dim
        Dimensionality of variable.
        Only required if `vtype = "array"`.

    shared
        Shared hyperparameter between model parameters.

    Returns
    -------
    hyppar_dict :
        Dictionary including all hyperparameter settings.

    Raises
    ------
    ValueError
        ``lower``, ``upper`` take only values that are float
        or `"-inf"`or `"inf"`.

        ``lower`` value should not be higher than ``upper`` value.

        ``vtype`` value can only be either 'real', 'array', 'cov', or 'cov2tril'

        ``dim`` value can't be '1' if 'vtype="array"'

    Examples
    --------
    >>> # sigma hyperparameter of a parametric distribution
    >>> el.hyper(name="sigma0", lower=0)  # doctest: +SKIP

    >>> # shared hyperparameter
    >>> el.hyper(name="sigma", lower=0, shared=True)  # doctest: +SKIP

    """
    # check correct value for lower
    if lower == "-inf":  # type: ignore
        lower = float("-inf")

    if (type(lower) is str) and (lower != "-inf"):  # type: ignore
        msg = "Lower must be either '-inf' or a float." "Other strings are not allowed."
        raise ValueError(msg)

    # check correct value for upper
    if upper == "inf":  # type: ignore
        upper = float("inf")
    if (type(upper) is str) and (upper != "inf"):  # type: ignore
        msg = "Upper must be either 'inf' or a float." "Other strings are not allowed."
        raise ValueError(msg)

    if lower > upper:
        msg = "The value for 'lower' must be smaller than the value for 'upper'."
        raise ValueError(msg)

    # check values for vtype are implemented
    if vtype not in ["real", "array", "cov", "cov2tril"]:
        msg = (
            "vtype must be either 'real', 'array', 'cov', 'cov2tril'. "
            f"You provided {vtype=}."
        )
        raise ValueError(msg)

    # check that dimensionality is adapted when "array" is chosen
    if (vtype == "array") and dim == 1:
        msg = "For vtype='array', the 'dim' argument must have a value greater 1."
        raise ValueError(msg)

    # constraints
    # only lower bound
    if (lower != float("-inf")) and (upper == float("inf")):
        lower_bound = LowerBound(lower=lower)
        transform = lower_bound.inverse
        constraint_name = "softplusL"
    # only upper bound
    elif (upper != float("inf")) and (lower == float("-inf")):
        upper_bound = UpperBound(upper=upper)
        transform = upper_bound.inverse
        constraint_name = "softplusU"
    # upper and lower bound
    elif (upper != float("inf")) and (lower != float("-inf")):
        double_bound = DoubleBound(lower=lower, upper=upper)
        transform = double_bound.inverse  # type: ignore
        constraint_name = "invlogit"
    # unbounded
    else:
        transform = identity_bijection  # type: ignore
        constraint_name = "identity"

    # value type
    dtype_dim = Dtype(vtype, dim)

    hyper_dict: Hyper = dict(
        name=name,
        constraint=transform,
        constraint_name=constraint_name,
        vtype=dtype_dim,
        dim=dim,
        shared=shared,
    )

    return hyper_dict


def parameter(
    name: str,
    family: Optional[Any] = None,
    hyperparams: Optional[dict[str, Hyper]] = None,
    lower: float = float("-inf"),
    upper: float = float("inf"),
) -> Parameter:
    """
    Specify model parameters.

    Parameters
    ----------
    name
        Custom name of parameter.

    family
        Prior distribution family for model parameter.
        Only required for ``parametric_prior`` method.
        Must be a member of [`tfp.distributions`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions).

    hyperparams
        Hyperparameters of distribution as specified in **family**.
        Only required for ``parametric_prior`` method.
        Structure of dictionary: *keys* must match arguments of
        [`tfp.distributions`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions)
        object and *values* have to be specified using the [`hyper`][elicito.specs.hyper]
        method.

    lower
        Only used if ``method = "deep_prior"``.
        Lower bound of parameter.

    upper
        Only used if ``method = "deep_prior"``.
        Upper bound of parameter.

    Returns
    -------
    param_dict : dict
        Dictionary including all model (hyper)parameter settings.

    Raises
    ------
    ValueError
        ``hyperparams`` value is a dict with keys corresponding to arguments of
        tfp.distributions object in 'family'. Raises error if key does not
        correspond to any argument of distribution.

    Examples
    --------
    >>> el.parameter(name="beta0",  # doctest: +SKIP
    >>>              family=tfd.Normal,  # doctest: +SKIP
    >>>              hyperparams=dict(loc=el.hyper("mu0"),  # doctest: +SKIP
    >>>                               scale=el.hyper("sigma0", lower=0)  # doctest: +SKIP
    >>>                               )  # doctest: +SKIP
    >>>              )  # doctest: +SKIP

    """  # noqa: E501
    # check whether keys of hyperparams dict correspond to arguments of family
    if hyperparams is not None:
        for key in hyperparams:
            if key not in inspect.getfullargspec(family)[0]:
                raise ValueError(  # noqa: TRY003
                    f"'{family.__module__.split('.')[-1]}'"
                    f" family has no argument '{key}'. Check keys of "
                    "'hyperparams' dict."
                )

    # constraints
    # only lower bound
    if (lower != float("-inf")) and (upper == float("inf")):
        lower_bound = LowerBound(lower)
        transform = lower_bound.inverse
        constraint_name: str = "softplusL"
    # only upper bound
    elif (upper != float("inf")) and (lower == float("-inf")):
        upper_bound = UpperBound(upper)
        transform = upper_bound.inverse
        constraint_name = "softplusU"
    # upper and lower bound
    elif (upper != float("inf")) and (lower != float("-inf")):
        double_bound = DoubleBound(lower, upper)
        transform = double_bound.inverse  # type: ignore
        constraint_name = "invlogit"
    # unbounded
    else:
        transform = identity_bijection  # type: ignore
        constraint_name = "identity"

    return Parameter(
        name=name,
        family=family,
        hyperparams=hyperparams,
        constraint_name=constraint_name,
        constraint=transform,  # type: ignore
    )


class Queries:
    """
    specify elicitation techniques
    """

    def quantiles(self, quantiles: tuple[float, ...]) -> QueriesDict:
        """
        Implement a quantile-based elicitation technique.

        Parameters
        ----------
        quantiles
            Tuple with respective quantiles ranging between 0 and 1.

        Returns
        -------
        elicit_dict :
            Dictionary including the quantile settings.

        Raises
        ------
        ValueError
            ``quantiles`` have to be specified as probability ranging between
            0 and 1.

        """
        # compute percentage from probability
        quantiles_perc = tuple([q * 100 for q in quantiles])

        # check that quantiles are provided as percentage
        for quantile in quantiles:
            if (quantile < 0) or (quantile > 1):
                msg = (
                    "Quantiles have to be expressed as "
                    f"probability (between 0 and 1). Got {quantile=}."
                )
                raise ValueError(msg)

        elicit_dict: QueriesDict = dict(name="quantiles", value=quantiles_perc)
        return elicit_dict

    def identity(self) -> QueriesDict:
        """
        Implement an identity function.

        Should be used if no further transformation of target quantity is required.

        Returns
        -------
        elicit_dict :
            Dictionary including the identity settings.

        """
        elicit_dict: QueriesDict = dict(name="identity", value=None)
        return elicit_dict

    def correlation(self) -> QueriesDict:
        """
        Calculate the pearson correlation between model parameters.

        Returns
        -------
        elicit_dict : dict
            Dictionary including the correlation settings.

        """
        elicit_dict: QueriesDict = dict(name="pearson_correlation", value=None)
        return elicit_dict

    def custom(self, func: Callable[[Any], Any]) -> QueriesDict:
        """
        Specify a custom target method.

        The custom method can be passed as argument.

        Parameters
        ----------
        func
            Custom target method.

        Returns
        -------
        elicit_dict :
            Dictionary including the custom settings.

        """
        elicit_dict: QueriesDict = dict(
            name="custom", func_name=func.__name__, value=func
        )
        return elicit_dict


# create an instance of the Queries class
queries = Queries()


def quantiles(  # noqa: PLR0913
    name: str,
    output: str | Callable[..., Any] | None = None,
    select: int | dict[str, Any] | None = None,
    probs: tuple[float, ...] = (0.05, 0.5, 0.95),
    weight: float = 1.0,
    loss: Callable[..., Any] | None = None,
) -> Target:
    """
    Ask the expert for quantiles of a model output.

    Parameters
    ----------
    name
        Name of the target. The expert answer uses this name as key.

    output
        Model output to summarize. A string names a model output. A callable
        receives the model outputs by argument name. The default is ``name``.

    select
        Part of the output to keep. An integer indexes the last axis. A
        dict maps the dims of a PyMC output to coord labels, such as
        ``{"participant": 3, "delay": 12}``. The default keeps all.

    probs
        Probabilities of the quantiles, each between 0 and 1.

    weight
        Weight of the target in the total loss.

    loss
        Loss between the expert answer and the model statistic. The default
        is ``MMD2(kernel="energy")``.

    Returns
    -------
    target :
        Specification of the target.

    Examples
    --------
    >>> el.quantiles("y_X0", probs=(0.25, 0.5, 0.75))  # doctest: +SKIP
    """
    return Target(
        name=name,
        query=queries.quantiles(probs),
        output=name if output is None else output,
        loss=losses.MMD2(kernel="energy") if loss is None else loss,
        weight=weight,
        select=select,
    )


def identity(
    name: str,
    output: str | Callable[..., Any] | None = None,
    select: int | dict[str, Any] | None = None,
    weight: float = 1.0,
    loss: Callable[..., Any] | None = None,
) -> Target:
    """
    Ask the expert for draws of a model output.

    Parameters
    ----------
    name
        Name of the target. The expert answer uses this name as key.

    output
        Model output to compare. A string names a model output. A callable
        receives the model outputs by argument name. The default is ``name``.

    select
        Part of the output to keep. An integer indexes the last axis. A
        dict maps the dims of a PyMC output to coord labels, such as
        ``{"participant": 3, "delay": 12}``. The default keeps all.

    weight
        Weight of the target in the total loss.

    loss
        Loss between the expert answer and the model output. The default
        is ``MMD2(kernel="energy")``.

    Returns
    -------
    target :
        Specification of the target.
    """
    return Target(
        name=name,
        query=queries.identity(),
        output=name if output is None else output,
        loss=losses.MMD2(kernel="energy") if loss is None else loss,
        weight=weight,
        select=select,
    )


def correlation(
    name: str,
    weight: float = 1.0,
    loss: Callable[..., Any] | None = None,
) -> Target:
    """
    Ask the expert for the correlations between the model parameters.

    Parameters
    ----------
    name
        Name of the target. The expert answer uses this name as key.

    weight
        Weight of the target in the total loss.

    loss
        Loss between the expert answer and the model correlations. The
        default is ``L2``.

    Returns
    -------
    target :
        Specification of the target.
    """
    return Target(
        name=name,
        query=queries.correlation(),
        output=name,
        loss=losses.L2 if loss is None else loss,
        weight=weight,
    )


def custom(
    name: str,
    func: Callable[..., Any],
    loss: Callable[..., Any],
    weight: float = 1.0,
    select: int | dict[str, Any] | None = None,
) -> Target:
    """
    Ask the expert for a statistic that you define.

    Parameters
    ----------
    name
        Name of the target. The expert answer uses this name as key.

    func
        Function that computes the statistic. It receives the model outputs
        by argument name.

    loss
        Loss between the expert answer and the statistic.

    weight
        Weight of the target in the total loss.

    select
        Part of the output to keep. An integer indexes the last axis. A
        dict maps the dims of a PyMC output to coord labels, such as
        ``{"participant": 3, "delay": 12}``. The default keeps all.

    Returns
    -------
    target :
        Specification of the target.
    """
    return Target(
        name=name,
        query=queries.custom(func),
        output=name,
        loss=loss,
        weight=weight,
        select=select,
    )


def partial(
    over: list[str] | tuple[str, ...],
    tau: float | dict[str, float] | None = None,
) -> Partial:
    """
    Pool a panel of experts partially.

    Each expert gets an offset from the panel value of the named
    hyperparameters. The other hyperparameters are shared by all experts.

    Parameters
    ----------
    over
        Names of the hyperparameters that differ between the experts.

    tau
        Spread of the offsets: the disagreement in the panel. A small value
        pulls the experts toward the panel value. A float applies to all
        names in ``over``. A dict gives one value for each name. ``None``
        estimates the spread from the answers.

    Returns
    -------
    pool :
        Specification of the pooling. Give it to ``Elicit(pool=...)``.

    Examples
    --------
    >>> el.partial(over=["mu0", "mu1"])  # doctest: +SKIP

    >>> el.partial(over=["mu0", "mu1"], tau={"mu0": 1.5, "mu1": 0.5})  # doctest: +SKIP
    """
    return Partial(over=tuple(over), tau=tau)


class Expert:
    """
    specify the expert data
    """

    def simulator(
        self, ground_truth: dict[str, Any], num_samples: int = 10_000
    ) -> ExpertDict:
        """
        Simulate data from an oracle

        Define a ground truth (i.e., specify 'true' prior distribution(s)).

        Parameters
        ----------
        ground_truth
            True prior distribution(s). *Keys* refer to parameter names and
            *values* to prior distributions implemented as
            [`tfp.distributions`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions)
            object with predetermined hyperparameter values.
            You can specify a prior distribution for each model parameter or
            a joint prior for all model parameters at once or any approach in
            between. Only requirement is that the dimensionality of all priors
            in ground truth match with the number of model parameters.
            Order of priors in ground truth must match order of
            [`Elicit`][elicito.elicit.Elicit] argument `parameters`.

        num_samples
            Number of draws from the prior distribution.
            It is recommended to use a high value to min. sampling variation.

        Returns
        -------
        expert_data :
            Settings of oracle for simulating from ground truth. True elicited
            statistics are used as `expert-data` in loss function.

        Examples
        --------
        >>> el.expert.simulator(  # doctest: +SKIP
        >>>     ground_truth = {  # doctest: +SKIP
        >>>         "beta0": tfd.Normal(loc=5, scale=1),  # doctest: +SKIP
        >>>         "beta1": tfd.Normal(loc=2, scale=1),  # doctest: +SKIP
        >>>         "sigma": tfd.HalfNormal(scale=10.0),  # doctest: +SKIP
        >>>     },  # doctest: +SKIP
        >>>     num_samples = 10_000  # doctest: +SKIP
        >>> )  # doctest: +SKIP

        >>> el.expert.simulator(  # doctest: +SKIP
        >>>     ground_truth = {  # doctest: +SKIP
        >>>         "betas": tfd.MultivariateNormalDiag(  # doctest: +SKIP
        >>>                 [5.,2.], [1.,1.]),  # doctest: +SKIP
        >>>         "sigma": tfd.HalfNormal(scale=10.0),  # doctest: +SKIP
        >>>     },  # doctest: +SKIP
        >>>     num_samples = 10_000  # doctest: +SKIP
        >>> )  # doctest: +SKIP

        >>> el.expert.simulator(  # doctest: +SKIP
        >>>     ground_truth = {  # doctest: +SKIP
        >>>         "thetas": tfd.MultivariateNormalDiag([5.,2.,1.],  # doctest: +SKIP
        >>>                                              [1.,1.,1.]),  # doctest: +SKIP
        >>>     },  # doctest: +SKIP
        >>>     num_samples = 10_000  # doctest: +SKIP
        >>> )  # doctest: +SKIP
        """
        # Note: check whether dimensionality of ground truth and number of
        # model parameters is identical is done in Elicit class

        expert_data: ExpertDict = dict(
            ground_truth=ground_truth, num_samples=int(num_samples)
        )
        return expert_data


# create an instantiation of Expert class
expert = Expert()
