"""
setting-up the elicitation method with Elicit
"""

import inspect
from typing import Any, Callable, Optional

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito.types import (
    ExpertDict,
    Hyper,
    Initializer,
    Parameter,
    QueriesDict,
    Target,
    Trainer,
    Uniform,
)
from elicito.utils import (
    DoubleBound,
    LowerBound,
    UpperBound,
    identity,
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

    def __init__(self, vtype: str, dim: int):
        """
        Initialize Dtype

        Parameters
        ----------
        vtype
            Type of input x.

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
        if self.vtype == "real":
            dtype_dim = tf.cast(x, dtype=tf.float32)
        if self.vtype == "array":
            dtype_dim = tf.constant(x, dtype=tf.float32, shape=(self.dim,))
        return dtype_dim


def hyper(  # noqa: PLR0913
    name: str,
    lower: float = float("-inf"),
    upper: float = float("inf"),
    vtype: str = "real",
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
        Hyperparameter type.

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

        ``vtype`` value can only be either 'real' or 'array'.

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
        msg = (
            "lower must be either '-inf' or a float.",
            "Other strings are not allowed.",
        )
        raise ValueError(msg)

    # check correct value for upper
    if upper == "inf":  # type: ignore
        upper = float("inf")
    if (type(upper) is str) and (upper != "inf"):  # type: ignore
        msg = (
            "upper must be either 'inf' or a float.",
            "Other strings are not allowed.",
        )
        raise ValueError(msg)

    if lower > upper:
        msg = "The value for 'lower' must be smaller than the value for 'upper'."  # type: ignore
        raise ValueError(msg)

    # check values for vtype are implemented
    if vtype not in ["real", "array"]:
        msg = f"vtype must be either 'real' or 'array'. You provided '{vtype=}'."  # type: ignore
        raise ValueError(msg)

    # check that dimensionality is adapted when "array" is chosen
    if (vtype == "array") and dim == 1:
        msg = "For vtype='array', the 'dim' argument must have a value greater 1."  # type: ignore
        raise ValueError(msg)

    # constraints
    # only lower bound
    if (lower != float("-inf")) and (upper == float("inf")):
        lower_bound = LowerBound(lower)
        transform = lower_bound.inverse
        constraint_name = "softplusL"
    # only upper bound
    elif (upper != float("inf")) and (lower == float("-inf")):
        upper_bound = UpperBound(upper)
        transform = upper_bound.inverse
        constraint_name = "softplusU"
    # upper and lower bound
    elif (upper != float("inf")) and (lower != float("-inf")):
        double_bound = DoubleBound(lower, upper)
        transform = double_bound.inverse
        constraint_name = "invlogit"
    # unbounded
    else:
        transform = identity  # type: ignore
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
        Must be an [`tfp.distributions`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions).

    hyperparams
        Hyperparameters of distribution as specified in **family**.
        Only required for ``parametric_prior`` method.
        Structure of dictionary: *keys* must match arguments of
        [`tfp.distributions`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions)
        object and *values* have to be specified using the [`hyper`][elicito.elicit.hyper]
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
        ``family`` has to be a tfp.distributions object.

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
    # check that family is a tfp.distributions object
    if family is not None:
        if family.__module__.split(".")[-1] not in dir(tfd):
            raise ValueError(
                "[section: parameters] The argument 'family'"
                + "has to be a tfp.distributions object."
            )

    # check whether keys of hyperparams dict correspond to arguments of family
    if hyperparams is not None:
        for key in hyperparams:
            if key not in inspect.getfullargspec(family)[0]:
                raise ValueError(
                    f"[section: parameters] '{family.__module__.split('.')[-1]}'"
                    + f" family has no argument '{key}'. Check keys of "
                    + "'hyperparams' dict."
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
        transform = double_bound.inverse
        constraint_name = "invlogit"
    # unbounded
    else:
        transform = identity  # type: ignore
        constraint_name = "identity"

    param_dict: Parameter = dict(
        name=name,
        family=family,
        hyperparams=hyperparams,
        constraint_name=constraint_name,
        constraint=transform,  # type: ignore
    )

    return param_dict


def model(obj: Callable[[str], tf.Tensor], **kwargs: dict[Any, Any]) -> dict[str, Any]:
    """
    Specify the generative model.

    Parameters
    ----------
    obj
        Generative model class as defined by the user.

    **kwargs
        additional keyword arguments expected by `obj`.

    Returns
    -------
    generator_dict :
        Dictionary including all generative model settings.

    Raises
    ------
    ValueError
        generative model in `obj` requires the input argument
        'prior_samples', but argument has not been found.

        optional argument(s) of the generative model specified in `obj` are
        not specified

    Examples
    --------
    >>> # specify the generative model class
    >>> class ToyModel:  # doctest: +SKIP
    >>>     def __call__(self, prior_samples, design_matrix):  # doctest: +SKIP
    >>> # linear predictor
    >>>         epred = tf.matmul(prior_samples, design_matrix,  # doctest: +SKIP
    >>>                           transpose_b=True)  # doctest: +SKIP
    >>> # data-generating model
    >>>         likelihood = tfd.Normal(  # doctest: +SKIP
    >>>             loc=epred,  # doctest: +SKIP
    >>>             scale=tf.expand_dims(prior_samples[:, :, -1], -1)  # doctest: +SKIP
    >>>             )  # doctest: +SKIP
    >>> # prior predictive distribution
    >>>         ypred = likelihood.sample()  # doctest: +SKIP
    >>>
    >>>         return dict(  # doctest: +SKIP
    >>>             likelihood=likelihood,  # doctest: +SKIP
    >>>             ypred=ypred, epred=epred,  # doctest: +SKIP
    >>>             prior_samples=prior_samples  # doctest: +SKIP
    >>>             )  # doctest: +SKIP

    >>> # specify the model category in the elicit object
    >>> el.model(obj=ToyModel,  # doctest: +SKIP
    >>>          design_matrix=design_matrix  # doctest: +SKIP
    >>>          )  # doctest: +SKIP
    """
    # get input arguments of generative model class
    input_args = inspect.getfullargspec(obj.__call__)[0]  # type: ignore
    # check correct input form of generative model class
    if "prior_samples" not in input_args:
        msg = (
            "[section: model] The generative model class 'obj' requires the",
            " input variable 'prior_samples' but argument has not been found",
            " in 'obj'.",
        )
        raise ValueError(msg)

    # check that all optional arguments have been provided by the user
    optional_args = set(input_args).difference({"prior_samples", "self"})
    for arg in optional_args:
        if arg not in list(kwargs.keys()):
            msg = (
                f"[section: model] The argument '{arg=}' required by the",
                "generative model class 'obj' is missing.",
            )  # type: ignore
            raise ValueError(msg)

    generator_dict = dict(obj=obj)

    for key in kwargs:  # noqa: PLC0206
        generator_dict[key] = kwargs[key]  # type: ignore

    return generator_dict


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
                    "[section: targets] Quantiles have to be expressed as",
                    "probability (between 0 and 1).",
                    f" Found quantile={quantile=}",
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


def target(
    name: str,
    loss: Callable[[Any], Any],
    query: QueriesDict,
    target_method: Optional[Callable[[Any], Any]] = None,
    weight: float = 1.0,
) -> Target:
    """
    Specify target quantity and corresponding elicitation technique.

    Parameters
    ----------
    name
        Name of the target quantity. Two approaches are possible:
        (1) Target quantity is identical to an output from the generative
        model: The name must match the output variable name. (2) Custom target
        quantity is computed using the `target_method` argument.

    query
        Specify the elicitation technique by using one of the methods
        implemented in [`Queries`][elicito.elicit.Queries].

    loss
        Lossfunction for computing the discrepancy between expert data and
        model simulations. See [`losses`][elicito.losses].

    target_method
        Custom method for computing a target quantity.
        Note: This method hasn't been implemented yet and will raise an
        ``NotImplementedError``. See
        [GitHub issue #34](https://github.com/florence-bockting/prior_elicitation/issues/34).

    weight
        Weight of the corresponding elicited quantity in the total loss.

    Returns
    -------
    target_dict :
        Dictionary including all settings regarding the target quantity and
        corresponding elicitation technique.

    Examples
    --------
    >>> el.target(name="y_X0",  # doctest: +SKIP
    >>>           query=el.queries.quantiles(  # doctest: +SKIP
    >>>                 (.05, .25, .50, .75, .95)),  # doctest: +SKIP
    >>>           loss=el.losses.MMD2(kernel="energy"),  # doctest: +SKIP
    >>>           weight=1.0  # doctest: +SKIP
    >>>           )  # doctest: +SKIP

    >>> el.target(name="correlation",  # doctest: +SKIP
    >>>           query=el.queries.correlation(),  # doctest: +SKIP
    >>>           loss=el.losses.L2,  # doctest: +SKIP
    >>>           weight=1.0  # doctest: +SKIP
    >>>           )  # doctest: +SKIP
    """
    # create instance of loss class
    loss_instance = loss

    target_dict: Target = dict(
        name=name,
        query=query,
        target_method=target_method,
        loss=loss_instance,
        weight=weight,
    )
    return target_dict


class Expert:
    """
    specify the expert data
    """

    def data(self, dat: dict[str, list[tf.Tensor]]) -> ExpertDict:
        """
        Provide elicited-expert data for learning prior distributions.

        Parameters
        ----------
        dat
            Elicited data from expert provided as dictionary. Data must be
            provided in a standardized format.
            Use [`get_expert_datformat`][elicito.utils.get_expert_datformat]
            to get correct data format for your method specification.

        Returns
        -------
        expert_data :
            Expert-elicited information used for learning prior distributions.

        Examples
        --------
        >>> expert_dat = {  # doctest: +SKIP
        >>>     "quantiles_y_X0": [-12.55, -0.57, 3.29, 7.14, 19.15],  # doctest: +SKIP
        >>>     "quantiles_y_X1": [-11.18, 1.45, 5.06, 8.83, 20.42],  # doctest: +SKIP
        >>>     "quantiles_y_X2": [-9.28, 3.09, 6.83, 10.55, 23.29]  # doctest: +SKIP
        >>> }  # doctest: +SKIP
        """
        # Note: check for correct expert data format is done in Elicit class
        dat_prep: dict[Any, Any] = {
            f"{key}": tf.expand_dims(
                tf.cast(tf.convert_to_tensor(dat[key]), dtype=tf.float32), 0
            )
            for key in dat
        }

        data_dict: ExpertDict = dict(data=dat_prep)
        return data_dict

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
            [`Elicit`][elicito.Elicit] argument `parameters`.

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


def optimizer(
    optimizer: Any = tf.keras.optimizers.Adam(), **kwargs: dict[Any, Any]
) -> dict[str, Any]:
    """
    Specify optimizer and its settings for SGD.

    Parameters
    ----------
    optimizer
        Optimizer used for SGD implemented.
        Must be an object implemented in [`tf.keras.optimizers`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)

    **kwargs
        Additional keyword arguments expected by **optimizer**.

    Returns
    -------
    optimizer_dict :
        Dictionary specifying the SGD optimizer and its additional settings.

    Raises
    ------
    TypeError
        ``optimizer`` is not a tf.keras.optimizers object
    ValueError
        ``optimizer`` could not be found in tf.keras.optimizers

    Examples
    --------
    >>> optimizer = el.optimizer(  # doctest: +SKIP
    >>>     optimizer=tf.keras.optimizers.Adam,  # doctest: +SKIP
    >>>     learning_rate=0.1,  # doctest: +SKIP
    >>>     clipnorm=1.0  # doctest: +SKIP
    >>> )  # doctest: +SKIP
    """
    # check whether optimizer is a tf.keras.optimizers object
    opt_module = ".".join(optimizer.__module__.split(".")[:-1])
    if opt_module != "keras.src.optimizers":
        msg = (
            "[section: optimizer] The 'optimizer' must be a",
            " tf.keras.optimizers object.",
        )
        raise TypeError(msg)

    # check whether the optimizer object can be found in tf.keras.optimizers
    opt_name = str(optimizer).split(".")[-1][:-2]
    if opt_name not in dir(tf.keras.optimizers):
        msg = (  # type: ignore
            "[section: optimizer] The argument 'optimizer' has to be a",
            " tf.keras.optimizers object.",
            f" Couldn't find {opt_name=} in list of tf.keras.optimizers.",
        )
        raise ValueError(msg)

    optimizer_dict = dict(optimizer=optimizer)

    for key in kwargs:  # noqa: PLC0206
        optimizer_dict[key] = kwargs[key]

    return optimizer_dict


def initializer(
    method: Optional[str] = None,
    distribution: Optional[Uniform] = None,
    loss_quantile: Optional[Any] = None,
    iterations: Optional[int] = None,
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
        Name of initialization method.
        Currently supported are "random", "lhs", and "sobol".

    distribution
        Specification of initialization distribution.
        Currently implemented methods: [`uniform`][elicito.initialization.uniform]

    loss_quantile
        Quantile indicating which loss value should be used for selecting the
        initial hyperparameters.Specified as probability value between 0-1.

    iterations
        Number of samples drawn from the initialization distribution.

    hyperparams
        Dictionary with specific initial values per hyperparameter.
        **Note:** Initial values are considered to be on the *unconstrained
        scale*. Use  the ``forward`` method of [`LowerBound`][elicito.utils.LowerBound],
        [`UpperBound`][elicito.utils.UpperBound] and
        [`DoubleBound`][elicito.utils.DoubleBound]
        for transforming a constrained hyperparameter into an
        unconstrained one. In hyperparams dictionary, *keys* refer to
        hyperparameter names, as specified in [`hyper`][elicito.elicit.hyper]
        and *values* to the respective initial values.

    Returns
    -------
    init_dict :
        Dictionary specifying the initialization method.

    Raises
    ------
    ValueError
        ``method`` can only take the values "random", "sobol", or "lhs"

        ``loss_quantile`` must be a probability ranging between 0 and 1.

        Either ``method`` or ``hyperparams`` has to be specified.

    Examples
    --------
    >>> el.initializer(  # doctest: +SKIP
    >>>     method="lhs",  # doctest: +SKIP
    >>>     loss_quantile=0,  # doctest: +SKIP
    >>>     iterations=32,  # doctest: +SKIP
    >>>     distribution=el.initialization.uniform(  # doctest: +SKIP
    >>>         radius=1,  # doctest: +SKIP
    >>>         mean=0   # doctest: +SKIP
    >>>         )  # doctest: +SKIP
    >>>     )  # doctest: +SKIP

    >>> el.initializer(  # doctest: +SKIP
    >>>     hyperparams = dict(  # doctest: +SKIP
    >>>         mu0=0.,  # doctest: +SKIP
    >>>         sigma0=el.utils.LowerBound(lower=0.).forward(0.3),  # doctest: +SKIP
    >>>         mu1=1.,  # doctest: +SKIP
    >>>         sigma1=el.utils.LowerBound(lower=0.).forward(0.5),  # doctest: +SKIP
    >>>         sigma2=el.utils.LowerBound(lower=0.).forward(0.4)  # doctest: +SKIP
    >>>         )  # doctest: +SKIP
    >>>     )  # doctest: +SKIP
    """
    # check that method is implemented

    if method is None:
        for arg in [distribution, loss_quantile, iterations]:
            if arg is not None:
                msg = f"If method is None {arg=} must also be None."
                raise ValueError(msg)

        if hyperparams is None:
            msg = (  # type: ignore
                "[section: initializer] Either 'method' or 'hyperparams' has",
                "to be specified. Use method for sampling from an",
                "initialization distribution and 'hyperparams' for",
                "specifying exact initial values per hyperparameter.",
            )
            raise ValueError(msg)

        quantile_perc = loss_quantile

    else:
        for arg in [distribution, loss_quantile, iterations]:
            if arg is None:
                msg = f"If method is None {arg} must be specified."
                raise ValueError(msg)

        # compute percentage from probability
        if loss_quantile is not None:
            quantile_perc = int(loss_quantile * 100)
        # ensure that iterations is an integer
        if iterations is not None:
            iterations = int(iterations)

        if method not in ["random", "lhs", "sobol"]:
            msg = (  # type: ignore
                "[section: initializer] Currently implemented initialization",
                f"methods are 'random', 'sobol', and 'lhs', but got '{method=}'",
                "as input.",
            )
            raise ValueError(msg)

        # check that quantile is provided as probability
        if loss_quantile is not None:
            if (loss_quantile < 0.0) or (loss_quantile > 1.0):
                msg = (  # type: ignore
                    "[section: initializer] 'loss_quantile' must be a",
                    "value between 0",
                    f"and 1. Found 'loss_quantile={loss_quantile=}'.",
                )
                raise ValueError(msg)

    init_dict: Initializer = dict(
        method=method,
        distribution=distribution,
        loss_quantile=quantile_perc,
        iterations=iterations,
        hyperparams=hyperparams,
    )

    return init_dict


def trainer(  # noqa: PLR0913
    method: str,
    seed: int,
    epochs: int,
    B: int = 128,
    num_samples: int = 200,
    progress: int = 1,
) -> Trainer:
    """
    Specify training settings for learning the prior distribution(s).

    Parameters
    ----------
    method
        Method for learning the prior distribution. Available is either
        ``parametric_prior`` for learning independent parametric priors
        or ``deep_prior`` for learning a joint non-parameteric prior.

    seed
        Seed used for learning.

    epochs
        Number of iterations until training is stopped.

    B
        Batch size.

    num_samples
        Number of samples from the prior(s).

    progress
        whether training progress should be printed. Progress is shown if
        `progress=1` and muted if `progress=0`.

    Returns
    -------
    train_dict :
        dictionary specifying the training settings for learning the prior
        distribution(s).

    Raises
    ------
    ValueError
        ``method`` can only take the value "parametric_prior" or "deep_prior"

        ``epochs`` can only take positive integers. Minimum number of epochs
        is 1.

        `progress` can only be 0 (mute progress) or 1 (print progress)

    Examples
    --------
    >>> el.trainer(  # doctest: +SKIP
    >>>     method="parametric_prior",  # doctest: +SKIP
    >>>     seed=0,  # doctest: +SKIP
    >>>     epochs=400,  # doctest: +SKIP
    >>>     B=128,  # doctest: +SKIP
    >>>     num_samples=200  # doctest: +SKIP
    >>> )  # doctest: +SKIP
    """
    # check that progress is either 0 or 1
    if progress not in [0, 1]:
        raise ValueError("Progress has to be either 0 or 1. " + "Got {progress=}.")
    # check that epochs are positive numbers
    if epochs <= 0:
        msg = (
            "[section: trainer] The number of epochs has to be at least 1.",
            f" Got {epochs=} epochs",
        )
        raise ValueError(msg)

    # check that method is implemented
    if method not in ["parametric_prior", "deep_prior"]:
        msg = (
            "[section: trainer] Currently only the methods 'deep_prior' and",
            f"'parametric prior' are implemented but got '{method=}'.",
        )
        raise ValueError(msg)

    train_dict: Trainer = dict(
        method=method,
        seed=int(seed),
        B=int(B),
        num_samples=int(num_samples),
        epochs=int(epochs),
        progress=progress,
    )
    return train_dict
