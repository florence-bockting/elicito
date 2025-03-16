"""
setting-up the elicitation method with Elicit
"""

import inspect
from typing import Annotated, Any, Callable, Optional

import joblib
import tensorflow as tf
import tensorflow_probability as tfp

from elicito import initialization, networks, optimization
from elicito.types import (
    ExpertDict,
    Hyper,
    Initializer,
    NFDict,
    Parallel,
    Parameter,
    QueriesDict,
    SaveHist,
    SaveResults,
    Target,
    Trainer,
    Uniform,
)
from elicito.utils import (
    DoubleBound,
    LowerBound,
    UpperBound,
    clean_savings,
    get_expert_data,
    get_expert_datformat,
    identity,
    save,
    save_history,
    save_results,
)

tfd = tfp.distributions


class Dtype:
    """
    Create a tensorflow scalar or array depending on the vtype attribute.

    Attributes
    ----------
    vtype: str, ("real", "array")
        Type of input parameter x.
    dim: int
        Dimensionality of input parameter x. For scalar: dim=1, for vector: dim>1

    Returns
    -------
    tf.Tensor
        Tensor of correct shape depending on vtype and dim.
    """

    def __init__(self, vtype: str, dim: int):
        """
        Initialize Dtype

        Parameters
        ----------
        vtype
            Type of input parameter x.
        dim
            dimensionality of input parameter
        """
        self.vtype = vtype
        self.dim = dim

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply data type to input x

        Parameters
        ----------
        x
            input variable

        Returns
        -------
        casted_x
            input variable with correct type
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
    name : string
        Custom name of hyperparameter.
    lower : float
        Lower bound of hyperparameter.
        The default is unbounded: ``float("-inf")``.
    upper : float
        Upper bound of hyperparameter.
        The default is unbounded: ``float("inf")``.
    vtype : string, ("real", "array")
        Hyperparameter type. The default is ``"real"``.
    dim : integer
        Dimensionality of variable. Only required if vtype is "array".
        The default is ``1``.
    shared : bool
        Shared hyperparameter between model parameters.
        The default is ``False``.

    Returns
    -------
    hyppar_dict : dict
        Dictionary including all hyperparameter settings.

    Raises
    ------
    ValueError
        ``lower``, ``upper`` take only values that are float or "-inf"/"inf".

        ``lower`` value should not be higher than ``upper`` value.

        ``vtype`` value can only be either 'real' or 'array'.

        ``dim`` value can't be '1' if 'vtype="array"'

    Examples
    --------
    >>> # sigma hyperparameter of a parametric distribution
    >>> el.hyper(name="sigma0", lower=0)

    >>> # shared hyperparameter
    >>> el.hyper(name="sigma", lower=0, shared=True)

    """
    # check correct value for lower
    if lower == "-inf":
        lower = float("-inf")

    if (type(lower) is str) and (lower != "-inf"):
        raise ValueError(
            "lower must be either '-inf' or a float."
            + " Other strings are not allowed."
        )

    # check correct value for upper
    if upper == "inf":
        upper = float("inf")
    if (type(upper) is str) and (upper != "inf"):
        msg = "upper must be either 'inf' or a float. Other strings are not allowed."
        raise ValueError(msg)

    if lower > upper:
        msg = "The value for 'lower' must be smaller than the value for 'upper'."
        raise ValueError(msg)

    # check values for vtype are implemented
    if vtype not in ["real", "array"]:
        msg = f"vtype must be either 'real' or 'array'. You provided '{vtype}'."
        raise ValueError(msg)

    # check that dimensionality is adapted when "array" is chosen
    if (vtype == "array") and dim == 1:
        msg = "For vtype='array', the 'dim' argument must have a value greater 1."
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
        transform = identity
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
    family: Optional[tfp.distributions.Distribution] = None,
    hyperparams: Optional[dict[str, Hyper]] = None,
    lower: float = float("-inf"),
    upper: float = float("inf"),
) -> Parameter:
    """
    Specify model parameters.

    Parameters
    ----------
    name : string
        Custom name of parameter.
    family : tfp.distributions.Distribution, optional
        Prior distribution family for model parameter.
        Only required for ``parametric_prior`` method.
        Must be an `tfp.distributions <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions>`_ object.
    hyperparams : elicit.hyper, optional
        Hyperparameters of distribution as specified in **family**.
        Only required for ``parametric_prior`` method.
        Structure of dictionary: *keys* must match arguments of
        `tfp.distributions <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions>`_
        object and *values* have to be specified using the :func:`hyper`
        method.
        Further details are provided in
        `How-To specify prior hyperparameters (TODO) <url>`_.
        Default value is ``None``.
    lower : float
        Only used if ``method="deep_prior"``.
        Lower bound of parameter.
        The default value is ``float("-inf")``.
    upper : float
        Only used if ``method="deep_prior"``.
        Upper bound of parameter.
        The default value is ``float("inf")``.

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
    >>> el.parameter(name="beta0",
    >>>              family=tfd.Normal,
    >>>              hyperparams=dict(loc=el.hyper("mu0"),
    >>>                               scale=el.hyper("sigma0", lower=0)
    >>>                               )
    >>>              )

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
        transform = identity
        constraint_name = "identity"

    param_dict: Parameter = dict(
        name=name,
        family=family,
        hyperparams=hyperparams,
        constraint_name=constraint_name,
        constraint=transform,
    )

    return param_dict


def model(obj: Callable, **kwargs) -> dict[str, Any]:
    """
    Specify the generative model.

    Parameters
    ----------
    obj : class
        class that implements the generative model.
        See `How-To specify the generative_model for details (TODO) <url>`_.
    **kwargs : keyword arguments
        additional keyword arguments expected by **obj**.

    Returns
    -------
    generator_dict : dict
        Dictionary including all generative model settings.

    Raises
    ------
    ValueError
        generative model in ``obj`` requires the input argument
        'prior_samples', but argument has not been found.

        optional argument(s) of the generative model specified in ``obj`` are
        not specified

    Examples
    --------
    >>> # specify the generative model class
    >>> class ToyModel:
    >>>     def __call__(self, prior_samples, design_matrix):
    >>> # linear predictor
    >>>         epred = tf.matmul(prior_samples, design_matrix,
    >>>                           transpose_b=True)
    >>> # data-generating model
    >>>         likelihood = tfd.Normal(
    >>>             loc=epred, scale=tf.expand_dims(prior_samples[:, :, -1], -1)
    >>>             )
    >>> # prior predictive distribution
    >>>         ypred = likelihood.sample()
    >>>
    >>>         return dict(
    >>>             likelihood=likelihood,
    >>>             ypred=ypred, epred=epred,
    >>>             prior_samples=prior_samples
    >>>             )  # doctest: +SKIP

    >>> # specify the model category in the elicit object
    >>> el.model(obj=ToyModel,
    >>>          design_matrix=design_matrix
    >>>          )  # doctest: +SKIP
    """
    # get input arguments of generative model class
    input_args = inspect.getfullargspec(obj.__call__)[0]
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
            )
            raise ValueError(msg)

    generator_dict = dict(obj=obj)

    for key in kwargs:  # noqa: PLC0206
        generator_dict[key] = kwargs[key]

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
        quantiles : tuple
            Tuple with respective quantiles ranging between 0 and 1.

        Returns
        -------
        elicit_dict : dict
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
        elicit_dict : dict
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

    def custom(self, func: Callable) -> QueriesDict:
        """
        Specify a custom target method.

        The custom method can be passed as argument.

        Parameters
        ----------
        func : callable
            Custom target method.

        Returns
        -------
        elicit_dict : dict
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
    loss: Callable,
    query: QueriesDict,
    target_method: Optional[Callable] = None,
    weight: float = 1.0,
) -> Target:
    """
    Specify target quantity and corresponding elicitation technique.

    Parameters
    ----------
    name : string
        Name of the target quantity. Two approaches are possible:
        (1) Target quantity is identical to an output from the generative
        model: The name must match the output variable name. (2) Custom target
        quantity is computed using the **target_method** argument.
    query : dict
        Specify the elicitation technique by using one of the methods
        implemented in :func:`Queries`.
        See `How-To specify custom elicitation techniques (TODO) <url>`_.
    loss : callable
        Loss function for computing the discrepancy between expert data and
        model simulations. Implemented classes can be found
        in :mod:`elicit.losses`.
        The default is the maximum mean discrepancy with
        an energy kernel: :func:`elicit.losses.MMD2`
    target_method : callable, optional
        Custom method for computing a target quantity.
        Note: This method hasn't been implemented yet and will raise an
        ``NotImplementedError``. See for further information the corresponding
        `GitHub issue #34 <https://github.com/florence-bockting/prior_elicitation/issues/34>`_.
        The default is ``None``.
    weight : float
        Weight of the corresponding elicited quantity in the total loss.
        The default is ``1.0``.

    Returns
    -------
    target_dict : dict
        Dictionary including all settings regarding the target quantity and
        corresponding elicitation technique.

    Examples
    --------
    >>> el.target(name="y_X0",
    >>>           query=el.queries.quantiles((.05, .25, .50, .75, .95)),
    >>>           loss=el.losses.MMD2(kernel="energy"),
    >>>           weight=1.0
    >>>           )

    >>> el.target(name="correlation",
    >>>           query=el.queries.correlation(),
    >>>           loss=el.losses.L2,
    >>>           weight=1.0
    >>>           )
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

    def data(self, dat: dict[str, list]) -> ExpertDict:
        """
        Provide elicited-expert data for learning prior distributions.

        Parameters
        ----------
        dat : dict
            Elicited data from expert provided as dictionary. Data must be
            provided in a standardized format.
            Use :func:`elicit.utils.get_expert_datformat` to get correct data
            format for your method specification.

        Returns
        -------
        expert_data : dict
            Expert-elicited information used for learning prior distributions.

        Examples
        --------
        >>> expert_dat = {
        >>>     "quantiles_y_X0": [-12.55, -0.57, 3.29, 7.14, 19.15],
        >>>     "quantiles_y_X1": [-11.18, 1.45, 5.06, 8.83, 20.42],
        >>>     "quantiles_y_X2": [-9.28, 3.09, 6.83, 10.55, 23.29]
        >>> }
        """
        # Note: check for correct expert data format is done in Elicit class
        dat_prep = {
            f"{key}": tf.expand_dims(
                tf.cast(tf.convert_to_tensor(dat[key]), dtype=tf.float32), 0
            )
            for key in dat
        }

        data_dict: ExpertDict = dict(data=dat_prep)
        return data_dict

    def simulator(self, ground_truth: dict, num_samples: int = 10_000) -> ExpertDict:
        """
        Simulate data from an oracle

        Define a ground truth (true prior distribution(s)).
        See `Explanation: Simulating from an oracle (TODO) <url>`_ for
        further details.

        Parameters
        ----------
        ground_truth : dict
            True prior distribution(s). *Keys* refer to parameter names and
            *values* to prior distributions implemented as
            `tfp.distributions <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions>`_
            object with predetermined hyperparameter values.
            You can specify a prior distribution for each model parameter or
            a joint prior for all model parameters at once or any approach in
            between. Only requirement is that the dimensionality of all priors
            in ground truth match with the number of model parameters.
            Order of priors in ground truth must match order of
            :func:`elicit.elicit.Elicit` argument ``parameters``.
        num_samples : int
            Number of draws from the prior distribution.
            It is recommended to use a high value to min. sampling variation.
            The default is ``10_000``.

        Returns
        -------
        expert_data : dict
            Settings of oracle for simulating from ground truth. True elicited
            statistics are used as `expert-data` in loss function.

        Examples
        --------
        >>> el.expert.simulator(
        >>>     ground_truth = {
        >>>         "beta0": tfd.Normal(loc=5, scale=1),
        >>>         "beta1": tfd.Normal(loc=2, scale=1),
        >>>         "sigma": tfd.HalfNormal(scale=10.0),
        >>>     },
        >>>     num_samples = 10_000
        >>> )

        >>> el.expert.simulator(
        >>>     ground_truth = {
        >>>         "betas": tfd.MultivariateNormalDiag([5.,2.], [1.,1.]),
        >>>         "sigma": tfd.HalfNormal(scale=10.0),
        >>>     },
        >>>     num_samples = 10_000
        >>> )

        >>> el.expert.simulator(
        >>>     ground_truth = {
        >>>         "thetas": tfd.MultivariateNormalDiag([5.,2.,1.],
        >>>                                              [1.,1.,1.]),
        >>>     },
        >>>     num_samples = 10_000
        >>> )
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
    optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(), **kwargs
) -> dict[str, Any]:
    """
    Specify optimizer and its settings for SGD.

    Parameters
    ----------
    optimizer : callable, `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ object.
        Optimizer used for SGD implemented.
        Must be an object implemented in `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ object.
        The default is ``tf.keras.optimizers.Adam``.
    **kwargs : keyword arguments
        Additional keyword arguments expected by **optimizer**.

    Returns
    -------
    optimizer_dict : dict
        Dictionary specifying the SGD optimizer and its additional settings.

    Raises
    ------
    TypeError
        ``optimizer`` is not a tf.keras.optimizers object
    ValueError
        ``optimizer`` could not be found in tf.keras.optimizers

    Examples
    --------
    >>> optimizer=el.optimizer(
    >>>     optimizer=tf.keras.optimizers.Adam,
    >>>     learning_rate=0.1,
    >>>     clipnorm=1.0
    >>> )
    """  # noqa: E501
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
        msg = (
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
    loss_quantile: Optional[Annotated[float, "0-1"]] = None,
    iterations: Optional[int] = None,
    hyperparams: Optional[dict] = None,
) -> Initializer:
    """
    Initialize hyperparameter values

    Only necessary for method ``parametric_prior``:
    Two approaches are currently possible:

        1. Specify specific initial values for each hyperparameter.
        2. Use one of the implemented sampling approaches to draw initial
           values from one of the provided initialization distributions

    In (2) initial values for each hyperparameter are drawn from a uniform
    distribution ranging from ``mean-radius`` to ``mean+radius``.
    Further details on the implemented initialization method can be found in
    `Explanation: Initialization method <url>`_.

    Parameters
    ----------
    method : string, optional
        Name of initialization method. Currently supported are "random", "lhs",
        and "sobol".
    distribution : dict, optional
        Specification of initialization distribution.
        Currently implemented methods: :func:`elicit.initialization.uniform`
    loss_quantile : float, optional
        Quantile indicating which loss value should be used for selecting the
        initial hyperparameters.Specified as probability value between 0-1.
    iterations : int, optional
        Number of samples drawn from the initialization distribution.
    hyperparams : dict, optional
        Dictionary with specific initial values per hyperparameter.
        **Note:** Initial values are considered to be on the *unconstrained
        scale*. Use  the ``forward`` method of :func:`elicit.utils.LowerBound`,
        :func:`elicit.utils.UpperBound` and :func:`elicit.utils.DoubleBound`
        for transforming a constrained hyperparameter into an unconstrained one.
        In hyperparams dictionary, *keys* refer to hyperparameter names,
        as specified in :func:`hyper` and *values* to the respective initial
        values.

    Returns
    -------
    init_dict : dict
        Dictionary specifying the initialization method.

    Raises
    ------
    ValueError
        ``method`` can only take the values "random", "sobol", or "lhs"

        ``loss_quantile`` must be a probability ranging between 0 and 1.

        Either ``method`` or ``hyperparams`` has to be specified.

    Examples
    --------
    >>> el.initializer(
    >>>     method="lhs",
    >>>     loss_quantile=0,
    >>>     iterations=32,
    >>>     distribution=el.initialization.uniform(
    >>>         radius=1,
    >>>         mean=0
    >>>         )
    >>>     )

    >>> el.initializer(
    >>>     hyperparams = dict(
    >>>         mu0=0., sigma0=el.utils.LowerBound(lower=0.).forward(0.3),
    >>>         mu1=1., sigma1=el.utils.LowerBound(lower=0.).forward(0.5),
    >>>         sigma2=el.utils.LowerBound(lower=0.).forward(0.4)
    >>>         )
    >>>     )
    """
    # check that method is implemented

    if method is None:
        for arg in [distribution, loss_quantile, iterations]:
            if arg is not None:
                msg = f"If method is None {arg=} must also be None."
                raise ValueError(msg)

        if hyperparams is None:
            msg = (
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
            quantile_perc: Annotated[int, "0-100"] = int(loss_quantile * 100)
        # ensure that iterations is an integer
        if iterations is not None:
            iterations = int(iterations)

        if method not in ["random", "lhs", "sobol"]:
            msg = (
                "[section: initializer] Currently implemented initialization",
                f"methods are 'random', 'sobol', and 'lhs', but got '{method=}'",
                "as input.",
            )
            raise ValueError(msg)

        # check that quantile is provided as probability
        if loss_quantile is not None:
            if (loss_quantile < 0.0) or (loss_quantile > 1.0):
                msg = (
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


def trainer(
    method: str, seed: int, epochs: int, B: int = 128, num_samples: int = 200
) -> Trainer:
    """
    Specify training settings for learning the prior distribution(s).

    Parameters
    ----------
    method : str
        Method for learning the prior distribution. Available is either
        ``parametric_prior`` for learning independent parametric priors
        or ``deep_prior`` for learning a joint non-parameteric prior.
    seed : int
        seed used for learning.
    epochs : int
        number of iterations until training is stopped.
    B : int
        batch size. The default is 128.
    num_samples : int
        number of samples from the prior(s). The default is 200.

    Returns
    -------
    train_dict : dict
        dictionary specifying the training settings for learning the prior
        distribution(s).

    Raises
    ------
    ValueError
        ``method`` can only take the value "parametric_prior" or "deep_prior"

        ``epochs`` can only take positive integers. Minimum number of epochs
        is 1.

    Examples
    --------
    >>> el.trainer(
    >>>     method="parametric_prior",
    >>>     seed=0,
    >>>     epochs=400,
    >>>     B=128,
    >>>     num_samples=200
    >>> )
    """
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
    )
    return train_dict


class Elicit:
    """
    setup the elicitation method
    """

    def __init__(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        model: dict[str, Any],
        parameters: list[Parameter],
        targets: list[Target],
        expert: ExpertDict,
        trainer: Trainer,
        optimizer: dict[str, Any],
        network: Optional[NFDict] = None,
        initializer: Optional[Initializer] = None,
    ):
        """
        Specify the elicitation method

        Parameters
        ----------
        model : dict
            specification of generative model using :func:`model`.
        parameters : list
            list of model parameters specified with :func:`parameter`.
        targets : list
            list of target quantities specified with :func:`target`.
        expert : dict
            provide input data from expert or simulate data from oracle with
            either the ``data`` or ``simulator`` method of the
            :mod:`elicit.elicit.Expert` module.
        trainer : dict
            specification of training settings and meta-information for
            workflow using :func:`trainer`
        optimizer : dict
            specification of SGD optimizer and its settings using
            :func:`optimizer`.
        network : dict, optional
            specification of neural network using a method implemented in
            :mod:`elicit.networks`.
            Only required for ``deep_prior`` method. For ``parametric_prior``
            use ``None``.
        initializer : dict, optional
            specification of initialization settings using
            :func:`initializer`. Only required for ``parametric_prior`` method.
            Otherwise the argument should be ``None``.

        Returns
        -------
        eliobj : class instance
            specification of all settings to run the elicitation workflow and
            fit the eliobj.

        Raises
        ------
        AssertionError
            ``expert`` data are not in the required format. Correct specification of
            keys can be checked using el.utils.get_expert_datformat

            Dimensionality of ``ground_truth`` for simulating expert data, must be
            the same as the number of model parameters.

        ValueError
            if ``method="deep_prior"``, ``network`` can't be None and ``initialization``
            should be None.

            if ``method="deep_prior"``, ``num_params`` as specified in the ``network_specs``
            argument (section: network) does not match the number of parameters
            specified in the parameters section.

            if ``method="parametric_prior"``, ``network`` should be None and
            ``initialization`` can't be None.

            if ``method ="parametric_prior" and multiple hyperparameter have
            the same name but are not shared by setting ``shared=True``."

            if ``hyperparams`` is specified in section ``initializer`` and a
            hyperparameter name (key in hyperparams dict) does not match any
            hyperparameter name specified in :func:`hyper`.

        NotImplementedError
            [network] Currently only the standard normal distribution is
            implemented as base distribution. See `GitHub issue #35 <https://github.com/florence-bockting/prior_elicitation/issues/35>`_.

        """  # noqa: E501
        # check expert data
        expected_dict = get_expert_datformat(targets)
        try:
            expert["ground_truth"]
        except KeyError:
            # input expert data: ensure data has expected format
            if list(expert["data"].keys()) != list(expected_dict.keys()):
                msg = (
                    "[section: expert] Provided expert data is not in the",
                    "correct format. Please use",
                    "el.utils.get_expert_datformat to check expected format.",
                )
                raise AssertionError(msg)

        else:
            # oracle: ensure ground truth has same dim as number of model param
            expected_params = [param["name"] for param in parameters]
            num_params = 0
            for k in expert["ground_truth"]:
                num_params += expert["ground_truth"][k].sample(1).shape[-1]

            if len(expected_params) != num_params:
                msg = (
                    "[section: expert] Dimensionality of ground truth in",
                    "'expert' is not the same  as number of model",
                    f"parameters.Got {num_params=}, expected",
                    f"{len(expected_params)}.",
                )
                raise AssertionError(msg)

        # check that network architecture is provided when method is deep prior
        # and initializer is none
        if trainer["method"] == "deep_prior":
            if network is None:
                msg = (
                    "[section network] If method is 'deep prior',",
                    " the section 'network' can't be None.",
                )
                raise ValueError(msg)

            if initializer is not None:
                msg = (
                    "[section initializer] For method 'deep_prior' the ",
                    "'initializer' is not used and should be set to None.",
                )
                raise ValueError(msg)

            if network["network_specs"]["num_params"] != len(parameters):
                msg = (
                    "[section network] The number of model parameters as ",
                    "specified in the parameters section, must match the",
                    "number of parameters specified in the network (see",
                    "network_specs['num_params'] argument).",
                    f"Expected {len(parameters)} but got",
                    f"{network['network_specs']['num_params']}",
                )
                raise ValueError(msg)

            if network["base_distribution"].__class__ != networks.BaseNormal:
                msg = (
                    "[network] Currently only the standard normal distribution",
                    "is implemented as base distribution.",
                    "See GitHub issue #35.",
                )
                raise NotImplementedError(msg)

        # check that initializer is provided when method=parametric prior
        # and network is none
        if trainer["method"] == "parametric_prior":
            if initializer is None:
                msg = (
                    "[section initializer] If method is 'parametric_prior',",
                    " the section 'initializer' can't be None.",
                )
                raise ValueError(msg)

            if network is not None:
                msg = (
                    "[section network] If method is 'parametric prior'",
                    "the 'network' is not used and should be set to None.",
                )
                raise ValueError(msg)

            # check that hyperparameter names are not redundant
            hyp_names = []
            hyp_shared = []
            for i in range(len(parameters)):
                if parameters[i]["hyperparams"] is None:
                    msg = (
                        "When using method='parametric_prior', the argument",
                        "'hyperparams' of el.parameter",
                        "cannot be None.",
                    )
                    raise ValueError(msg)

                hyp_names.append(
                    [
                        parameters[i]["hyperparams"][key]["name"]
                        for key in parameters[i]["hyperparams"].keys()
                    ]
                )
                hyp_shared.append(
                    [
                        parameters[i]["hyperparams"][key]["shared"]
                        for key in parameters[i]["hyperparams"].keys()
                    ]
                )
            # flatten nested list
            hyp_names_flat = sum(hyp_names, [])  # noqa: RUF017
            hyp_shared_flat = sum(hyp_shared, [])  # noqa: RUF017

            if initializer["method"] is None:
                for k in initializer["hyperparams"]:
                    if k not in hyp_names_flat:
                        msg = (
                            f"[initializer] Hyperparameter name '{k}' doesn't",
                            "match any name specified in the parameters ",
                            "section. Have you misspelled the name?",
                        )
                        raise ValueError(msg)

            seen = []
            duplicate = []
            share = []
            for n, s in zip(hyp_names_flat, hyp_shared_flat):
                if n not in seen:
                    seen.append(n)
                elif s:
                    share.append(n)
                else:
                    duplicate.append(n)

            if len(duplicate) != 0:
                msg = (
                    "[parameters] The following hyperparameter have the same",
                    f"name but are not shared: {duplicate}.",
                    "Have you forgot to set shared=True?",
                )
                raise ValueError(msg)

        self.model = model
        self.parameters = parameters
        self.targets = targets
        self.expert = expert
        self.trainer = trainer
        self.optimizer = optimizer
        self.network = network
        self.initializer = initializer

        self.history: list = []
        self.results: list = []

        # set seed
        tf.random.set_seed(self.trainer["seed"])

    def fit(
        self,
        save_history: SaveHist = save_history(),
        save_results: SaveResults = save_results(),
        overwrite: bool = False,
        parallel: Optional[Parallel] = None,
    ) -> None:
        """
        Fit the eliobj and learn prior distributions.

        Parameters
        ----------
        overwrite : bool
            If the eliobj was already fitted and the user wants to refit it,
            the user is asked whether they want to overwrite the previous
            fitting results. Setting ``overwrite=True`` allows the user to
            force overfitting without being prompted. The default is ``False``.
        save_history : dict, :func:`elicit.utils.save_history`
            Exclude or include sub-results in the final result file.
            In the ``history`` object are all results that are saved across epochs.
            For usage information see
            `How-To: Save and load the eliobj <https://florence-bockting.github.io/prior_elicitation/howto/saving_loading.html>`_
        save_results : dict, :func:`elicit.utils.save_results`
            Exclude or include sub-results in the final result file.
            In the ``results`` object are all results that are saved for the last
            epoch only. For usage information see
            `How-To: Save and load the eliobj <https://florence-bockting.github.io/prior_elicitation/howto/saving_loading.html>`_
        parallel : dict from :func:`elicit.utils.parallel`, optional
            specify parallelization settings if multiple trainings should run
            in parallel.

        Examples
        --------
        >>> eliobj.fit()  # doctest: +SKIP

        >>> eliobj.fit(overwrite=True,
        >>>            save_history=el.utils.save_history(
        >>>                loss_component=False
        >>>                )
        >>>            )  # doctest: +SKIP

        >>> eliobj.fit(parallel=el.utils.parallel(runs=4))  # doctest: +SKIP

        """
        # set seed
        tf.random.set_seed(self.trainer["seed"])

        # check whether elicit object is already fitted
        refit = True
        if len(self.history) != 0 and not overwrite:
            user_answ = input(
                "eliobj is already fitted."
                + " Do you want to fit it again and overwrite the results?"
                + " Press 'n' to stop process and 'y' to continue fitting."
            )

            while user_answ not in ["n", "y"]:
                user_answ = input(
                    "Please press either 'y' for fitting or 'n'"
                    + " for abording the process."
                )

            if user_answ == "n":
                refit = False
                print("Process aborded; eliobj is not re-fitted.")

        # run single time if no parallelization is required
        if (parallel is None) and (refit):
            results, history = self.workflow(self.trainer["seed"])
            # include seed information into results
            results["seed"] = self.trainer["seed"]
            # remove results that user wants to exclude from saving
            results_prep, history_prep = clean_savings(
                history, results, save_history, save_results
            )
            # save results in list attribute
            self.history.append(history_prep)
            self.results.append(results_prep)
        # run multiple replications
        if (parallel is not None) and (refit):
            # create a list of seeds if not provided
            if parallel["seeds"] is None:
                # generate seeds
                seeds = [
                    int(s) for s in tfd.Uniform(0, 999999).sample(parallel["runs"])
                ]
            else:
                seeds = parallel["seeds"]

            # run training simultaneously for multiple seeds
            (*res,) = joblib.Parallel(n_jobs=parallel["cores"])(
                joblib.delayed(self.workflow)(seed) for seed in seeds
            )

            for i, seed in enumerate(seeds):
                self.results.append(res[i][0])
                self.history.append(res[i][1])
                self.results[i]["seed"] = seed

                self.results[i], self.history[i] = clean_savings(
                    self.history[i], self.results[i], save_history, save_results
                )

    def save(
        self,
        name: Optional[str] = None,
        file: Optional[str] = None,
        overwrite: bool = False,
    ):
        """
        Save data on disk

        Parameters
        ----------
        name: str, optional
            file name used to store the eliobj. Saving is done
            according to the following rule: ``./{method}/{name}_{seed}.pkl``
            with 'method' and 'seed' being arguments of
            :func:`elicit.elicit.trainer`.
        file : str, optional
            user-specific path for saving the eliobj. If file is specified
            **name** must be ``None``. The default value is ``None``.
        overwrite : bool
            If already a fitted object exists in the same path, the user is
            asked whether the eliobj should be refitted and the results
            overwritten. With the ``overwrite`` argument, you can disable this
            behavior. In this case the results are automatically overwritten
            without prompting the user. The default is ``False``.

        Raises
        ------
        AssertionError
            ``name`` and ``file`` can't be specified simultaneously.

        Examples
        --------
            >>> eliobj.save(name="toymodel")  # doctest: +SKIP

            >>> eliobj.save(file="res/toymodel", overwrite=True)  # doctest: +SKIP

        """
        # check that either name or file is specified
        if not (name is None) ^ (file is None):
            msg = (
                "Name and file cannot be both None or both specified.",
                "Either one has to be None.",
            )
            raise AssertionError(msg)

        # add a saving path
        return save(self, name=name, file=file, overwrite=overwrite)

    def update(self, **kwargs):
        """
        Update attributes of Elicit object

        Method for updating the attributes of the Elicit class. Updating
        an eliobj leads to an automatic reset of results.

        Parameters
        ----------
        **kwargs
            keyword argument used for updating an attribute of Elicit class.
            Key must correspond to one attribute of the class and value refers
            to the updated value.

        Raises
        ------
        ValueError
            key of provided keyword argument is not an eliobj attribute. Please
            check dir(eliobj).

        Examples
        --------
        >>> eliobj.update(parameter=updated_parameter_dict)  # doctest: +SKIP

        """
        # check that arguments exist as eliobj attributes
        for key in kwargs:
            if str(key) not in [
                "model",
                "parameters",
                "targets",
                "expert",
                "trainer",
                "optimizer",
                "network",
                "initializer",
            ]:
                msg = (
                    f"{key=} is not an eliobj attribute.",
                    "Use dir() to check for attributes.",
                )
                raise ValueError(msg)

        for key in kwargs:  # noqa: PLC0206
            setattr(self, key, kwargs[key])
            # reset results
            self.results: list = list()
            self.history: list = list()
            # inform user about reset of results
            print("INFO: Results have been reset.")

    def workflow(self, seed: int) -> tuple[dict, dict]:
        """
        Build the main workflow of the prior elicitation method.

        Get expert data, initialize method, run optimization.
        Results are returned for further post-processing.

        Parameters
        ----------
        seed : int
            seed information used for reproducing results.

        Returns
        -------
        results, history = Tuple(dict, dict)
            results of the optimization process.

        """
        self.trainer["seed_chain"] = seed
        # get expert data
        expert_elicits, expert_prior = get_expert_data(
            self.trainer,
            self.model,
            self.targets,
            self.expert,
            self.parameters,
            self.network,
            self.trainer["seed"],
        )

        # initialization of hyperparameter
        (init_prior_model, loss_list, init_prior_obj, init_matrix) = (
            initialization.init_prior(
                expert_elicits,
                self.initializer,
                self.parameters,
                self.trainer,
                self.model,
                self.targets,
                self.network,
                self.expert,
                seed,
            )
        )

        # run dag with optimal set of initial values
        # save results in corresp. attributes
        history, results = optimization.sgd_training(
            expert_elicits,
            init_prior_model,
            self.trainer,
            self.optimizer,
            self.model,
            self.targets,
            self.parameters,
            seed,
        )
        # add some additional results
        results["expert_elicited_statistics"] = expert_elicits
        try:
            self.expert["ground_truth"]
        except KeyError:
            pass
        else:
            results["expert_prior_samples"] = expert_prior

        if self.trainer["method"] == "parametric_prior":
            results["init_loss_list"] = loss_list
            results["init_prior"] = init_prior_obj
            results["init_matrix"] = init_matrix

        return results, history
