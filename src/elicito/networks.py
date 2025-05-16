"""
setup network argument of Elicit class

imported code from BayesFlow==1.1.6 with approval by author Stefan Radev
Code needs to be adjusted to elicito structure
"""

import copy
from typing import Any, Callable, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore
from numpy import e as EULER_CONST
from numpy import pi as PI_CONST
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from elicito.types import NFDict

tfd = tfp.distributions


class MetaDictSetting:
    """Implement interface for a default meta_dict"""

    def __init__(self, meta_dict: dict, mandatory_fields: list = []):
        """Configure meta dict with mandatory arguments

        Parameters
        ----------
        meta_dict        : dict
            Default dictionary.
        mandatory_fields : list, default: []
            List of keys in `meta_dict` that need to be provided by the user.
        """
        self.meta_dict = meta_dict
        self.mandatory_fields = mandatory_fields


DEFAULT_SETTING_INVERTIBLE_NET = MetaDictSetting(
    meta_dict={
        "num_coupling_layers": 5,
        "coupling_net_settings": None,
        "coupling_design": "affine",
        "permutation": "fixed",
        "use_act_norm": True,
        "act_norm_init": None,
        "use_soft_flow": False,
        "soft_flow_bounds": (1e-3, 5e-2),
    },
    mandatory_fields=["num_params"],
)


DEFAULT_SETTING_AFFINE_COUPLING = MetaDictSetting(
    meta_dict={
        "dense_args": dict(
            units=128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        ),
        "num_dense": 2,
        "spec_norm": False,
        "mc_dropout": False,
        "dropout": True,
        "residual": False,
        "dropout_prob": 0.05,
        "soft_clamping": 1.9,
    },
    mandatory_fields=[],
)


DEFAULT_SETTING_SPLINE_COUPLING = MetaDictSetting(
    meta_dict={
        "dense_args": dict(
            units=128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        ),
        "num_dense": 2,
        "spec_norm": False,
        "mc_dropout": False,
        "dropout": True,
        "residual": False,
        "dropout_prob": 0.05,
        "bins": 16,
        "default_domain": (-5.0, 5.0, -5.0, 5.0),
    },
    mandatory_fields=[],
)


def NF(
    inference_network: Callable[[Any], Any],
    network_specs: dict[str, Any],
    base_distribution: Callable[[Any], Any],
) -> NFDict:
    """
    Specify normalizing flow used from BayesFlow library

    Parameters
    ----------
    inference_network
        type of inference network as specified by bayesflow.inference_networks.

    network_specs
        specification of normalizing flow architecture. Arguments are inherited
        from chosen bayesflow.inference_networks.

    base_distribution
        Base distribution from which should be sampled during learning.
        Normally the base distribution is a multivariate normal.

    Returns
    -------
    nf_dict :
        dictionary specifying the normalizing flow settings.

    """
    nf_dict: NFDict = dict(
        inference_network=inference_network,
        network_specs=network_specs,
        base_distribution=base_distribution,
    )

    return nf_dict


class BaseNormal:
    """
    standard normal base distribution for normalizing flow
    """

    def __call__(self, num_params: int) -> Any:
        """
        Multivariate standard normal distribution

        distribution has as many dimensions as parameters in the generative model.

        Parameters
        ----------
        num_params
            number of model parameters.

        Returns
        -------
        :
            tfp.distributions object.

        """
        base_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(num_params), scale_diag=tf.ones(num_params)
        )
        return base_dist


# initialized instance of the BaseNormal class
base_normal = BaseNormal()


class DenseCouplingNet(tf.keras.Model):
    """Implement a conditional version of a standard fully connected network.

    Would also work as an unconditional estimator.
    """

    def __init__(self, settings, dim_out, **kwargs):
        """Create a conditional coupling net (FC neural network).

        Parameters
        ----------
        settings : dict
            A dictionary holding arguments for a dense layer:
            See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

            As well as custom arguments for settings such as residual networks,
            dropout, and spectral normalization.
        dim_out  : int
            Number of outputs of the coupling net. Determined internally by the
            consumer classes.
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the `tf.keras.Model` constructor.
        """
        super().__init__(**kwargs)

        # Create network body (input and hidden layers)
        self.fc = Sequential()
        for _ in range(settings["num_dense"]):
            # Create dense layer with dict kwargs
            layer = Dense(**settings["dense_args"])

            # Wrap in spectral normalization, if specified
            if settings.get("spec_norm") is True:
                layer = SpectralNormalization(layer)
            self.fc.add(layer)

            # Figure out which dropout to use, MC has precedence over standard
            # Fails gently, if no dropout_prob is specified
            # Case both specified, MC wins
            if settings.get("dropout") and settings.get("mc_dropout"):
                self.fc.add(MCDropout(dropout_prob=settings["dropout_prob"]))

            # Case only dropout, use standard
            elif settings.get("dropout") and not settings.get("mc_dropout"):
                self.fc.add(Dropout(rate=settings["dropout_prob"]))

            # Case only MC, use MC
            elif not settings.get("dropout") and settings.get("mc_dropout"):
                self.fc.add(MCDropout(dropout_prob=settings["dropout_prob"]))

            # No dropout
            else:
                pass

        # Set residual flag
        if settings.get("residual"):
            self.fc.add(
                Dense(
                    dim_out,
                    **{k: v for k, v in settings["dense_args"].items() if k != "units"},
                )
            )
            self.residual_output = Dense(dim_out, kernel_initializer="zeros")
        else:
            self.fc.add(Dense(dim_out, kernel_initializer="zeros"))
            self.residual_output = None

        self.fc.build(input_shape=())

    def call(self, target, condition, **kwargs):
        r"""Concatenate target and condition (forward mode)

        Parameters
        ----------
        target      : tf.Tensor
          The split estimation quntities, for instance,
          parameters :math:`\\theta \\sim p(\\theta)` of interest,
          shape (batch_size, ...)
        condition   : tf.Tensor or None
            the conditioning vector of interest, for instance ``x = summary(x)``,
            shape (batch_size, summary_dim)
        """
        # Handle case no condition
        if condition is None:
            if self.residual_output is not None:
                return self.residual_output(
                    self.fc(target, **kwargs) + target, **kwargs
                )
            else:
                return self.fc(target, **kwargs)

        # Handle 3D case for a set-flow and repeat condition over
        # the second `time` or `n_observations` axis of `target``
        if len(tf.shape(target)) == 3 and len(tf.shape(condition)) == 2:  # noqa: PLR2004
            shape = tf.shape(target)
            condition = tf.expand_dims(condition, 1)
            condition = tf.tile(condition, [1, shape[1], 1])
        inp = tf.concat((target, condition), axis=-1)
        out = self.fc(inp, **kwargs)

        if self.residual_output is not None:
            out = self.residual_output(out + target, **kwargs)
        return out


class SpectralNormalization(tf.keras.layers.Wrapper):
    """Performs spectral normalization on neural network weights.

    Adapted from:
    https://www.tensorflow.org/addons/api_docs/python/tfa/layers/SpectralNormalization

    This wrapper controls the Lipschitz constant of a layer by
    constraining its spectral norm, which can stabilize the
    training of generative networks.

    See [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957).
    """

    def __init__(self, layer, power_iterations=1, **kwargs):
        super().__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(  # noqa: TRY003
                "`power_iterations` should be greater than zero, got "
                f"`power_iterations={power_iterations}`"
            )
        self.power_iterations = power_iterations
        self._initialized = False

    def build(self, input_shape):
        """Build `Layer`"""
        # Register input shape
        super().build(input_shape)

        # Store reference to weights
        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(  # noqa: TRY003
                f"{type(self.layer).__name__} object has no attribute 'kernel' nor "
                "'embeddings'"
            )

        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="sn_u",
            dtype=self.w.dtype,
        )

    def call(self, inputs, training=False):
        """Call `Layer`

        Parameters
        ----------
        inputs : tf.Tensor of shape (None,...,condition_dim + target_dim)
            The inputs to the corresponding layer.
        """
        if training:
            self.normalize_weights()
        output = self.layer(inputs)
        return output

    def normalize_weights(self):
        """Generate spectral normalized weights.

        This method will update the value of `self.w` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """
        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        with tf.name_scope("spectral_normalize"):
            for _ in range(self.power_iterations):
                v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
                u = tf.math.l2_normalize(tf.matmul(v, w))
            u = tf.stop_gradient(u)
            v = tf.stop_gradient(v)
            sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
            self.u.assign(tf.cast(u, self.u.dtype))
            self.w.assign(
                tf.cast(tf.reshape(self.w / sigma, self.w_shape), self.w.dtype)
            )

    def get_config(self):  # noqa: D102
        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}


class Permutation(tf.keras.Model):
    """Implement a permutation layer

    layer to permute the inputs entering a (conditional) coupling layer.
    Uses fixed permutations, as these perform equally well compared to
    learned permutations.
    """

    def __init__(self, input_dim):
        """Create an invertible permutation layer

        Parameters
        ----------
        input_dim  : int
            Ihe dimensionality of the input to the (conditional)
            coupling layer.
        """
        super().__init__()

        permutation_vec = np.random.permutation(input_dim)  # noqa: NPY002
        inv_permutation_vec = np.argsort(permutation_vec)
        self.permutation = tf.Variable(
            initial_value=permutation_vec,
            trainable=False,
            dtype=tf.int32,
            name="permutation",
        )
        self.inv_permutation = tf.Variable(
            initial_value=inv_permutation_vec,
            trainable=False,
            dtype=tf.int32,
            name="inv_permutation",
        )

    def call(self, target, inverse=False):
        """Permute a batch of target vectors over the last axis.

        Parameters
        ----------
        target   : tf.Tensor of shape (batch_size, ...)
            The target vector to be permuted over its last axis.
        inverse  : bool, optional, default: False
            Controls if the current pass is forward (``inverse=False``)
            or inverse (``inverse=True``).

        Returns
        -------
        out      : tf.Tensor of the same shape as `target`.
            The (un-)permuted target vector.
        """
        if not inverse:
            return self._forward(target)
        else:
            return self._inverse(target)

    def _forward(self, target):
        """Perform a fixed permutation over the last axis."""
        return tf.gather(target, self.permutation, axis=-1)

    def _inverse(self, target):
        """Undo the fixed permutation over the last axis."""
        return tf.gather(target, self.inv_permutation, axis=-1)


class Orthogonal(tf.keras.Model):
    """Implement a learnable orthogonal transformation

    Implementation according to [1]. Can be used as an alternative
    to a fixed ``Permutation`` layer.

    [1] Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow
    with invertible 1x1 convolutions. Advances in neural information
    processing systems, 31.
    """

    def __init__(self, input_dim):
        """Create an invertible orthogonal transformation

        Parameters
        ----------
        input_dim  : int
            The dimensionality of the input to the (conditional)
            coupling layer.
        """
        super().__init__()

        init = tf.keras.initializers.Orthogonal()
        self.W = tf.Variable(
            initial_value=init(shape=(input_dim, input_dim)),
            trainable=True,
            dtype=tf.float32,
            name="learnable_permute",
        )

    def call(self, target, inverse=False):
        """Transform a batch of target vectors

        Transformation over the last axis through an approximately
        orthogonal transform.

        Parameters
        ----------
        target   : tf.Tensor of shape (batch_size, ...)
            The target vector to be rotated over its last axis.
        inverse  : bool, optional, default: False
            Controls if the current pass is forward (``inverse=False``)
            or inverse (``inverse=True``).

        Returns
        -------
        out      : tf.Tensor of the same shape as `target`.
            The (un-)rotated target vector.
        """
        if not inverse:
            return self._forward(target)
        else:
            return self._inverse(target)

    def _forward(self, target):
        """Perform a learnable generalized permutation over the last axis."""
        shape = tf.shape(target)
        rank = len(shape)
        log_det = tf.math.log(tf.math.abs(tf.linalg.det(self.W)))
        if rank == 2:  # noqa: PLR2004
            z = tf.linalg.matmul(target, self.W)
        else:
            z = tf.tensordot(target, self.W, [[rank - 1], [0]])
            log_det = tf.cast(shape[1], tf.float32) * log_det
        return z, log_det

    def _inverse(self, z):
        """Undo the learnable permutation over the last axis."""
        W_inv = tf.linalg.inv(self.W)
        rank = len(tf.shape(z))
        if rank == 2:  # noqa: PLR2004
            return tf.linalg.matmul(z, W_inv)
        return tf.tensordot(z, W_inv, [[rank - 1], [0]])


class MCDropout(tf.keras.Model):
    """Implement Monte Carlo Dropout

    Dropout is implemented as a Bayesian approximation according to [1].

    [1] Gal, Y., & Ghahramani, Z. (2016, June). Dropout as a bayesian
    approximation: Representing model uncertainty in deep learning.
    In international conference on machine learning (pp. 1050-1059). PMLR.
    """

    def __init__(self, dropout_prob=0.1, **kwargs):
        """Create a custom instance of an MC Dropout layer.

        Will be used both during training and inference.

        Parameters
        ----------
        dropout_prob  : float, optional, default: 0.1
            The dropout rate to be passed to ``tf.keras.layers.Dropout()``.
        """
        super().__init__(**kwargs)
        self.drop = Dropout(dropout_prob)

    def call(self, inputs):
        """Set randomly elements of ``inputs`` to zero.

        Parameters
        ----------
        inputs : tf.Tensor
            Input of shape (batch_size, ...)

        Returns
        -------
        out    : tf.Tensor
            Output of shape (batch_size, ...), same as ``inputs``.

        """
        out = self.drop(inputs, training=True)
        return out


class ActNorm(tf.keras.Model):
    """Implement an Activation Normalization (ActNorm) Layer.

    Activation Normalization is learned invertible normalization,
    using a Scale (s) and Bias (b) vector::

       y = s * x + b(forward)
       x = (y - b) / s(inverse)

    Notes
    -----
    The scale and bias can be data dependent initialized, such that the
    output has a mean of zero and standard deviation of one [1]_[2]_.
    Alternatively, it is initialized with vectors of ones (scale) and
    zeros (bias).

    References
    ----------

    .. [1] Kingma, Diederik P., and Prafulla Dhariwal.
       "Glow: Generative flow with invertible 1x1 convolutions."
       arXiv preprint arXiv:1807.03039 (2018).

    .. [2] Salimans, Tim, and Durk P. Kingma.
       "Weight normalization: A simple reparameterization to accelerate
       training of deep neural networks."
       Advances in neural information processing systems 29 (2016): 901-909.
    """

    def __init__(self, latent_dim: int, act_norm_init: np.ndarray, **kwargs):
        """Create an instance of an ActNorm Layer as proposed by [1].

        Parameters
        ----------
        latent_dim
            The dimensionality of the latent space
            (equal to the dimensionality of the target variable)
        act_norm_init
            Optional data-dependent initialization for the internal
            ``ActNorm`` layers, as done in [1]. Could be helpful
            for deep invertible networks.
        """
        super().__init__(**kwargs)

        # Initialize scale and bias with zeros and ones if no batch for
        # initialization was provided.
        if act_norm_init is None:
            self.scale = tf.Variable(
                tf.ones((latent_dim,)), trainable=True, name="act_norm_scale"
            )

            self.bias = tf.Variable(
                tf.zeros((latent_dim,)), trainable=True, name="act_norm_bias"
            )
        else:
            self._initalize_parameters_data_dependent(act_norm_init)

    def call(self, target: tf.Tensor, inverse: bool = False) -> tuple[Any]:
        r"""Perform one pass through the actnorm layer

        (either inverse or forward) and normalizes the last axis of
        `target`.

        Parameters
        ----------
        target
            the target variables of interest, i.e., parameters for
            posterior estimation
        inverse
            Flag indicating whether to run the block forward or
            backwards

        Returns
        -------
        (z, log_det_J) :
            If inverse=False: The transformed input and the corresponding
            Jacobian of the transformation,
            v shape: (batch_size, inp_dim), log_det_J shape: (,)
        target :
            If inverse=True: The inversely transformed targets,
            shape == target.shape

        Notes
        -----
        If ``inverse=False``, the return is ``(z, log_det_J)``.\n
        If ``inverse=True``, the return is ``target``.
        """
        if not inverse:
            return self._forward(target)
        else:
            return self._inverse(target)

    def _forward(self, target):
        """Perform a forward pass through the layer."""
        z = self.scale * target + self.bias
        ldj = tf.math.reduce_sum(tf.math.log(tf.math.abs(self.scale)), axis=-1)
        return z, ldj

    def _inverse(self, target):
        """Perform an inverse pass through the layer."""
        return (target - self.bias) / self.scale

    def _initalize_parameters_data_dependent(self, init_data: tf.Tensor):
        """Perform a data dependent initalization of the scale and bias.

        Initalize the scale and bias vector as proposed by [1], such that the
        layer output has a mean of zero and a standard deviation of one.

        [1] - Salimans, Tim, and Durk P. Kingma.
        Weight normalization: A simple reparameterization to accelerate
        training of deep neural networks.
        Advances in neural information processing systems 29
        (2016): 901-909.

        Parameters
        ----------
        init_data
            Initiall values to estimate the scale and bias parameters by computing
            the mean and standard deviation along the first dimension of `init_data`.
        """
        # 2D Tensor case, assume first batch dimension
        if tf.rank(init_data) == 2:  # noqa: PLR2004
            mean = tf.math.reduce_mean(init_data, axis=0)
            std = tf.math.reduce_std(init_data, axis=0)
        # 3D Tensor case, assume first batch dimension, second number of
        # observations dimension
        elif tf.rank(init_data) == 3:  # noqa: PLR2004
            mean = tf.math.reduce_mean(init_data, axis=(0, 1))
            std = tf.math.reduce_std(init_data, axis=(0, 1))
        # Raise other cases
        else:
            raise ValueError(  # type: ignore
                "Currently, ActNorm supports only 2D and 3D Tensors, "
                + "but act_norm_init contains data with shape {init_data.shape}."
            )

        scale = 1.0 / std
        bias = (-1.0 * mean) / std

        self.scale = tf.Variable(scale, trainable=True, name="act_norm_scale")
        self.bias = tf.Variable(bias, trainable=True, name="act_norm_bias")


class InvertibleNetwork(tf.keras.Model):
    """Implement a chain of conditional invertible coupling layers

    Implementation for conditional density estimation.
    """

    available_designs = ("affine", "spline", "interleaved")

    def __init__(  # noqa: PLR0913
        self,
        num_params: int,
        num_coupling_layers: int = 6,
        coupling_design: str | Callable = "affine",
        coupling_settings: Optional[dict[str, Any]] = None,
        permutation: Optional[str] = "fixed",
        use_act_norm: bool = True,
        act_norm_init: Optional[np.ndarray] = None,
        use_soft_flow: bool = False,
        soft_flow_bounds: tuple[float, float] = (1e-3, 5e-2),
        **kwargs,
    ):
        """Create a chain of coupling layers

        Implementation with optional `ActNorm` layers in-between.
        Implements ideas from:

        [1] Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020).
        BayesFlow: Learning complex stochastic models with invertible neural networks.
        IEEE Transactions on Neural Networks and Learning Systems.

        [2] Kim, H., Lee, H., Kang, W. H., Lee, J. Y., & Kim, N. S. (2020).
        Softflow: Probabilistic framework for normalizing flow on manifolds.
        Advances in Neural Information Processing Systems, 33, 16388-16397.

        [3] Ardizzone, L., Kruse, J., Lüth, C., Bracher, N., Rother, C., & Köthe, U. (2020).
        Conditional invertible neural networks for diverse image-to-image translation.
        In DAGM German Conference on Pattern Recognition (pp. 373-387). Springer, Cham.

        [4] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019).
        Neural spline flows. Advances in Neural Information Processing Systems, 32.

        [5] Kingma, D. P., & Dhariwal, P. (2018).
        Glow: Generative flow with invertible 1x1 convolutions.
        Advances in Neural Information Processing Systems, 31.

        Parameters
        ----------
        num_params
            The number of parameters to perform inference on. Equivalently, the dimensionality of the
            latent space.
        num_coupling_layers
            The number of coupling layers to use as defined in [1] and [2]. In general, more coupling layers
            will give you more expressive power, but will be slower and may need more simulations to train.
            Typically, between 4 and 10 coupling layers should suffice for most applications.
        coupling_design
            The type of internal coupling network to use. Must be in ['affine', 'spline', 'interleaved'].
            The first corresponds to the architecture in [3, 5], the second corresponds to a modified
            version of [4]. The third option will alternate between affine and spline layers, for example,
            if num_coupling_layers == 3, the chain will consist of ["affine", "spline", "affine"] layers.

            In general, spline couplings run slower than affine couplings, but require fewer coupling
            layers. Spline couplings may work best with complex (e.g., multimodal) low-dimensional
            problems. The difference will become less and less pronounced as we move to higher dimensions.

            Note: This is the first setting you may want to change, if inference does not work as expected!
        coupling_settings
            The coupling network settings to pass to the internal coupling layers. See ``default_settings``
            for possible settings. Below are two examples.

        Examples
        --------
            1. If using ``coupling_design='affine``, you may want to turn on Monte Carlo Dropout and
            use an ELU activation function for the internal networks. You can do this by providing:
            ``
            coupling_settings={
                'mc_dropout' : True,
                'dense_args' : dict(units=128, activation='elu')
            }
            ``

            2. If using ``coupling_design='spline'``, you may want to change the number of learnable bins
            and increase the dropout probability (i.e., more regularization to guard against overfitting):
            ``
            coupling_settings={
                'dropout_prob': 0.2,
                'bins' : 32,
            }
            ``
        permutation
            Whether to use permutations between coupling layers. Highly recommended if ``num_coupling_layers > 1``
            Important: Must be in ['fixed', 'learnable', None]
        use_act_norm
            Whether to use activation normalization after each coupling layer, as used in [5].
            Recommended to keep default.
        act_norm_init
            Optional data-dependent initialization for the internal ``ActNorm`` layers, as done in [5]. Could be helpful
            for deep invertible networks.
        use_soft_flow
            Whether to perturb the target distribution (i.e., parameters) with small amount of independent
            noise, as done in [2]. Could be helpful for degenerate distributions.
        soft_flow_bounds
            The bounds of the continuous uniform distribution from which the noise scale would be sampled
            at each iteration. Only relevant when ``use_soft_flow=True``.
        **kwargs
            Optional keyword arguments (e.g., name) passed to the tf.keras.Model __init__ method.
        """  # noqa: E501
        super().__init__(**kwargs)

        layer_settings = dict(
            latent_dim=num_params,
            permutation=permutation,
            use_act_norm=use_act_norm,
            act_norm_init=act_norm_init,
        )
        self.coupling_layers = self._create_coupling_layers(
            layer_settings, coupling_settings, coupling_design, num_coupling_layers
        )
        self.soft_flow = use_soft_flow
        self.soft_low = soft_flow_bounds[0]
        self.soft_high = soft_flow_bounds[1]
        self.permutation = permutation
        self.use_act_norm = use_act_norm
        self.latent_dim = num_params

    def call(
        self, targets: tf.Tensor, condition: tf.Tensor, inverse: bool = False, **kwargs
    ) -> tuple[Any, tf.Tensor]:
        r"""Perform one pass through an invertible chain

        Can be either inverse or forward mode

        Parameters
        ----------
        targets
            The estimation quantities of interest,
            shape (batch_size, ...)
        condition
            The conditional data x, shape
            (batch_size, summary_dim)
        inverse
            Flag indicating whether to run the chain
            forward or backwards

        Returns
        -------
        (z, log_det_J) :
            If inverse=False: The transformed input and the
            corresponding Jacobian of the transformation,
            v shape: (batch_size, ...), log_det_J shape: (batch_size, ...)

        target :
            If inverse=True: The transformed out, shape
            (batch_size, ...)

        Notes
        -----
        If ``inverse=False``, the return is ``(z, log_det_J)``.\n
        If ``inverse=True``, the return is ``target``.
        """
        if inverse:
            return self.inverse(targets, condition, **kwargs)
        return self.forward(targets, condition, **kwargs)

    def forward(self, targets, condition, **kwargs):
        """Perform a forward pass through the chain."""
        # Add noise to target if using SoftFlow, use explicitly
        # not in call(), since methods are public
        if self.soft_flow and condition is not None:
            # Extract shapes of tensors
            target_shape = tf.shape(targets)
            condition_shape = tf.shape(condition)

            # Needs to be concatinable with condition
            if len(condition_shape) == 2:  # noqa: PLR2004
                shape_scale = (condition_shape[0], 1)
            else:
                shape_scale = (condition_shape[0], condition_shape[1], 1)

            # Case training mode
            if kwargs.get("training"):
                noise_scale = tf.random.uniform(
                    shape=shape_scale, minval=self.soft_low, maxval=self.soft_high
                )
            # Case inference mode
            else:
                noise_scale = tf.zeros(shape=shape_scale) + self.soft_low

            # Perturb data with noise (will broadcast to all dimensions)
            if len(shape_scale) == 2 and len(target_shape) == 3:  # noqa: PLR2004
                targets += tf.expand_dims(noise_scale, axis=1) * tf.random.normal(
                    shape=target_shape
                )
            else:
                targets += noise_scale * tf.random.normal(shape=target_shape)

            # Augment condition with noise scale variate
            condition = tf.concat((condition, noise_scale), axis=-1)

        z = targets
        log_det_Js = []
        for layer in self.coupling_layers:
            z, log_det_J = layer(z, condition, **kwargs)
            log_det_Js.append(log_det_J)

        # Sum Jacobian determinants for all layers (coupling blocks) to
        # obtain total Jacobian.
        log_det_J = tf.add_n(log_det_Js)
        return z, log_det_J

    def inverse(self, z, condition, **kwargs):
        """Perform a reverse pass through the chain.

        Assumes that it is only used in inference mode, so
        ``**kwargs`` contains ``training=False``.
        """
        # Add noise to target if using SoftFlow, use explicitly
        # not in call(), since methods are public
        if self.soft_flow and condition is not None:
            # Needs to be concatinable with condition
            shape_scale = (
                (condition.shape[0], 1)
                if len(condition.shape) == 2  # noqa: PLR2004
                else (condition.shape[0], condition.shape[1], 1)
            )
            noise_scale = tf.zeros(shape=shape_scale) + 2.0 * self.soft_low

            # Augment condition with noise scale variate
            condition = tf.concat((condition, noise_scale), axis=-1)

        target = z
        for layer in reversed(self.coupling_layers):
            target = layer(target, condition, inverse=True, **kwargs)
        return target

    @staticmethod
    def _create_coupling_layers(
        settings, coupling_settings, coupling_design, num_coupling_layers
    ):
        """Create a list of coupling layers.

        Takes care of the different options for coupling design.
        """
        if coupling_design not in InvertibleNetwork.available_designs:
            raise NotImplementedError(
                "Coupling design should be one of", InvertibleNetwork.available_designs
            )

        # Case affine or spline
        if coupling_design != "interleaved":
            design = coupling_design
            _coupling_settings = coupling_settings
            coupling_layers = [
                CouplingLayer(
                    coupling_design=design,
                    coupling_settings=_coupling_settings,
                    **settings,
                )
                for _ in range(num_coupling_layers)
            ]
        # Case interleaved, starts with affine
        else:
            coupling_layers = []
            designs = (["affine", "spline"] * int(np.ceil(num_coupling_layers / 2)))[
                :num_coupling_layers
            ]
            for design in designs:
                # Fail gently, if neither None, nor a dictionary with
                # keys ("spline", "affine")
                _coupling_settings = (
                    None if coupling_settings is None else coupling_settings[design]
                )
                layer = CouplingLayer(
                    coupling_design=design,
                    coupling_settings=_coupling_settings,
                    **settings,
                )
                coupling_layers.append(layer)
        return coupling_layers

    @classmethod
    def create_config(cls, **kwargs):
        """Create the settings dictionary

        Used for the internal networks of the invertible network.
        """
        settings = build_meta_dict(
            user_dict=kwargs, default_setting=DEFAULT_SETTING_INVERTIBLE_NET
        )
        return settings


class AffineCoupling(tf.keras.Model):
    """Implement a conditional affine coupling block

    Implementation according to [1, 2], with additional
    options, such as residual blocks or Monte Carlo Dropout.


    [1] Kingma, D. P., & Dhariwal, P. (2018).
    Glow: Generative flow with invertible 1x1 convolutions.
    Advances in neural information processing systems, 31.

    [2] Ardizzone, L., Lüth, C., Kruse, J., Rother, C., & Köthe, U. (2019).
    Guided image generation with conditional invertible neural networks.
    arXiv preprint arXiv:1907.02392.
    """

    def __init__(self, dim_out: int, settings_dict: dict[str, Any], **kwargs):
        """Create one half of an affine coupling layer

        To be used as part of a ``CouplingLayer`` in an
        ``InvertibleNetwork`` instance.

        Parameters
        ----------
        dim_out
            The output dimensionality of the affine coupling layer.
        settings_dict
            The settings for the inner networks. Defaults will use:
            ``settings_dict={
                "dense_args"    : dict(units=128, activation="relu"),
                "num_dense"     : 2,
                "spec_norm"     : False,
                "mc_dropout"    : False,
                "dropout"       : True,
                "residual"      : False,
                "dropout_prob"  : 0.01,
                "soft_clamping" : 1.9
            }
            ``
        """
        super().__init__(**kwargs)

        self.dim_out = dim_out
        self.soft_clamp = settings_dict["soft_clamping"]

        # Check if separate settings for s and t are provided and adjust accordingly
        if (
            settings_dict.get("s_args") is not None
            and settings_dict.get("t_args") is not None
        ):
            s_settings, t_settings = (
                settings_dict.get("s_args"),
                settings_dict.get("t_args"),
            )
        elif (
            settings_dict.get("s_args") is not None
            and settings_dict.get("t_args") is None
        ):
            raise ValueError(  # noqa: TRY003
                "s_args were provided, but you also need to provide t_args!"
            )
        elif (
            settings_dict.get("s_args") is None
            and settings_dict.get("t_args") is not None
        ):
            raise ValueError(  # noqa: TRY003
                "t_args were provided, but you also need to provide s_args!"
            )
        else:
            s_settings, t_settings = settings_dict, settings_dict

        # Internal network (learnable scale and translation)
        self.scale = DenseCouplingNet(s_settings, dim_out)
        self.translate = DenseCouplingNet(t_settings, dim_out)

    def call(
        self,
        split1: tf.Tensor,
        split2: tf.Tensor,
        condition: Optional[tf.Tensor],
        inverse: Optional[bool] = False,
        **kwargs,
    ) -> tuple[Any, tf.Tensor]:
        """Perform one pass through an affine coupling layer

        Parameters
        ----------
        split1
            The first partition of the input vector(s),
            shape (batch_size, ..., input_dim//1)
        split2
            The second partition of the input vector(s)
            shape (batch_size, ..., ceil[input_dim//2])
        condition
            The conditioning data of interest, for instance,
            x = summary_fun(x), shape (batch_size, ...).
            If ``condition is None``, then the layer reduces
            to an unconditional coupling.
        inverse
            Flag indicating whether to run the block forward
            or backward.

        Returns
        -------
        (z, log_det_J) :
            If inverse=False: The transformed input and the
            corresponding Jacobian of the transformation,
            z shape: (batch_size, ..., input_dim//2),
            log_det_J shape: (batch_size, ...)

        target :
            If inverse=True: The back-transformed z,
            shape (batch_size, ..., inp_dim//2)
        """
        if not inverse:
            return self._forward(split1, split2, condition, **kwargs)
        return self._inverse(split1, split2, condition, **kwargs)

    def _forward(
        self, u1: tf.Tensor, u2: tf.Tensor, condition: Optional[tf.Tensor], **kwargs
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Perform a forward pass through the coupling layer.

        Used internally by the instance.

        Parameters
        ----------
        v1
            The first partition of the input,
            shape (batch_size, ..., dim_1)
        v2
            The second partition of the input,
            shape (batch_size, ..., dim_2)
        condition
            The optional conditioning vector.
            Batch size must match the batch size
            of the partitions: (batch_size, ..., dim_condition)

        Returns
        -------
        (v, log_det_J) :
            The transformed input and the corresponding
            Jacobian of the transformation.
        """
        s = self.scale(u2, condition, **kwargs)
        if self.soft_clamp is not None:
            s = (2.0 * self.soft_clamp / PI_CONST) * tf.math.atan(s / self.soft_clamp)
        t = self.translate(u2, condition, **kwargs)
        v = u1 * tf.math.exp(s) + t
        log_det_J = tf.reduce_sum(s, axis=-1)
        return v, log_det_J

    def _inverse(
        self, v1: tf.Tensor, v2: tf.Tensor, condition: Optional[tf.Tensor], **kwargs
    ) -> tf.Tensor:
        """Perform an inverse pass through the affine coupling block.

        Used internally by the instance.

        Parameters
        ----------
        v1
            The first partition of the latent vector,
            shape (batch_size, ..., dim_1)
        v2
            The second partition of the latent vector,
            shape (batch_size, ..., dim_2)
        condition
            The optional conditioning vector.
            Batch size must match the batch size
            of the partitions: (batch_size, ..., dim_condition)

        Returns
        -------
        u :
            The back-transformed input.
        """
        s = self.scale(v1, condition, **kwargs)
        if self.soft_clamp is not None:
            s = (2.0 * self.soft_clamp / PI_CONST) * tf.math.atan(s / self.soft_clamp)
        t = self.translate(v1, condition, **kwargs)
        u = (v2 - t) * tf.math.exp(-s)
        return u


class SplineCoupling(tf.keras.Model):
    """Implement a conditional spline coupling block

    Implementation according to [1, 2], with additional
    options, such as residual blocks or Monte Carlo Dropout.

    [1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019).
    Neural spline flows. Advances in Neural Information Processing Systems, 32.

    [2] Ardizzone, L., Lüth, C., Kruse, J., Rother, C., & Köthe, U. (2019).
    Guided image generation with conditional invertible neural networks.
    arXiv preprint arXiv:1907.02392.

    Implement only rational quadratic splines (RQS),
    since these appear to work best in practice and
    lead to stable training.
    """

    def __init__(self, dim_out: int, settings_dict: dict[str, Any], **kwargs):
        """Create one half of a spline coupling layer

        To be used as part of a ``CouplingLayer`` in an
        ``InvertibleNetwork`` instance.

        Parameters
        ----------
        dim_out
            The output dimensionality of the coupling layer.
        settings_dict
            The settings for the inner networks.
            Defaults will use:
            ``settings_dict={
                "dense_args"     : dict(units=128, activation="relu"),
                "num_dense"      : 2,
                "spec_norm"      : False,
                "mc_dropout"     : False,
                "dropout"        : True,
                "residual"       : False,
                "dropout_prob"   : 0.05,
                "bins"           : 16,
                "default_domain" : (-5., 5., -5., 5.)
            }
            ``
        """
        super().__init__(**kwargs)

        self.dim_out = dim_out
        self.bins = settings_dict["bins"]
        self.default_domain = settings_dict["default_domain"]
        self.spline_params_counts = {
            "left_edge": 1,
            "bottom_edge": 1,
            "widths": self.bins,
            "heights": self.bins,
            "derivatives": self.bins - 1,
        }
        self.num_total_spline_params = (
            sum(self.spline_params_counts.values()) * self.dim_out
        )

        # Internal network (learnable spline parameters)
        self.net = DenseCouplingNet(settings_dict, self.num_total_spline_params)

    def call(
        self,
        split1: tf.Tensor,
        split2: tf.Tensor,
        condition: Optional[tf.Tensor],
        inverse: bool = False,
        **kwargs,
    ) -> tuple[Any, tf.Tensor]:
        """Perform one pass through a spline coupling layer

        Pass either inverse or forward.

        Parameters
        ----------
        split1
            The first partition of the input vector(s),
            shape (batch_size, ..., input_dim//2)
        split2
            The second partition of the input vector(s),
            shape (batch_size, ..., input_dim//2)
        condition
            The conditioning data of interest, for instance,
            x = summary_fun(x), shape (batch_size, ...).
            If ``condition is None``, then the layer reduces
            to an unconditional coupling.
        inverse
            Flag indicating whether to run the block forward
            or backward.

        Returns
        -------
        (z, log_det_J)  :
            If inverse=False: The transformed input and the
            corresponding Jacobian of the transformation,
            z shape: (batch_size, ..., input_dim//2),
            log_det_J shape: (batch_size, ...)

        target          :
            If inverse=True: The back-transformed z,
            shape (batch_size, ..., inp_dim//2)
        """
        if not inverse:
            return self._forward(split1, split2, condition, **kwargs)
        return self._inverse(split1, split2, condition, **kwargs)

    def _forward(
        self, u1: tf.Tensor, u2: tf.Tensor, condition: Optional[tf.Tensor], **kwargs
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Perform a forward pass through the spline coupling layer.

        Used internally by the instance.

        Parameters
        ----------
        v1
            The first partition of the input,
            shape (batch_size, ..., dim_1)
        v2
            The second partition of the input,
            shape (batch_size, ..., dim_2)
        condition
            The optional conditioning vector.
            Batch size must match the batch size of
            the partitions: (batch_size, ..., dim_condition)

        Returns
        -------
        (v, log_det_J)  :
            The transformed input and the corresponding
            Jacobian of the transformation.
        """
        spline_params = self.net(u2, condition, **kwargs)
        spline_params = self._semantic_spline_parameters(spline_params)
        spline_params = self._constrain_parameters(spline_params)
        v, log_det_J = self._calculate_spline(u1, spline_params, inverse=False)
        return v, log_det_J

    def _inverse(self, v1, v2, condition, **kwargs):
        """Perform an inverse pass through the coupling block.

        Used internally by the instance.

        Parameters
        ----------
        v1        : tf.Tensor of shape (batch_size, ..., dim_1)
            The first partition of the latent vector
        v2        : tf.Tensor of shape (batch_size, ..., dim_2)
            The second partition of the latent vector
        condition : tf.Tensor of shape (batch_size, ..., dim_condition)
            The optional conditioning vector. Batch size must match the batch size
            of the partitions.

        Returns
        -------
        u  :  tf.Tensor of shape (batch_size, ..., dim_1)
            The back-transformed input.
        """
        spline_params = self.net(v1, condition, **kwargs)
        spline_params = self._semantic_spline_parameters(spline_params)
        spline_params = self._constrain_parameters(spline_params)
        u = self._calculate_spline(v2, spline_params, inverse=True)
        return u

    def _calculate_spline(  # noqa: PLR0915
        self, target: tf.Tensor, spline_params: tuple[Any], inverse: bool = False
    ) -> tuple[Any, tf.Tensor]:
        """Compute both directions of a rational quadratic spline

        Relevant citation:
        https://github.com/vislearn/FrEIA/blob/master/FrEIA/modules/splines/rational_quadratic.py

        At this point, ``spline_params`` represents a tuple with
        the parameters of the RQS learned by the internal neural
        network (given optional conditional information).

        Parameters
        ----------
        target
            The target partition of the input vector to transform.
            shape (batch_size, ..., dim_2)
        spline_params
            A tuple with tensors corresponding to the learnable
            spline features: (left_edge, bottom_edge, widths, heights,
            derivatives)
        inverse
            Flag indicating whether to run the block forward or backward.

        Returns
        -------
        (result, log_det_J) :
            If inverse=False: The transformed input and the corresponding
            Jacobian of the transformation,
            result shape: (batch_size, ..., dim_2),
            log_det_J shape: (batch_size, ...)

        result              :
            If inverse=True: The back-transformed latent,
            shape (batch_size, ..., dim_2)
        """
        # Extract all learnable parameters
        left_edge, bottom_edge, widths, heights, derivatives = spline_params

        # Placeholders for results
        result = tf.zeros_like(target)
        log_jac = tf.zeros_like(target)

        total_width = tf.reduce_sum(widths, axis=-1, keepdims=True)
        total_height = tf.reduce_sum(heights, axis=-1, keepdims=True)

        knots_x = tf.concat(
            [left_edge, left_edge + tf.math.cumsum(widths, axis=-1)], axis=-1
        )
        knots_y = tf.concat(
            [bottom_edge, bottom_edge + tf.math.cumsum(heights, axis=-1)], axis=-1
        )

        # Determine which targets are in domain and which are not
        if not inverse:
            target_in_domain = tf.logical_and(
                knots_x[..., 0] < target, target <= knots_x[..., -1]
            )
            higher_indices = tf.searchsorted(knots_x, target[..., None])
        else:
            target_in_domain = tf.logical_and(
                knots_y[..., 0] < target, target <= knots_y[..., -1]
            )
            higher_indices = tf.searchsorted(knots_y, target[..., None])
        target_in = target[target_in_domain]
        target_in_idx = tf.where(target_in_domain)
        target_out = target[~target_in_domain]
        target_out_idx = tf.where(~target_in_domain)

        # In-domain computation
        if tf.size(target_in_idx) > 0:
            # Index crunching
            higher_indices = tf.gather_nd(higher_indices, target_in_idx)
            higher_indices = tf.cast(higher_indices, tf.int32)
            lower_indices = higher_indices - 1
            lower_idx_tuples = tf.concat(
                [tf.cast(target_in_idx, tf.int32), lower_indices], axis=-1
            )
            higher_idx_tuples = tf.concat(
                [tf.cast(target_in_idx, tf.int32), higher_indices], axis=-1
            )

            # Spline computation
            dk = tf.gather_nd(derivatives, lower_idx_tuples)
            dkp = tf.gather_nd(derivatives, higher_idx_tuples)
            xk = tf.gather_nd(knots_x, lower_idx_tuples)
            xkp = tf.gather_nd(knots_x, higher_idx_tuples)
            yk = tf.gather_nd(knots_y, lower_idx_tuples)
            ykp = tf.gather_nd(knots_y, higher_idx_tuples)
            x = target_in
            dx = xkp - xk
            dy = ykp - yk
            sk = dy / dx
            xi = (x - xk) / dx

            # Forward pass
            if not inverse:
                numerator = dy * (sk * xi**2 + dk * xi * (1 - xi))
                denominator = sk + (dkp + dk - 2 * sk) * xi * (1 - xi)
                result_in = yk + numerator / denominator
                # Log Jacobian for in-domain
                numerator = sk**2 * (
                    dkp * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2
                )
                denominator = (sk + (dkp + dk - 2 * sk) * xi * (1 - xi)) ** 2
                log_jac_in = tf.math.log(numerator + 1e-10) - tf.math.log(
                    denominator + 1e-10
                )
                log_jac = tf.tensor_scatter_nd_update(
                    log_jac, target_in_idx, log_jac_in
                )
            # Inverse pass
            else:
                y = x
                a = dy * (sk - dk) + (y - yk) * (dkp + dk - 2 * sk)
                b = dy * dk - (y - yk) * (dkp + dk - 2 * sk)
                c = -sk * (y - yk)
                discriminant = tf.maximum(b**2 - 4 * a * c, 0.0)
                xi = 2 * c / (-b - tf.math.sqrt(discriminant))
                result_in = xi * dx + xk

            result = tf.tensor_scatter_nd_update(result, target_in_idx, result_in)

        # Out-of-domain
        if tf.size(target_out_idx) > 1:
            scale = total_height / total_width
            shift = bottom_edge - scale * left_edge
            scale_out = tf.gather_nd(scale, target_out_idx)
            shift_out = tf.gather_nd(shift, target_out_idx)

            if not inverse:
                result_out = scale_out * target_out[..., None] + shift_out
                # Log Jacobian for out-of-domain points
                log_jac_out = tf.math.log(scale_out + 1e-10)
                log_jac_out = tf.squeeze(log_jac_out, axis=-1)
                log_jac = tf.tensor_scatter_nd_update(
                    log_jac, target_out_idx, log_jac_out
                )
            else:
                result_out = (target_out[..., None] - shift_out) / scale_out

            result_out = tf.squeeze(result_out, axis=-1)
            result = tf.tensor_scatter_nd_update(result, target_out_idx, result_out)

        if not inverse:
            return result, tf.reduce_sum(log_jac, axis=-1)
        return result

    def _semantic_spline_parameters(self, parameters: tf.Tensor) -> tuple[Any]:
        """Build a tuple of tensors from the output of the coupling net.

        Parameters
        ----------
        parameters
            All learnable spline parameters packed in a single tensor,
            which will be partitioned according to the role of each
            spline parameter. shape (batch_size, ..., num_spline_parameters)

        Returns
        -------
        parameters    :
            The partitioned spline parameters according to their
            role in the spline computation.
        """
        shape = tf.shape(parameters)
        rank = len(shape)
        if rank == 2:  # noqa: PLR2004
            new_shape = (shape[0], self.dim_out, -1)
        elif rank == 3:  # noqa: PLR2004
            new_shape = (shape[0], shape[1], self.dim_out, -1)
        else:
            raise NotImplementedError(
                "Spline flows can currently only operate on 2D and 3D inputs!"
            )
        parameters = tf.reshape(parameters, new_shape)
        parameters = tf.split(
            parameters, list(self.spline_params_counts.values()), axis=-1
        )
        return parameters

    def _constrain_parameters(self, parameters: tuple[Any]) -> tuple[Any]:
        """Take care of zero spline parameters

        Can happen due to zeros kernel initializer and
        applies parameter constraints for stability.

        Parameters
        ----------
        parameters
            The unconstrained spline parameters.

        Returns
        -------
        parameters :
            The constrained spline parameters.
        """
        left_edge, bottom_edge, widths, heights, derivatives = parameters

        # Set lower corners of domain relative to default domain
        left_edge = left_edge + self.default_domain[0]
        bottom_edge = bottom_edge + self.default_domain[2]

        # Compute default widths and heights
        default_width = (self.default_domain[1] - self.default_domain[0]) / self.bins
        default_height = (self.default_domain[3] - self.default_domain[2]) / self.bins

        # Compute shifts for softplus function
        xshift = tf.math.log(tf.math.exp(default_width) - 1)
        yshift = tf.math.log(tf.math.exp(default_height) - 1)

        # Constrain widths and heights to be positive
        widths = tf.math.softplus(widths + xshift)
        heights = tf.math.softplus(heights + yshift)

        # Compute spline derivatives
        shift = tf.math.log(EULER_CONST - 1.0)
        derivatives = tf.nn.softplus(derivatives + shift)

        # Add in edge derivatives
        total_height = tf.reduce_sum(heights, axis=-1, keepdims=True)
        total_width = tf.reduce_sum(widths, axis=-1, keepdims=True)
        scale = total_height / total_width
        derivatives = tf.concat([scale, derivatives, scale], axis=-1)
        return left_edge, bottom_edge, widths, heights, derivatives


class CouplingLayer(tf.keras.Model):
    """General wrapper for a coupling layer with different settings."""

    def __init__(  # noqa: PLR0913
        self,
        latent_dim: int,
        coupling_settings: Optional[dict[str, Any]] = None,
        coupling_design: str | Callable = "affine",
        permutation: Optional[str] = "fixed",
        use_act_norm: bool = True,
        act_norm_init: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Create an invertible coupling layers instance

        Parameters
        ----------
        latent_dim
            The dimensionality of the latent space (equal to the
            dimensionality of the target variable)
        coupling_settings
            The coupling network settings to pass to the internal
            coupling layers. See ``default_settings``
            for the required entries.
        coupling_design
            The type of internal coupling network to use.
            Must be in ['affine', 'spline'].
            In general, spline couplings run slower than affine
            couplings, but requires fewer coupling
            layers. Spline couplings may work best with complex
            (e.g., multimodal) low-dimensional
            problems. The difference will become less and less pronounced
            as we move to higher dimensions.
        permutation
            Whether to use permutations between coupling layers.
            Highly recommended if ``num_coupling_layers > 1``
            Important: Must be in ['fixed', 'learnable', None]
        use_act_norm
            Whether to use activation normalization after each
            coupling layer. Recommended to keep default.
        act_norm_init
            Optional data-dependent initialization for the
            internal ``ActNorm`` layers.
        **kwargs
            Optional keyword arguments (e.g., name) passed to
            the tf.keras.Model __init__ method.
        """
        super().__init__(**kwargs)

        # Set dimensionality attributes
        self.latent_dim = latent_dim
        self.dim_out1 = self.latent_dim // 2
        self.dim_out2 = (
            self.latent_dim // 2
            if self.latent_dim % 2 == 0
            else self.latent_dim // 2 + 1
        )

        # Determine coupling net settings
        if coupling_settings is None:
            user_dict = dict()
        elif isinstance(coupling_settings, dict):
            user_dict = coupling_settings
        else:
            raise ValueError(  # noqa: TRY003
                "coupling_net_settings argument must be None or a dict!"
            )

        # Determine type of coupling (affine or spline) and build settings
        if coupling_design == "affine":
            coupling_type = AffineCoupling
            coupling_settings = build_meta_dict(
                user_dict=user_dict, default_setting=DEFAULT_SETTING_AFFINE_COUPLING
            )
        elif coupling_design == "spline":
            coupling_type = SplineCoupling
            coupling_settings = build_meta_dict(
                user_dict=user_dict, default_setting=DEFAULT_SETTING_SPLINE_COUPLING
            )
        else:
            raise NotImplementedError('coupling_design must be in ["affine", "spline"]')

        # Two-in-one coupling block (i.e., no inactive part after a forward pass)
        self.net1 = coupling_type(self.dim_out1, coupling_settings)
        self.net2 = coupling_type(self.dim_out2, coupling_settings)

        # Optional (learnable or fixed) permutation
        if permutation not in ["fixed", "learnable", None]:
            raise ValueError(  # noqa: TRY003
                'Argument permutation should be in ["fixed", "learnable", None]'
            )
        if permutation == "fixed":
            self.permutation = Permutation(self.latent_dim)
            self.permutation.trainable = False
        elif permutation == "learnable":
            self.permutation = Orthogonal(self.latent_dim)
        else:
            self.permutation = None

        # Optional learnable activation normalization
        if use_act_norm:
            self.act_norm = ActNorm(latent_dim, act_norm_init)
        else:
            self.act_norm = None

    def call(
        self,
        target_or_z: tf.Tensor,
        condition: Optional[tf.Tensor],
        inverse: bool = False,
        **kwargs,
    ) -> tuple[Any, tf.Tensor]:
        r"""Perform one pass through the affine coupling layer.

        Parameters
        ----------
        target_or_z
            The estimation quantities of interest or latent
            representations z ~ p(z), shape (batch_size, ...)
        condition
            The conditioning data of interest, for instance,
            x = summary_fun(x), shape (batch_size, ...).
            If `condition is None`, then the layer reduces to
            an unconditional ACL.
        inverse
            Flag indicating whether to run the block forward or
            backward.

        Returns
        -------
        (z, log_det_J)  :
            If inverse=False: The transformed input and the corresponding
            Jacobian of the transformation,
            z shape: (batch_size, inp_dim),
            log_det_J shape: (batch_size, )

        target          :
            If inverse=True: The back-transformed z,
            shape (batch_size, inp_dim)

        Notes
        -----
        If ``inverse=False``, the return is ``(z, log_det_J)``.\n
        If ``inverse=True``, the return is ``target``
        """
        if not inverse:
            return self.forward(target_or_z, condition, **kwargs)
        return self.inverse(target_or_z, condition, **kwargs)

    def forward(
        self, target: tf.Tensor, condition: Optional[tf.Tensor], **kwargs
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Perform a forward pass through a coupling layer

        Use an optinal `Permutation` and `ActNorm` layer.

        Parameters
        ----------
        target
            The estimation quantities of interest, for instance,
            parameter vector of shape (batch_size, theta_dim)
        condition
            The conditioning vector of interest, for instance,
            x = summary(x), shape (batch_size, summary_dim)
            If `None`, transformation amounts to unconditional
            estimation.

        Returns
        -------
        (z, log_det_J)  :
            The transformed input and the corresponding Jacobian
            of the transformation.
        """
        # Initialize log_det_Js accumulator
        log_det_Js = tf.zeros(1)

        # Normalize activation, if specified
        if self.act_norm is not None:
            target, log_det_J_act = self.act_norm(target)
            log_det_Js += log_det_J_act

        # Permute, if indicated
        if self.permutation is not None:
            target = self.permutation(target)
        if self.permutation.trainable:
            target, log_det_J_p = target
            log_det_Js += log_det_J_p

        # Pass through coupling layer
        latent, log_det_J_c = self._forward(target, condition, **kwargs)
        log_det_Js += log_det_J_c
        return latent, log_det_Js

    def inverse(
        self, latent: tf.Tensor, condition: Optional[tf.Tensor], **kwargs
    ) -> tf.Tensor:
        """Perform an inverse pass through a coupling layer

        Use an optional `Permutation` and `ActNorm` layer.

        Parameters
        ----------
        z
            latent variables z ~ p(z),
            shape (batch_size, theta_dim)
        condition
            The conditioning vector of interest, for instance,
            x = summary(x), shape (batch_size, summary_dim).
            If `None`, transformation amounts to unconditional
            estimation.

        Returns
        -------
        target  :
            The back-transformed latent variable z.
        """
        target = self._inverse(latent, condition, **kwargs)
        if self.permutation is not None:
            target = self.permutation(target, inverse=True)
        if self.act_norm is not None:
            target = self.act_norm(target, inverse=True)
        return target

    def _forward(self, target, condition, **kwargs):
        """Perform a forward pass through the coupling layer.

        Used internally by the instance.

        Parameters
        ----------
        target     : tf.Tensor
            The estimation quantities of interest, for instance,
            parameter vector of shape (batch_size, theta_dim)
        condition  : tf.Tensor or None
            The conditioning vector of interest, for instance,
            x = summary(x), shape (batch_size, summary_dim)
            If `None`, transformation amounts to unconditional
            estimation.

        Returns
        -------
        (v, log_det_J)  :  tuple(tf.Tensor, tf.Tensor)
            The transformed input and the corresponding
            Jacobian of the transformation.
        """
        # Split input along last axis and perform forward coupling
        u1, u2 = tf.split(target, [self.dim_out1, self.dim_out2], axis=-1)
        v1, log_det_J1 = self.net1(u1, u2, condition, inverse=False, **kwargs)
        v2, log_det_J2 = self.net2(u2, v1, condition, inverse=False, **kwargs)
        v = tf.concat((v1, v2), axis=-1)

        # Compute log determinat of the Jacobians from both splits
        log_det_J = log_det_J1 + log_det_J2
        return v, log_det_J

    def _inverse(self, latent, condition, **kwargs):
        """Perform an inverse pass through the coupling block.

        Used internally by the instance.

        Parameters
        ----------
        latent       : tf.Tensor
            latent variables z ~ p(z), shape (batch_size, theta_dim)
        condition    : tf.Tensor or None
            The conditioning vector of interest, for instance,
            x = summary(x), shape (batch_size, summary_dim).
            If `None`, transformation amounts to unconditional
            estimation.

        Returns
        -------
        u  :  tf.Tensor
            The back-transformed input.
        """
        # Split input along last axis and perform inverse coupling
        v1, v2 = tf.split(latent, [self.dim_out1, self.dim_out2], axis=-1)
        u2 = self.net2(v1, v2, condition, inverse=True, **kwargs)
        u1 = self.net1(u2, v1, condition, inverse=True, **kwargs)
        u = tf.concat((u1, u2), axis=-1)
        return u


def merge_left_into_right(left_dict, right_dict):
    """Merge nested dict `left_dict` into nested dict `right_dict`."""
    for k, v in left_dict.items():
        if isinstance(v, dict):
            if right_dict.get(k) is not None:
                right_dict[k] = merge_left_into_right(v, right_dict.get(k))
            else:
                right_dict[k] = v
        else:
            right_dict[k] = v
    return right_dict


def build_meta_dict(user_dict: dict, default_setting: MetaDictSetting) -> dict:
    """Integrate a user-defined dictionary into a default dictionary.

    Takes a user-defined dictionary and a default dictionary.

    #. Scan the `user_dict` for violations by unspecified
        mandatory fields.
    #. Merge `user_dict` entries into the `default_dict`.
        Considers nested dict structure.

    Parameters
    ----------
    user_dict       : dict
        The user's dictionary
    default_setting : MetaDictSetting
        The specified default setting with attributes:

        -  `meta_dict`: dictionary with default values.
        -  `mandatory_fields`: list(str) keys that need to be
            specified by the `user_dict`

    Returns
    -------
    merged_dict: dict
        Merged dictionary.
    """
    default_dict = copy.deepcopy(default_setting.meta_dict)
    mandatory_fields = copy.deepcopy(default_setting.mandatory_fields)

    # Check if all mandatory fields are provided by the user
    if not all([field in user_dict.keys() for field in mandatory_fields]):
        raise ValueError(
            "Not all mandatory fields provided! Need at least"
            + " the following: {mandatory_fields}"
        )

    # Merge the user dict into the default dict
    merged_dict = merge_left_into_right(user_dict, default_dict)
    return merged_dict
