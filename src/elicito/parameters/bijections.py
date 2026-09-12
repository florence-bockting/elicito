"""
Transforms between constrained and unconstrained hyperparameters
"""

from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore


def identity(x: float) -> Any:
    """
    Identity function. Returns the input

    Parameters
    ----------
    x
        Input x

    Returns
    -------
    x :
        input x without transformation.

    """
    return x


class DoubleBound:
    """
    constrain double-bounded distributions
    """

    def __init__(self, lower: float, upper: float):
        """
        Constrain double-bounded distribution

        A variable constrained to be in the open interval
        (``lower``, ``upper``) is transformed to an unconstrained variable Y
        via a scaled and translated log-odds transform.

        Basis for the here used constraints, is the
        `constraint transforms implementation in [Stan](https://mc-stan.org/docs/reference-manual/transforms.html).

        Parameters
        ----------
        lower
            Lower bound of variable x.

        upper
            Upper bound of variable x.

        """
        self.lower = lower
        self.upper = upper

    def logit(self, u: tf.Tensor) -> tf.Tensor:
        r"""
        Implement the logit transformation for :math:`u \in (0,1)`:

        .. math::

            logit(u) = \log\left(\frac{u}{1-u}\right)

        Parameters
        ----------
        u
            Variable in open unit interval.

        Returns
        -------
        v
            Log-odds of u.

        """
        # log-odds definition
        v = tf.math.log(u / (1 - u))
        # cast v into correct dtype
        v = tf.cast(v, dtype=tf.float32)
        return v

    def inv_logit(self, v: tf.Tensor) -> tf.Tensor:
        r"""
        Implement the inverse-logit transformation

        The inverse-logit transformation is the logistic
        sigmoid for :math:`v \in (-\infty,+\infty)`:

        .. math::

            logit^{-1}(v) = \frac{1}{1+\exp(-v)}

        Parameters
        ----------
        v
            Unconstrained variable

        Returns
        -------
        u
            Logistic sigmoid of the unconstrained variable

        """
        # logistic sigmoid transform
        u = tf.divide(1.0, (1.0 + tf.exp(-v)))
        # cast v to correct dtype
        u = tf.cast(u, dtype=tf.float32)
        return u

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        r"""
        Scale and translate logit transformed variable

        transform variable x with ``lower`` and ``upper`` bound
        into an unconstrained variable y.

        .. math::

            Y = logit\left(\frac{X - lower}{upper - lower}\right)

        Parameters
        ----------
        x
            Variable with lower and upper bound.

        Returns
        -------
        y
            Unconstrained variable.

        """
        # scaled and translated logit transform
        y = self.logit(tf.divide((x - self.lower), (self.upper - self.lower)))
        # cast y to correct dtype
        y = tf.cast(y, dtype=tf.float32)
        return y

    def inverse(self, y: tf.Tensor) -> tf.Tensor:
        r"""
        Apply inverse of the log-odds transform

        unconstrained variable y is transformed into a constrained variable x
        with ``lower`` and ``upper`` bound.

        .. math::

            X = lower + (upper - lower) \cdot logit^{-1}(Y)

        Parameters
        ----------
        y
            Unconstrained variable

        Returns
        -------
        x :
            Constrained variable with lower and upper bound

        """
        # inverse of log-odds transform
        x = self.lower + (self.upper - self.lower) * self.inv_logit(y)
        # cast x to correct dtype
        x = tf.cast(x, dtype=tf.float32)
        return x


class LowerBound:
    """
    constrain lower-bounded distributions
    """

    def __init__(self, lower: float):
        """
        Transform ``lower`` bound variable to unconstrained variable Y

        use inverse-softplus transform.

        References
        ----------
        - [Stan](https://mc-stan.org/docs/reference-manual/transforms.html)

        Parameters
        ----------
        lower
            Lower bound of variable X.

        """
        self.lower = lower

    def forward(self, x: float) -> Any:
        r"""
        Transform ``lower``-bounded x via inverse-softplus into an unconstrained y.

        .. math::

            Y = softplus^{-1}(X - lower)

        Parameters
        ----------
        x
            Variable with a lower bound.

        Returns
        -------
        y :
            Unconstrained variable.

        """
        # inverse softplus transform
        y = tfp.math.softplus_inverse(x - self.lower)
        # cast y into correct type
        y = tf.cast(y, dtype=tf.float32)
        return y

    def inverse(self, y: float) -> tf.Tensor:
        r"""
        Apply softplus to unconstrained y to get ``lower``-bounded x

        .. math::

            X = softplus(Y) + lower

        Parameters
        ----------
        y
            Unconstrained variable.

        Returns
        -------
        x :
            Variable with a lower bound.

        """
        # softplus transform
        x = tf.math.softplus(y) + self.lower
        # cast x into correct dtype
        x = tf.cast(x, dtype=tf.float32)
        return x


class UpperBound:
    """
    transform ``upper`` bounded distribution
    """

    def __init__(self, upper: float):
        """
        Transform ``upper`` bounded x into unconstrained y

        use inverse-softplus transform.

        Parameters
        ----------
        upper
            Upper bound of variable X.

        References
        ----------
        + [Stan](https://mc-stan.org/docs/reference-manual/transforms.html)

        """
        self.upper = upper

    def forward(self, x: float) -> Any:
        r"""
        Transform upper-bouned into unconstarined variable

        use inverse-softplus transform

        .. math::

            Y = softplus^{-1}(upper - X)

        Parameters
        ----------
        x
            Variable with an upper bound.

        Returns
        -------
        y :
            Unconstrained variable.

        """
        # logarithmic transform
        y = tfp.math.softplus_inverse(self.upper - x)
        # cast y into correct dtype
        y = tf.cast(y, dtype=tf.float32)
        return y

    def inverse(self, y: float) -> tf.Tensor:
        r"""
        Transform uncstrained into lower-bounded variable

        use softplus transform

        .. math::

            X = upper - softplus(Y)

        Parameters
        ----------
        y
            Unconstrained variable.

        Returns
        -------
        x :
            Variable with an upper bound.

        """
        # exponential transform
        x = self.upper - tf.math.softplus(y)
        # cast x into correct dtype
        x = tf.cast(x, dtype=tf.float32)
        return x
