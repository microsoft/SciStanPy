"""
Defines the growth models for deep mutational scanning data that are used by DMS
Stan. These models are subclasses of the `dms_stan.model.Model` class. Inheriting
from them will automatically register the appropriate parameters and observables
for the model.
"""

import numpy as np
import numpy.typing as npt

import dms_stan.param as dmsp


class GrowthModel(dmsp.TransformedParameter):
    """Base class for growth models."""

    def __init__(
        self,
        *,
        t: dmsp.CombinableParameterType,
        shape: tuple[int, ...] = tuple(),
        **params: dmsp.CombinableParameterType,
    ):
        # Store all parameters as a list by calling the super class
        super().__init__(t=t, shape=shape, **params)


class ExponentialGrowth(GrowthModel):
    """
    A distribution that describes the exponential growth model. Specifically, parameters
    `t`, `A`, and `r` are used to calculate the exponential growth model as follows:

    $$
    x &= A \textrm{e}^{rt}
    $$
    """

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: dmsp.CombinableParameterType,
        A: dmsp.CombinableParameterType,
        r: dmsp.CombinableParameterType,
        shape: tuple[int, ...] = tuple(),
    ):
        """Initializes the ExponentialGrowth distribution.

        Args:
            t (dmsp.CombinableParameterType): The time parameter.

            A (dmsp.CombinableParameterType): The amplitude parameter.

            r (dmsp.CombinableParameterType): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(t=t, A=A, r=r, shape=shape)

    def operation(  # pylint: disable=arguments-differ
        self, *, t: dmsp.SampleType, A: dmsp.SampleType, r: dmsp.SampleType
    ) -> npt.NDArray:
        return A * np.exp(r * t)


class SigmoidGrowth(GrowthModel):
    r"""
    A distribution that describes the sigmoid growth model. Specifically, parameters
    `t`, `A`, `r`, and `c` are used to calculate the sigmoid growth model as follows:

    $$
    x &= \frac{A}{1 + \textrm{e}^{-r(t - c)}}
    $$
    """

    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        t: dmsp.CombinableParameterType,
        A: dmsp.CombinableParameterType,
        r: dmsp.CombinableParameterType,
        c: dmsp.CombinableParameterType,
        shape: tuple[int, ...] = tuple(),
    ):
        """Initializes the SigmoidGrowth distribution.

        Args:
            t (dmsp.CombinableParameterType): The time parameter.

            A (dmsp.CombinableParameterType): The amplitude parameter.

            r (dmsp.CombinableParameterType): The rate parameter.

            c (dmsp.CombinableParameterType): The offset parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(t=t, A=A, r=r, c=c, shape=shape)

    def operation(  # pylint: disable=arguments-differ
        self,
        *,
        t: dmsp.SampleType,
        A: dmsp.SampleType,
        r: dmsp.SampleType,
        c: dmsp.SampleType,
    ) -> npt.NDArray:
        return A / (1 + np.exp(-r * (t - c)))


class LogExponentialGrowth(GrowthModel):
    """
    A distribution that models the natural log of the `ExponentialGrowth` distribution.
    Specifically, parameters `t`, `log_A`, and `r` are used to calculate the log
    of the exponential growth model as follows:

    $$
    log(x) = log_A + rt
    $$

    Note that, with this parametrization, we guarantee that $x > 0$. It is also
    only defined for $A > 0$ and $r > 0$, assuming that the time parameter $t$ is
    always positive.

    This parametrization is particularly useful for modeling the proportions of
    different populations as is done in DMS Stan, as proportions are always positive.
    """

    def __init__(
        self,
        *,
        t: dmsp.SampleType,
        log_A: dmsp.SampleType,
        r: dmsp.SampleType,
        shape: tuple[int, ...] = tuple(),
    ):
        """Initializes the LogExponentialGrowth distribution.

        Args:
            t (dmsp.SampleType): The time parameter.

            log_A (dmsp.SampleType): The log of the amplitude parameter.

            r (dmsp.SampleType): The rate parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(t=t, log_A=log_A, r=r, shape=shape)

    def operation(  # pylint: disable=arguments-differ
        self,
        *,
        t: dmsp.SampleType,
        log_A: dmsp.SampleType,
        r: dmsp.SampleType,
    ) -> npt.NDArray:
        return log_A + r * t


class LogSigmoidGrowth(GrowthModel):
    r"""
    A distribution that models the natural log of the `SigmoidGrowth` distribution.
    Specifically, parameters `t`, `log_A`, `r`, and `c` are used to calculate the
    log of the sigmoid growth model as follows:

    $$
    log(x) = log_A - log(1 + \textrm{e}^{-r(t - c)})
    $$

    As with the `LogExponentialGrowth` distribution, this parametrization guarantees
    that $x > 0$.
    """

    def __init__(
        self,
        *,
        t: dmsp.SampleType,
        log_A: dmsp.SampleType,
        r: dmsp.SampleType,
        c: dmsp.SampleType,
        shape: tuple[int, ...] = tuple(),
    ):
        """Initializes the LogSigmoidGrowth distribution.

        Args:
            t (dmsp.SampleType): The time parameter.

            log_A (dmsp.SampleType): The log of the amplitude parameter.

            r (dmsp.SampleType): The rate parameter.

            c (dmsp.SampleType): The offset parameter.

            shape (tuple[int, ...], optional): The shape of the distribution. In
                most cases, this can be ignored as it will be calculated automatically.
        """
        super().__init__(t=t, log_A=log_A, r=r, c=c, shape=shape)

    def operation(  # pylint: disable=arguments-differ
        self,
        *,
        t: dmsp.SampleType,
        log_A: dmsp.SampleType,
        r: dmsp.SampleType,
        c: dmsp.SampleType,
    ) -> npt.NDArray:
        return log_A - np.log(1 + np.exp(-r * (t - c)))
