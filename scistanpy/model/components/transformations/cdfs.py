# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Cumulative distribution function transformations for SciStanPy parameters.

This module provides specialized transformation classes for computing cumulative
distribution functions (CDFs) and related probability functions for SciStanPy
model parameters.

The module implements a unified interface for CDF-like computations across
multiple computational backends (NumPy/SciPy, PyTorch) while automatically
generating appropriate Stan code for each transformation.

CDF-like Transformation Types:
    - :py:class:`~scistanpy.model.components.transformations.cdfs.CDF`
    - :py:class:`~scistanpy.model.components.transformations.cdfs.SurvivalFunction`
    - :py:class:`~scistanpy.model.components.transformations.cdfs.LogCDF`
    - :py:class:`~scistanpy.model.components.transformations.cdfs.LogSurvivalFunction`

Each CDF class automatically handles backend-specific implementations:
    - **NumPy/SciPy**: Uses SciPy distribution methods with parameter transforms as needed
    - **PyTorch**: Uses PyTorch distribution objects with appropriate methods
    - **Stan**: Generates function calls with proper parameter ordering

The classes are not intended to be accessed directly. Instead, they are used as
templates by the :py:class:`~scistanpy.model.components.parameters.ParameterMeta`
metaclass to build :py:class:`~scistanpy.model.components.parameters.ParameterMeta`
-specific classes on module import, which are assigned to the ``CDF``, ``LOG_CDF``,
``SF``, and ``LOG_SF`` properties of each :py:class:`~scistanpy.model.components.
parameters.Parameter`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch

from scistanpy import utils
from scistanpy.model.components.transformations import transformed_parameters

if TYPE_CHECKING:
    from scistanpy import custom_types
    from scistanpy.model.components import parameters


class CDFLike(transformed_parameters.TransformedParameter):
    """Base class for cumulative distribution function transformations.

    This abstract base class provides the common infrastructure for all CDF-like
    transformations including parameter validation, backend selection, and
    Stan code generation. It cannot be instantiated directly but serves as the
    foundation for specific CDF transformation types.

    :param x: Input values for CDF evaluation
    :type x: custom_types.CombinableParameterType
    :param shape: Shape of the transformation output. Defaults to ().
    :type shape: Union[tuple[custom_types.Integer, ...], custom_types.Integer]
    :param params: Distribution parameters required for the CDF computation
    :type params: custom_types.CombinableParameterType

    :raises TypeError: If unexpected or missing parameters are provided

    The class provides a unified interface for computing probability functions
    across different computational backends while maintaining compatibility
    with the SciStanPy model component system.

    Key Responsibilities:
        - Parameter validation against expected parameter sets
        - Backend detection and appropriate method dispatch
        - Parameter transformation for SciPy compatibility
        - Stan code generation for probability function calls

    The class automatically handles the complexities of:
        - Converting between parameter naming conventions
        - Applying parameter transformations for different backends
        - Generating appropriate function calls for each backend
    """

    # Class variables for each CDF
    PARAMETER: "parameters.Parameter"
    """
    Reference to the :py:class:`~scistanpy.model.components.parameters.Parameter`
    subclass for which this ``CDFLike`` class applies. Should be set by the metaclass.
    """

    SCIPY_FUNC: str  # cdf, sf, log_cdf, log_sf
    """
    Name of the SciPy method for this operation (e.g., 'cdf', 'sf', 'log_cdf',
    'log_sf'). Should be set by subclasses.
    """

    TORCH_FUNC: str  # cdf, log_cdf, log_sf
    """
    Name of the PyTorch method for this operation (e.g., 'cdf', 'log_cdf', 'log_sf').
    Should be set by subclasses.
    """

    STAN_SUFFIX: str  # The suffix for the Stan operation, e.g., "cdf"
    """
    Suffix for Stan function name generation (e.g., "cdf"). Should be set by subclasses.
    """

    def __init__(
        self,
        x: "custom_types.CombinableParameterType",
        shape: tuple[custom_types.Integer, ...] | custom_types.Integer = (),
        **params: "custom_types.CombinableParameterType",
    ):
        """Initialize CDF transformation with parameter validation.

        :param x: Input values for CDF evaluation
        :type x: custom_types.CombinableParameterType
        :param shape: Shape of the transformation output
        :type shape: Union[tuple[custom_types.Integer, ...], custom_types.Integer]
        :param params: Distribution parameters for the CDF computation
        :type params: custom_types.CombinableParameterType

        :raises TypeError: If parameters don't match those required by the target distribution

        The initialization process validates that all required distribution
        parameters are provided and no unexpected parameters are included.
        """
        # Check if the parameters passed are the ones required for the CDF
        self.check_parameters(set(params.keys()))

        super().__init__(x=x, **params)

    def check_parameters(self, kwargset: set[str]) -> None:
        """Validate that provided parameters match distribution requirements.

        :param kwargset: Set of parameter names provided
        :type kwargset: set[str]

        :raises TypeError: If unexpected parameters are provided
        :raises TypeError: If required parameters are missing

        This method ensures that the CDF transformation receives exactly the
        parameters required by the underlying probability distribution, with
        no additional or missing parameters.
        """
        # Make sure that these are the only parameters passed
        if (
            additional_params := kwargset
            - self.__class__.PARAMETER.STAN_TO_SCIPY_NAMES.keys()
        ):
            raise TypeError(
                f"Unexpected parameters {additional_params} passed to "
                f"{self.__class__.__name__}."
            )
        if (
            missing_params := self.__class__.PARAMETER.STAN_TO_SCIPY_NAMES.keys()
            - kwargset
        ):
            raise TypeError(
                f"Missing parameters {missing_params} for {self.__class__.__name__}."
            )

    @abstractmethod
    def run_np_torch_op(self, **draws):
        """Execute the CDF-like operation using NumPy or PyTorch backend as appropriate.

        :param draws: Dictionary of parameter draws for the operation
        :type draws: dict

        :returns: CDFLike evaluation results
        :rtype: Union[np.ndarray, torch.Tensor]

        :raises TypeError: If unsupported module type is detected

        This abstract method implements the core computational logic for
        evaluating the CDFLike transformation. It automatically detects the
        computational backend and applies appropriate parameter transformations.

        Backend Handling:
            - **NumPy**: Uses SciPy distribution methods with parameter transforms
            - **PyTorch**: Creates distribution objects and calls appropriate methods
            - **Other**: Raises TypeError for unsupported backends

        The method separates the evaluation point (``x``) from distribution
        parameters and handles backend-specific parameter naming and
        transformation requirements.
        """
        # Get the module for the CDF function
        module = utils.choose_module(next(iter(draws.values())))

        # We need to separate the x value from the draws
        draws_copy = draws.copy()
        x = draws_copy.pop("x")

        # If numpy use scipy dist. If torch, use torch dist. Torch will always
        # return the CDF, so child classes need to override this method.
        if module is np:
            kwargs = {
                self.__class__.PARAMETER.STAN_TO_SCIPY_NAMES[
                    name
                ]: self.__class__.PARAMETER.STAN_TO_SCIPY_TRANSFORMS.get(
                    name, lambda x: x
                )(
                    draw
                )
                for name, draw in draws_copy.items()
            }
            return getattr(self.__class__.PARAMETER.SCIPY_DIST, self.SCIPY_FUNC)(
                x, **kwargs
            )

        # Torch separates distribution creation and function operation, so we need
        # to split out the 'x' value from the draws.
        elif module is torch:

            # Build the distribution
            dist = self.__class__.PARAMETER.TORCH_DIST(
                **{
                    self.__class__.PARAMETER.STAN_TO_TORCH_NAMES[name]: draw
                    for name, draw in draws_copy.items()
                }
            )

            # Run the appropriate function. Some torch dists have custom functions
            # that explicitly calculate the target value. Others extend the CDF.
            return getattr(
                dist,
                (
                    self.__class__.TORCH_FUNC
                    if hasattr(dist, self.__class__.TORCH_FUNC)
                    else "cdf"
                ),
            )(x)
        else:
            raise TypeError(
                f"Unsupported module {module} for CDF operation. "
                "Expected numpy or torch."
            )

    def write_stan_operation(self, **kwargs) -> str:
        """Generate Stan code for the ``CDFLike`` operation.

        :param kwargs: Formatted parameter strings for Stan code generation
        :type kwargs: dict[str, str]

        :returns: Stan function call string
        :rtype: str

        This method constructs the appropriate Stan function call for the
        CDF operation, using the distribution name, operation suffix, and
        properly ordered parameters.

        The generated Stan code follows the pattern:
        distribution_suffix(x | param1, param2, ...)

        Where the parameters are ordered according to the distribution's
        Stan parameter ordering conventions.
        """
        # Get the function and arguments for the operation
        func = f"{self.__class__.PARAMETER.STAN_DIST}_{self.STAN_SUFFIX}"
        args = ", ".join(
            kwargs[name] for name in self.__class__.PARAMETER.STAN_TO_SCIPY_NAMES
        )

        return f"{func}({kwargs['x']} | {args})"


class CDF(CDFLike):
    r"""Standard cumulative distribution function transformation.

    Computes :math:`P(X \leq x)` for a given distribution and evaluation point.

    :param x: Values at which to evaluate the CDF
    :type x: custom_types.CombinableParameterType
    :param shape: Shape of the output. Defaults to ().
    :type shape: Union[tuple[custom_types.Integer, ...], custom_types.Integer]
    :param params: Distribution parameters
    :type params: custom_types.CombinableParameterType

    Mathematical Definition:
        .. math::
            F(X) = P(X \leq x) = \int_{-\infty}^{x} f(t) dt

    Where :math:`f(t)` is the probability density function of the distribution.

    Common Applications:
        - Computing tail probabilities
        - Implementing truncated distributions
        - Calculating quantiles and percentiles
        - Model validation through probability plots

    Example:
        >>> # Via parameter instance (typical usage)
        >>> normal_param = Normal(mu=0.0, sigma=1.0)
        >>> cdf_transform = normal_param.cdf(x=data_points)
        >>>
        >>> # Direct instantiation (less common)
        >>> cdf_transform = Normal.CDF(x=values, mu=0.0, sigma=1.0)
    """

    SCIPY_FUNC = "cdf"  # The SciPy function for the CDF
    STAN_SUFFIX = "cdf"  # The suffix for the Stan operation
    TORCH_FUNC = "cdf"

    def run_np_torch_op(self, **draws):  # pylint: disable=useless-parent-delegation
        """Execute CDF computation using appropriate backend.

        :param draws: Parameter draws for the computation
        :type draws: dict

        :returns: CDF values P(X ≤ x)
        :rtype: Union[np.ndarray, torch.Tensor]

        This implementation uses the base class method directly as both
        NumPy and PyTorch provide direct CDF computation methods.
        """
        # Run using the function returned by the parent method.
        return super().run_np_torch_op(**draws)


class SurvivalFunction(CDFLike):
    r"""Survival function (complementary CDF) transformation.

    Computes :math:`P(X \gt x) = 1 - P(X \leq x)` for a given distribution and
    evaluation point.

    :param x: Values at which to evaluate the survival function
    :type x: custom_types.CombinableParameterType
    :param shape: Shape of the output. Defaults to ().
    :type shape: Union[tuple[custom_types.Integer, ...], custom_types.Integer]
    :param params: Distribution parameters
    :type params: custom_types.CombinableParameterType

    Mathematical Definition:
        .. math::
            S(x) = P(X \gt x) = 1 - F(x) = \int_{x}^{\infty} f(t) dt

    Where :math:`F(x)` is the CDF and :math:`f(t)` is the probability density function.

    Common Applications:
        - Survival analysis and time-to-event modeling
        - Reliability engineering and failure analysis
        - Risk assessment and hazard modeling
        - Complementary probability calculations

    The implementation automatically handles backend differences:
        - NumPy: Uses SciPy's direct survival function methods
        - PyTorch: Computes 1 - CDF

    Example:
        >>> # Survival analysis
        >>> survival_times = Exponential(rate=0.1)
        >>> survival_prob = survival_times.ccdf(x=time_points)
    """

    SCIPY_FUNC = "sf"  # The SciPy function for the survival function
    STAN_SUFFIX = "ccdf"  # The suffix for the Stan operation
    TORCH_FUNC = "cdf"

    def run_np_torch_op(self, **draws):
        r"""Execute survival function computation with backend-specific handling.

        :param draws: Parameter draws for the computation
        :type draws: dict

        :returns: Survival function values :math:`P(X \gt x)`
        :rtype: Union[np.ndarray, torch.Tensor]

        :raises TypeError: If unsupported output type is encountered

        This method handles the difference between NumPy and PyTorch:
            - NumPy: SciPy provides direct survival function methods
            - PyTorch: Computes 1 - CDF
        """
        # Get the output of the parent method
        output = super().run_np_torch_op(**draws)

        # If using numpy, just return
        if isinstance(output, np.ndarray):
            return output

        # If using torch, subtract from 1 to get the survival function
        elif isinstance(output, torch.Tensor):
            return 1 - output

        else:
            raise TypeError(
                f"Unsupported module {type(output)} for survival function operation. "
                "Expected numpy or torch."
            )


class LogCDF(CDFLike):
    r"""Logarithmic cumulative distribution function transformation.

    Computes :math:`\log(P(X \leq x)) = \log(F(x))` for numerical stability when
    dealing with very small probabilities. This is essential for
    computations involving extreme tail probabilities.

    :param x: Values at which to evaluate the log CDF
    :type x: custom_types.CombinableParameterType
    :param shape: Shape of the output. Defaults to ().
    :type shape: Union[tuple[custom_types.Integer, ...], custom_types.Integer]
    :param params: Distribution parameters
    :type params: custom_types.CombinableParameterType

    Mathematical Definition:
        .. math::
            \log F(x) = \log(P(X \leq x))

    Numerical Advantages:
        - Prevents underflow for very small probabilities
        - Enables stable computation in log-space
        - Essential for extreme value analysis

    Common Applications:
        - Extreme value analysis and rare event modeling
        - Numerical optimization in log-space
        - MCMC sampling with extreme parameter values
        - Likelihood computations for tail events

    The implementation handles backend-specific log CDF methods:
        - NumPy: Uses SciPy's logcdf methods when available
        - PyTorch: Uses log_cdf methods or log(cdf) fallback

    Example:
        >>> # Extreme tail probability
        >>> normal_param = Normal(mu=0, sigma=1)
        >>> log_tail_prob = normal_param.log_cdf(x=extreme_values)
    """

    SCIPY_FUNC = "logcdf"  # The SciPy function for the log CDF
    STAN_SUFFIX = "lcdf"  # The suffix for the Stan operation
    TORCH_FUNC = "log_cdf"

    def run_np_torch_op(self, **draws):
        r"""Execute log CDF computation with appropriate numerical handling.

        :param draws: Parameter draws for the computation
        :type draws: dict

        :returns: Log CDF values :math:`\log(P(X \leq x))`
        :rtype: Union[np.ndarray, torch.Tensor]

        :raises TypeError: If unsupported output type is encountered
        """
        # As above, get the output of the parent method and return it directly
        # if using numpy.
        output = super().run_np_torch_op(**draws)
        if isinstance(output, np.ndarray):
            return output

        # If using torch, return the log CDF
        elif isinstance(output, torch.Tensor):
            if hasattr(self.__class__.PARAMETER.TORCH_DIST, self.__class__.TORCH_FUNC):
                return output
            return torch.log(output)

        # If using an unsupported type, raise an error
        else:
            raise TypeError(
                f"Unsupported module {type(output)} for log CDF operation. "
                "Expected numpy or torch."
            )


class LogSurvivalFunction(CDFLike):
    r"""Logarithmic survival function transformation.

    Computes :math:`\log(P(X > x)) = \log(1 - F(x))` for numerical stability
    when dealing with survival probabilities that may be very close
    to zero or one. Essential for stable survival analysis computations.

    :param x: Values at which to evaluate the log survival function
    :type x: custom_types.CombinableParameterType
    :param shape: Shape of the output. Defaults to ().
    :type shape: Union[tuple[custom_types.Integer, ...], custom_types.Integer]
    :param params: Distribution parameters
    :type params: custom_types.CombinableParameterType

    Mathematical Definition:
        .. math::
            \log S(x) = \log(P(X > x)) = \log(1 - F(x))

        Where :math:`F(x)` is the CDF.

    Numerical Advantages:
        - Prevents underflow for probabilities near 0 or 1
        - Maintains precision for extreme survival times
        - Enables stable log-space arithmetic
        - Critical for numerical stability in survival models

    Common Applications:
        - Survival analysis with extreme event times
        - Reliability engineering with high reliability systems
        - Hazard modeling with rare failure events
        - Log-likelihood computations for survival models

    The implementation provides numerically stable computation:
        - NumPy: Uses SciPy's logsf methods for direct computation
        - PyTorch: Uses log_sf methods or :math:`\text{log1p}(-cdf)` as fallback
    """

    SCIPY_FUNC = "logsf"  # The SciPy function for the log survival function
    STAN_SUFFIX = "lccdf"  # The suffix for the Stan operation
    TORCH_FUNC = "log_sf"

    def run_np_torch_op(self, **draws):
        r"""Execute log survival function computation with numerical stability.

        :param draws: Parameter draws for the computation
        :type draws: dict

        :returns: Log survival function values :math:`\log(P(X \gt x))`
        :rtype: Union[np.ndarray, torch.Tensor]

        :raises TypeError: If unsupported output type is encountered

        This method ensures numerical stability by:
            - Using native log survival function methods when available
            - Using :math:`\text{log1p}(-cdf)` for PyTorch when direct methods unavailable
            - Handling precision issues near probability boundaries
        """
        # Get the output of the parent method
        output = super().run_np_torch_op(**draws)

        # If using numpy, return the log survival function directly
        if isinstance(output, np.ndarray):
            return output

        # If using torch, return the log of 1 minus the CDF
        elif isinstance(output, torch.Tensor):
            if hasattr(self.__class__.PARAMETER.TORCH_DIST, self.__class__.TORCH_FUNC):
                return output
            return torch.log1p(-output)

        else:
            raise TypeError(
                f"Unsupported module {type(output)} for log survival function operation. "
                "Expected numpy or torch."
            )
