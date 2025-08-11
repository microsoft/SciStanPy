"""
Holds CDFs for the Parameter classes. These are special TransformedParameter
classes.
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
    """
    Base class for cumulative distribution-related function (CDFs) of parameters.
    Note that this cannot be instantiated directly, but is used as a template for
    the `build_cdf_class` function to create classes for each Parameter type.
    """

    # Class variables for each CDF
    PARAMETER: "parameters.Parameter"
    SCIPY_FUNC: str  # cdf, sf, log_cdf, log_sf
    STAN_SUFFIX: str  # The suffix for the Stan operation, e.g., "cdf"

    # Init function makes sure we have the correct parameters before initializing
    def __init__(
        self,
        x: "custom_types.CombinableParameterType",
        **params: "custom_types.CombinableParameterType",
    ):

        # Check if the parameters passed are the ones required for the CDF
        self.check_parameters(set(params.keys()))

        super().__init__(x=x, **params)

    def check_parameters(self, kwargset: set[str]) -> None:
        """Checks if the parameters passed are the ones required for the CDF."""
        # Make sure that these are the only parameters passed
        if additional_params := kwargset - self.PARAMETER.STAN_TO_SCIPY_NAMES.keys():
            raise TypeError(
                f"Unexpected parameters {additional_params} passed to "
                f"{self.__class__.__name__}."
            )
        if missing_params := self.PARAMETER.STAN_TO_SCIPY_NAMES.keys() - kwargset:
            raise TypeError(
                f"Missing parameters {missing_params} for {self.__class__.__name__}."
            )

    # The below returns the appropriate CDF function based on the class
    @abstractmethod
    def run_np_torch_op(self, **draws):

        # Get the module for the CDF function
        module = utils.choose_module(draws.values()[0])

        # If numpy use scipy dist. If torch, use torch dist. Torch will always
        # return the CDF, so child classes need to override this method.
        if module is np:
            return getattr(self.PARAMETER.SCIPY_DIST, self.SCIPY_FUNC)(**draws)

        # Torch separates distribution creation and function operation, so we need
        # to split out the 'x' value from the draws.
        elif module is torch:
            draws_copy = draws.copy()
            val = draws_copy.pop("x")
            return getattr(self.PARAMETER.TORCH_DIST, "cdf")(**draws_copy)(val)
        else:
            raise TypeError(
                f"Unsupported module {module} for CDF operation. "
                "Expected numpy or torch."
            )

    def write_stan_operation(self, **kwargs) -> str:

        # Get the function and arguments for the operation
        func = f"{self.PARAMETER.STAN_DIST}_{self.STAN_SUFFIX}"
        args = ", ".join(kwargs[name] for name in self.PARAMETER.STAN_TO_SCIPY_NAMES)

        return f"{func}({kwargs['x']} | {args})"


class CDF(CDFLike):
    """Represents the cumulative distribution function (CDF) of a parameter."""

    SCIPY_FUNC = "cdf"  # The SciPy function for the CDF
    STAN_SUFFIX = "cdf"  # The suffix for the Stan operation

    def run_np_torch_op(self, **draws):  # pylint: disable=useless-parent-delegation
        # Run using the function returned by the parent method.
        return super().run_np_torch_op(**draws)


class SurvivalFunction(CDFLike):
    """Represents the survival function (CCDF) of a parameter."""

    SCIPY_FUNC = "sf"  # The SciPy function for the survival function
    STAN_SUFFIX = "ccdf"  # The suffix for the Stan operation

    def run_np_torch_op(self, **draws):

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
    """Represents the log cumulative distribution function (log CDF) of a parameter."""

    SCIPY_FUNC = "logcdf"  # The SciPy function for the log CDF
    STAN_SUFFIX = "lcdf"  # The suffix for the Stan operation

    def run_np_torch_op(self, **draws):

        # As above, get the output of the parent method and return it directly
        # if using numpy.
        output = super().run_np_torch_op(**draws)
        if isinstance(output, np.ndarray):
            return output

        # If using torch, return the log CDF
        elif isinstance(output, torch.Tensor):
            return torch.log(output)

        # If using an unsupported type, raise an error
        else:
            raise TypeError(
                f"Unsupported module {type(output)} for log CDF operation. "
                "Expected numpy or torch."
            )


class LogSurvivalFunction(CDFLike):
    """Represents the log survival function (log CCDF) of a parameter."""

    SCIPY_FUNC = "logsf"  # The SciPy function for the log survival function
    STAN_SUFFIX = "lccdf"  # The suffix for the Stan operation

    def run_np_torch_op(self, **draws):

        # Get the output of the parent method
        output = super().run_np_torch_op(**draws)

        # If using numpy, return the log survival function directly
        if isinstance(output, np.ndarray):
            return output

        # If using torch, return the log of 1 minus the CDF
        elif isinstance(output, torch.Tensor):
            return torch.log(1 - output)

        else:
            raise TypeError(
                f"Unsupported module {type(output)} for log survival function operation. "
                "Expected numpy or torch."
            )
