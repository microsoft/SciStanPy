"""
Defines the growth models for deep mutational scanning data that are used by DMS
Stan. These models are subclasses of the `dms_stan.model.Model` class. Inheriting
from them will automatically register the appropriate parameters and observables
for the model.
"""

from abc import abstractmethod
from typing import Union

import numpy as np
import numpy.typing as npt

import dms_stan.param as dmsp


class GrowthModel(dmsp.TransformedParameter):
    """Base class for growth models."""

    def __init__(
        self, *, t: dmsp.CombinableParameterType, **params: dmsp.CombinableParameterType
    ):

        # Store all parameters as a list by calling the super class
        super().__init__(t, *list(params.values()))

        # Time and other params are stored attributes
        self.t = t

        # Register other parameters as attributes
        for name, param in params.items():
            setattr(self, name, param)

        # All parameters are stored in a dictionary
        self._growth_params = {"t": t, **params}

    def draw(self, size: Union[int, tuple[int, ...]]) -> npt.NDArray:
        return self.operation(
            **{
                name: (
                    param.draw(size)
                    if isinstance(param, dmsp.AbstractParameter)
                    else param
                )
                for name, param in self._growth_params.items()
            }
        )

    @abstractmethod
    def operation(  # pylint: disable=arguments-differ
        self, **params: dmsp.CombinableParameterType
    ) -> npt.NDArray: ...


class ExponentialGrowth(GrowthModel):
    """An exponential growth model."""

    def __init__(  # pylint: disable=useless-parent-delegation
        self, *, t: dmsp.CombinableParameterType, r: dmsp.CombinableParameterType
    ):
        super().__init__(t=t, r=r)

    def operation(self, **params: dmsp.CombinableParameterType) -> npt.NDArray:
        return np.exp(params["t"] * params["r"])
