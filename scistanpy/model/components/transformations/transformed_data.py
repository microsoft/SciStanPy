"""
Contains the TransformedData class, which is used to define the transformed data
block in a Stan model.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from scistanpy.model.components.transformations import transformed_parameters

if TYPE_CHECKING:
    from scistanpy.model.components import parameters


class TransformedData(transformed_parameters.Transformation):
    """Defines the `transformed data` block in a Stan model."""

    # The transformation is renamed to be more descriptive
    get_transformed_data_assignment = (
        transformed_parameters.Transformation._transformation  # pylint: disable=protected-access
    )

    # There is no implementation of `_draw` or `torch_parametrization` for this
    # class.
    def _draw(self, *args, **kwargs):
        """No implementation for drawing samples."""
        raise NotImplementedError("TransformedData does not implement _draw.")

    @property
    def torch_parametrization(self):
        """No implementation for torch parametrization."""
        raise NotImplementedError(
            "TransformedData does not implement torch_parametrization."
        )

    @property
    @abstractmethod
    def model_varname(self) -> str: ...


class LogMultinomialCoefficient(TransformedData):
    """
    When the multinomial distribution parametrized by log_theta is used to model
    an observable, we can pre-calculate the log of the multinomial coefficient and
    use it at each iteration. This speeds up the model.
    """

    SHAPE_CHECK = False

    def __init__(self, counts: "parameters.MultinomialLogTheta", **kwargs):
        """
        Initializes the LogMultinomialCoefficient class.

        Args:
            counts: The counts parameter to use for the log multinomial coefficient.
            **kwargs: Additional arguments to pass to the parent class.
        """
        # Initialize the parent class
        super().__init__(
            counts=counts,
            shape=counts.shape[:-1] + (1,),  # The last dimension is reduced to 1
            **kwargs,
        )

    def write_stan_operation(  # pylint: disable=arguments-differ
        self, counts: str
    ) -> str:
        """Writes the operation for the multinomial coefficient."""
        # The counts must be an observable
        if not self.counts.observable:
            raise ValueError(
                "We can only pre-calculate the multinomial coefficient for constant counts."
            )

        return f"log_multinomial_coeff({counts})"

    @property
    def model_varname(self) -> str:
        """Returns the model variable name for the multinomial coefficient."""
        return f"{self.counts.model_varname}.log_multinomial_coefficient"
