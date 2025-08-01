"""
Contains the TransformedData class, which is used to define the transformed data
block in a Stan model.
"""

from abc import abstractmethod

import dms_stan.model.components as dms_components

from .transformed_parameters import Transformation


class TransformedData(Transformation):
    """Defines the `transformed data` block in a Stan model."""

    # The transformation is renamed to be more descriptive
    get_transformed_data_assignment = Transformation._transformation

    def __init__(self, **kwargs):
        # We do not check shapes for transformed data
        super().__init__(**kwargs)

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

    def __init__(self, counts: "dms_components.MultinomialLogTheta", **kwargs):
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

    # We don't want to run the shape checking function for this class
    def _set_shape(self, *args, **kwargs):
        """No shape checking for LogMultinomialCoefficient."""

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
