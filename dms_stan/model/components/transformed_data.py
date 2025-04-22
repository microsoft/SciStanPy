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
        super().__init__(_override_shape_check=True, **kwargs)

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


class SharedAlphaDirichlet(TransformedData):
    """
    Used to define a transformed data component for use with the Dirichlet distribution
    when all dimensions share the same alpha parameter.
    """

    def __init__(self, alpha: "dms_components.Constant", **kwargs):
        """
        Initializes the SharedAlphaDirichlet class.

        Args:
            alpha: The alpha parameter to use for the Dirichlet distribution.
            **kwargs: Additional arguments to pass to the parent class.
        """
        # The alpha must be a constant
        if not isinstance(alpha, dms_components.Constant):
            raise ValueError("The alpha parameter must be a constant.")

        # Initialize the parent class
        super().__init__(alpha=alpha, **kwargs)

    def get_supporting_functions(self) -> list[str]:
        """We need the Dirichlet function."""
        return ["#include dirichlet.stanfunctions"]

    def _write_operation(self, alpha: str) -> str:  # pylint: disable=arguments-differ
        """Writes the Stan code for the transformed alpha parameter."""
        # We just want the sum of alphas
        return f"sum({alpha})"

    @property
    def model_varname(self) -> str:
        return f"{self.alpha.model_varname}.sum"

    @property
    def parallelized(self) -> bool:
        """Returns whether the operation is parallelized."""
        return self.alpha.parallelized


class MultinomialCoefficient(TransformedData):
    """
    When the multinomial or multinomial_logit functions are used to model an
    observable, we can pre-calculate the multinomial coefficient and use it at each
    iteration. This speeds up the model.
    """

    def __init__(self, counts: "dms_components.Parameter", **kwargs):
        """
        Initializes the MultinomialCoefficient class.

        Args:
            counts: The counts parameter to use for the multinomial coefficient.
            **kwargs: Additional arguments to pass to the parent class.
        """
        # The counts must be an observable
        if not counts.observable:
            raise ValueError(
                "We can only pre-calculate the multinomial coefficient for constant counts."
            )

        # Initialize the parent class
        super().__init__(counts=counts, **kwargs)

    def get_supporting_functions(self) -> list[str]:
        """We need the multinomial coefficient function."""
        return ["#include multinomial.stanfunctions"]

    def _write_operation(self, counts: str) -> str:  # pylint: disable=arguments-differ
        """Writes the Stan code for the multinomial coefficient.

        Args:
            counts (str): String representation of the counts parameter.
        """
        # If parallelized, use that version
        prefix = "" if self.parallelized else "un"
        return f"{prefix}parallelized_multinomial_factorial_component_lpmf({counts})"

    @property
    def model_varname(self) -> str:
        """Returns the model variable name for the multinomial coefficient."""
        return f"{self.counts.model_varname}.multinomial_coefficient"

    @property
    def parallelized(self) -> bool:
        """Returns whether the operation is parallelized."""
        return self.counts.parallelized
