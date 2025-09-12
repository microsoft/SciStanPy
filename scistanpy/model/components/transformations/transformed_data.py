# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Transformed data components for Stan model generation.

This module provides specialized transformation classes for components that
belong in Stan's ``transformed data`` block. These transformations represent
deterministic computations that can be performed once at the beginning of
Stan program execution, before any sampling begins.

Transformed data components differ from regular transformations in that they:
    - Execute only once per Stan program run
    - Cannot depend on parameters (only on data)
    - Reduce per-iteration computational overhead in Stan models

The module integrates with SciStanPy's transformation system while providing
the specialized behavior required for Stan's ``transformed data`` block, enabling
significant performance improvements for models with expensive deterministic
computations.

These optimizations are particularly valuable for:
    - Complex likelihood functions with constant terms
    - Expensive matrix operations on fixed data
    - Normalization constants for custom distributions
    - Any deterministic computation independent of parameters

Transformed data components are never directly accessed by users. Instead, they
are used internally by certain :py:class:`~scistanpy.model.components.parameters.Parameter`
subclasses to optimize model performance. For example, the :py:class:`~scistanpy.
model.components.parameters.MultinomialLogTheta` class automatically adds a
:py:class:`LogMultinomialCoefficient` transformed data component to pre-compute
the multinomial coefficient when the counts are known data.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from scistanpy.model.components.transformations import transformed_parameters

if TYPE_CHECKING:
    from scistanpy.model.components import parameters


class TransformedData(transformed_parameters.Transformation):
    """Abstract base class for Stan transformed data block components.

    This class provides the foundation for model components that generate
    Stan code for the transformed data block. Components in this block
    represent deterministic computations that are performed once at the
    beginning of Stan execution, before any parameter sampling begins.

    Transformed data components must satisfy strict requirements:
        - Can only depend on data (not parameters)
        - Must be deterministic (no random components)
        - Execute exactly once per Stan program run
        - Cannot be sampled from or optimized

    The class inherits from Transformation but disables sampling and PyTorch
    optimization capabilities since transformed data components represent
    fixed computations rather than random variables or learnable parameters.

    Subclasses must implement:
        - model_varname property for Stan variable naming
        - write_stan_operation method for generating Stan code

    The class provides performance benefits by:
        - Pre-computing expensive deterministic functions
        - Reducing per-iteration computational overhead
        - Enabling Stan compiler optimizations
        - Avoiding redundant calculations during sampling
    """

    # The transformation is renamed to be more descriptive
    get_transformed_data_assignment = (
        transformed_parameters.Transformation._transformation  # pylint: disable=protected-access
    )

    def _draw(self, *args, **kwargs):
        """Raise error for sampling attempts on transformed data.

        :param args: Unused positional arguments
        :param kwargs: Unused keyword arguments

        :raises NotImplementedError: Always, as transformed data cannot be sampled

        Transformed data components represent deterministic computations that
        are performed once at Stan initialization. They cannot be sampled from
        as they are not random variables.
        """
        raise NotImplementedError("TransformedData does not implement _draw.")

    @property
    def torch_parametrization(self):
        """Raise error for PyTorch parameterization attempts.

        :raises NotImplementedError: Always, as transformed data has no parameters

        Transformed data components are not learnable parameters and therefore
        cannot have PyTorch parameterizations. They represent fixed computations
        based on data rather than optimizable variables.
        """
        raise NotImplementedError(
            "TransformedData does not implement torch_parametrization."
        )

    @property
    @abstractmethod
    def model_varname(self) -> str:
        """Get the Stan model variable name for this transformed data component.

        :returns: Variable name to use in generated Stan code
        :rtype: str

        This abstract property must be implemented by all subclasses to provide
        appropriate variable names for the transformed data block. The name should
        be descriptive and follow Stan naming conventions.
        """


class LogMultinomialCoefficient(TransformedData):
    r"""Pre-computed logarithmic multinomial coefficient for performance optimization.

    This class implements a performance optimization for multinomial distributions
    parameterized by log_theta. When the multinomial is used to model observable
    data (known counts), the multinomial coefficient can be pre-calculated once
    rather than computed at each MCMC iteration.

    :param counts: Multinomial parameter with log_theta parameterization
    :type counts: parameters.MultinomialLogTheta
    :param kwargs: Additional keyword arguments passed to parent class

    :cvar SHAPE_CHECK: Disabled for this component (False)

    Mathematical Background:
        The multinomial probability mass function includes a coefficient term:
            .. math::
                C(n; k_1, k_2, ..., k_m) = \frac{n!}{k_1! \times k_2! \times ... \times k_m!}

        For fixed observed counts, this coefficient is constant across all
        MCMC iterations and can be pre-computed for efficiency.

    Performance Impact:
        - Eliminates factorial computations from each MCMC iteration
        - Reduces computational overhead for multinomial likelihoods
        - Particularly beneficial for large sample sizes or many categories
        - Can provide substantial speedup for multinomial-heavy models

    Usage Requirements:
        - The counts parameter must be observable (represent data)
        - Only applicable to MultinomialLogTheta distributions
        - Automatically managed by MultinomialLogTheta components

    The coefficient is automatically included in the transformed data block
    when appropriate and removed if the parameter becomes non-observable.
    """

    SHAPE_CHECK = False

    def __init__(self, counts: "parameters.MultinomialLogTheta", **kwargs):
        """Initialize the multinomial coefficient with shape adjustment.

        :param counts: Multinomial parameter to optimize
        :type counts: parameters.MultinomialLogTheta
        :param kwargs: Additional arguments for parent initialization

        The initialization creates a coefficient component with shape adjusted
        to remove the final dimension (since the coefficient is scalar for
        each multinomial trial).
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
        """Generate Stan code for computing the log multinomial coefficient.

        :param counts: Stan variable name for the count data
        :type counts: str

        :returns: Stan function call for log multinomial coefficient
        :rtype: str

        :raises ValueError: If counts parameter is not observable

        This method generates the Stan function call to compute the logarithmic
        multinomial coefficient. The computation is only valid for observable
        (fixed) count data, as the coefficient must be deterministic.
        """
        # The counts must be an observable
        if not self.counts.observable:
            raise ValueError(
                "We can only pre-calculate the multinomial coefficient for constant counts."
            )

        return f"log_multinomial_coeff({counts})"

    @property
    def model_varname(self) -> str:
        """Get the model variable name for the multinomial coefficient.

        :returns: Descriptive variable name based on the counts parameter
        :rtype: str

        The variable name follows the pattern:
        "{counts_variable_name}.log_multinomial_coefficient"

        This provides clear identification of the coefficient's purpose and
        its relationship to the associated multinomial parameter.
        """
        return f"{self.counts.model_varname}.log_multinomial_coefficient"
