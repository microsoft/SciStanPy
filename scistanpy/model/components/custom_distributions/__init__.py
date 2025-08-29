"""Custom distribution implementations for SciStanPy models.

This submodule provides enhanced and custom distribution implementations that
extend the capabilities of standard PyTorch and SciPy distributions. These
distributions address specific modeling needs and limitations in the standard
libraries while maintaining compatibility with SciStanPy's multi-backend
architecture.

The custom distributions are organized into two main categories:

PyTorch Extensions (custom_torch_dists):
    Enhanced PyTorch distributions with additional functionality and numerical
    improvements for probabilistic modeling scenarios.

    **Multinomial Extensions**:
        - Multinomial: Support for inhomogeneous total counts across batches
        - MultinomialProb: Probability-parameterized multinomial
        - MultinomialLogit: Logit-parameterized multinomial
        - MultinomialLogTheta: Normalized log-probability multinomial

    **Numerically Enhanced Distributions**:
        - Normal: Improved log-CDF and log-survival functions
        - LogNormal: Enhanced numerical stability for extreme values

    **Custom Distribution Types**:
        - Lomax: Shifted Pareto distribution implementation
        - ExpLomax: Log-transformed Lomax distribution
        - ExpExponential: Log-transformed exponential (Gumbel) distribution
        - ExpDirichlet: Log-transformed Dirichlet distribution

SciPy Extensions (custom_scipy_dists):
    Extended SciPy distributions with enhanced batch support and custom
    parameterizations for advanced statistical modeling.

    **Enhanced Multivariate Distributions**:
        - CustomDirichlet: Variable batch dimension support
        - CustomMultinomial: Flexible batch handling for multinomial
        - MultinomialLogit: Logit-parameterized multinomial
        - MultinomialLogTheta: Log-probability parameterized multinomial

    **Log-Transformed Distributions**:
        - ExpDirichlet: Log-space Dirichlet with Jacobian corrections
        - TransformedScipyDist: Framework for general transformations
        - LogUnivariateScipyTransform: Log-transformation implementation

Key Features:

**Enhanced Batch Operations**: Support for variable batch dimensions and
    flexible broadcasting that goes beyond standard library limitations.

**Alternative Parameterizations**: Multiple parameterization options (logits,
    log-probabilities, etc.) that are more convenient for different modeling
    scenarios and optimization approaches.

**Numerical Stability**: Improved implementations that handle extreme values,
    very small probabilities, and numerical edge cases more robustly than
    standard implementations.

**Multi-Backend Compatibility**: Consistent interfaces that work across
    PyTorch, SciPy, and Stan while maintaining backend-specific optimizations.

**Custom Distribution Types**: Implementations of distributions not available
    in standard libraries, particularly log-transformed variants useful for
    multiplicative modeling and log-space computations.

Integration with SciStanPy:

These custom distributions integrate with SciStanPy's parameter system
to provide:
- Automatic Stan code generation for custom distributions
- Consistent sampling interfaces across backends
- Support for transformed parameter relationships
- Enhanced error checking and validation
- Optimized computation paths for common modeling patterns

The distributions maintain the standard PyTorch and SciPy interfaces while
providing additional functionality needed for advanced probabilistic modeling
scenarios. They are designed to be drop-in replacements for standard
distributions in most cases, with enhanced capabilities for specific use cases.
"""
