# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Transformation components for mathematical operations on SciStanPy model parameters.

This submodule provides a comprehensive framework for mathematical transformations
that can be applied to model parameters in SciStanPy. These transformations enable
the construction of complex probabilistic models through composition of simple
mathematical operations while maintaining automatic differentiation capabilities
and Stan code generation.

The transformation system is organized into several key areas:

Core Transformation Infrastructure:
    - **transformed_parameters**: Base classes and concrete mathematical operations
    - **transformed_data**: Components for Stan's transformed data block
    - **cdfs**: Cumulative distribution function transformations

Key Features:
    - **Multi-Backend Support**: Operation across NumPy, PyTorch, and Stan
    - **Automatic Differentiation**: Maintains gradient flow for optimization
    - **Stan Code Generation**: Automatic translation to Stan programming language
    - **Operator Overloading**: Natural mathematical syntax through Python operators
    - **Shape Broadcasting**: Automatic handling of multi-dimensional operations
    - **Type Safety**: Comprehensive validation and error checking

Transformation Categories:

**Arithmetic Operations**: Basic mathematical operations between parameters
    - Addition, subtraction, multiplication, division
    - Exponentiation and negation
    - Automatic operator overloading support

**Mathematical Functions**: Standard mathematical transformations
    - Logarithms, exponentials, absolute values
    - Sigmoid and logistic functions
    - Trigonometric operations (where applicable)

**Statistical Operations**: Operations common in statistical modeling
    - Normalization (standard and log-space)
    - Reductions (sum, log-sum-exp)
    - Probability transformations

**Growth Models**: Specialized transformations for temporal modeling
    - Exponential growth (standard and log-scale)
    - Sigmoid growth with multiple parameterizations
    - Binary time-point specialized versions

**Array Operations**: Advanced array manipulation capabilities
    - NumPy-compatible indexing and slicing
    - Sequence convolution for pattern recognition
    - Multi-dimensional array transformations

**Probability Functions**: CDF and related probability computations
    - Standard and log-space CDFs
    - Survival functions and complementary CDFs
    - Numerical stability for extreme values

Usage Patterns:

**Operator Overloading** (Primary method):
    >>> param1 = Normal(mu=0, sigma=1)
    >>> param2 = Normal(mu=1, sigma=0.5)
    >>> combined = param1 + param2  # Creates AddParameter
    >>> scaled = 2 * param1        # Creates MultiplyParameter
    >>> transformed = torch.exp(param1)  # Creates ExpParameter

**Direct Instantiation** (Advanced usage):
    >>> from scistanpy.model.components.transformations.transformed_parameters import LogParameter
    >>> log_param = LogParameter(positive_param)

**CDF Access** (Automatic through Parameter classes):
    >>> normal_param = Normal(mu=0, sigma=1)
    >>> cdf_values = normal_param.cdf(x=data_points)
    >>> survival_probs = normal_param.ccdf(x=time_points)

**Growth Modeling**:
    >>> time_points = Constant([0, 1, 2, 3, 4])
    >>> initial_value = Normal(mu=100, sigma=10)
    >>> growth_rate = Normal(mu=0.1, sigma=0.02)
    >>> population = ExponentialGrowth(t=time_points, A=initial_value, r=growth_rate)

Integration with SciStanPy:

The transformation system integrates deeply with SciStanPy's model architecture:
- Automatic shape inference and broadcasting
- Parent-child relationship tracking
- Stan code generation for all transformation types
- PyTorch tensor support for gradient-based optimization
- Comprehensive error checking and validation

Performance Considerations:

**Transformed Data**: Components that execute once during Stan initialization
    - Pre-computed expensive operations
    - Constant coefficient calculations
    - Performance optimization for repeated computations

**Transformed Parameters**: Components that execute during sampling
    - Dynamic transformations applied each iteration
    - Maintains parameter relationships
    - Supports complex mathematical expressions

The transformation framework enables users to build sophisticated probabilistic
models using intuitive mathematical syntax while automatically handling the
complexities of multi-backend code generation and numerical stability.
"""
