"""Core model components for SciStanPy probabilistic modeling framework.

This submodule contains the fundamental building blocks for constructing
probabilistic models in SciStanPy. It provides a comprehensive set of
components that enable users to define complex probabilistic models through
composition of simple, well-defined elements.

The components system is designed around several key principles:

**Multi-Backend Support**: All components work across NumPy, PyTorch, and Stan
    backends, enabling flexible computation while maintaining consistent interfaces.

**Automatic Code Generation**: Components automatically generate appropriate
    Stan code for their mathematical operations, enabling compilation to
    high-performance sampling algorithms.

**Compositional Design**: Complex models are built by combining simple components
    through mathematical operations and transformations.

**Type Safety**: Comprehensive type checking and validation ensure model
    correctness and provide clear error messages.

Core Component Categories:

**Abstract Infrastructure** (abstract_model_component):
    Base classes and interfaces that define the fundamental behavior and
    structure of all model components. Provides the foundation for shape
    inference, Stan code generation, and multi-backend operation.

**Probability Distributions** (parameters):
    Comprehensive library of probability distributions for both continuous
    and discrete random variables. Includes standard distributions with
    enhanced functionality and multiple parameterizations for flexibility.

**Mathematical Transformations** (transformations):
    Operations that combine and transform parameters through mathematical
    functions. Enables creation of complex relationships while maintaining
    automatic differentiation and Stan code generation.

**Custom Distributions** (custom_distributions):
    Extended and custom distribution implementations that address limitations
    in standard libraries. Provides enhanced batch operations, numerical
    stability improvements, and distributions not available elsewhere.

**Constants and Data** (constants):
    Components for incorporating fixed data and constants into models.
    Handles data input, shape validation, and integration with variable
    components.

Key Features:

**Automatic Shape Inference**: Components automatically determine output
    shapes based on input shapes and mathematical operations, reducing
    the need for manual shape specification.

**Stan Code Generation**: All components can generate appropriate Stan
    code for their operations, enabling compilation to efficient MCMC
    and optimization algorithms.

**Gradient Compatibility**: Components maintain PyTorch tensor operations
    with automatic differentiation for gradient-based optimization and
    variational inference.

**Error Checking**: Comprehensive validation of parameter constraints,
    shape compatibility, and mathematical validity with informative
    error messages.
"""
