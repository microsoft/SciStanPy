Stan Functions Library Reference
================================

This reference covers the custom Stan functions that provide specialized probability distributions and mathematical operations for SciStanPy models. Users will not typically interact with these Stan code fragments directly, as they are automatically injected into Stan code during the conversion from Python to Stan when needed.

Broadly, the custom Stan functions can be categorized into "Custom Distributions" and "Specialized Operations". In lieu of documenting Stan code directly, this reference focuses on a high level overview of different files' contents that are most relevant to SciStanPy users. For detailed treatment of the distribution types mentioned, refer to the appropriate documentation in the :doc:`parameters <../components/parameters>` section.

Custom Distributions
--------------------

Multinomial Function Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File**: ``multinomial.stanfunctions``

This file provides functions that support multinomial-like distributions in SciStanPy. Most notably, it provides stan code in support of the :py:class:`~scistanpy.model.components.parameters.MultinomialLogTheta` custom parameter type. It also houses a utility function for computing the log multinomial coefficient, an operation that is particularly useful when the :py:class:`~scistanpy.model.components.parameters.MultinomialLogTheta` distribution is used to model an observable, as it allows for coefficient calculation in the ``transformed data`` block of Stan, eliminating redundant computation during sampling.


Exp-Exponential Function Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File**: ``expexponential.stanfunctions``

This file provides functions that support the :py:class:`~scistanpy.model.components.parameters.ExpExponential` custom parameter type, which is useful for modeling log-transformed exponential variables.

Exp-Dirichlet Function Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**File**: ``expdirichlet.stanfunctions``

This file provides functions that support the :py:class:`~scistanpy.model.components.parameters.ExpDirichlet` custom parameter type, which is useful for modeling log-transformed Dirichlet variables (i.e., log-transformed simplices).

Exp-Lomax Function Library
^^^^^^^^^^^^^^^^^^^^^^^^^^

**File**: ``explomax.stanfunctions``

This file provides functions that support the :py:class:`~scistanpy.model.components.parameters.ExpLomax` custom parameter type, which is useful for modeling log-transformed Lomax variables.

Specialized Operations
----------------------

Sequence Convolution
^^^^^^^^^^^^^^^^^^^^

**File**: ``pssm.stanfunctions``

This file provides functions for performing convolutions of sequences with position-specific scoring matrices (PSSMs). These operations are particularly useful in bioinformatics applications, such as motif scanning in DNA sequences. See :py:class:`~scistanpy.model.components.transformations.transformed_parameters.ConvolveSequence` for more details.