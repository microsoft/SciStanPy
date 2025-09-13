Custom PyTorch Distributions API Reference
==========================================

.. automodule:: scistanpy.model.components.custom_distributions.custom_torch_dists
   :undoc-members:
   :show-inheritance:

The distributions in this submodule can be broadly broken down into the following   categories:

**Multinomial Extensions**: Enhanced multinomial distributions
    - :py:class:`~scistanpy.model.components.custom_distributions.custom_torch_dists.Multinomial`: Base class with inhomogeneous total count support
    - :py:class:`~scistanpy.model.components.custom_distributions.custom_torch_dists.MultinomialProb`: Probability-parameterized multinomial
    - :py:class:`~scistanpy.model.components.custom_distributions.custom_torch_dists.MultinomialLogit`: Logit-parameterized multinomial
    - :py:class:`~scistanpy.model.components.custom_distributions.custom_torch_dists.MultinomialLogTheta`: Normalized log-probability multinomial

**Numerically Stable Distributions**: Improved standard distributions
    - :py:class:`~scistanpy.model.components.custom_distributions.custom_torch_dists.Normal`: Enhanced with stable log-CDF and log-survival functions
    - :py:class:`~scistanpy.model.components.custom_distributions.custom_torch_dists.LogNormal`: Enhanced with stable log-space probability functions

**Custom Distribution Implementations**: New distribution types
    - :py:class:`~scistanpy.model.components.custom_distributions.custom_torch_dists.Lomax`: Shifted Pareto distribution
    - :py:class:`~scistanpy.model.components.custom_distributions.custom_torch_dists.ExpLomax`: Exponential-Lomax distribution
    - :py:class:`~scistanpy.model.components.custom_distributions.custom_torch_dists.ExpExponential`: Exponential-Exponential distribution
    - :py:class:`~scistanpy.model.components.custom_distributions.custom_torch_dists.ExpDirichlet`: Exponential-Dirichlet distribution

The distributions in this module are designed to work within PyTorch's
distribution framework while providing the specific functionality required
for probabilistic modeling in SciStanPy.

Base Classes
------------
All custom distributions inherit from :py:class:`scistanpy.model.components.custom_distributions.custom_torch_dists.CustomDistribution`, which adds no additional functionality beyond PyTorch's :py:class:`torch.distributions.Distribution` but serves as a common ancestor for type checking and future extensions.

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.CustomDistribution
   :members:
   :undoc-members:
   :show-inheritance:

Multinomial Extensions
----------------------
The multinomial distributions extend PyTorch's built-in multinomial capabilities to support inhomogeneous total counts and various parameterizations.

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.Multinomial
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.MultinomialProb
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.MultinomialLogit
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.MultinomialLogTheta
   :members:
   :undoc-members:
   :show-inheritance:

Numerically Stable Distributions
--------------------------------
PyTorch does not have inbuilt support for numerically stable log-CDF and log-survival functions. SciStanPy provides enhanced versions of the :py:class:`torch.distributions.Normal` and :py:class:`torch.distributions.LogNormal` distributions that include such functions.

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.Normal
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.LogNormal
   :members:
   :undoc-members:
   :show-inheritance:

Custom Distribution Implementations
-----------------------------------
SciStanPy includes several custom distributions not available in PyTorch, implemented by extending or transforming existing PyTorch distributions.

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.Lomax
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.ExpLomax
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.ExpExponential
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.custom_distributions.custom_torch_dists.ExpDirichlet
   :members:
   :undoc-members:
   :show-inheritance: