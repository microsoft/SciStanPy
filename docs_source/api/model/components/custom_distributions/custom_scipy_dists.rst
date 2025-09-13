Custom SciPy Distributions API Reference
========================================

.. automodule:: scistanpy.model.components.custom_distributions.custom_scipy_dists
   :undoc-members:
   :show-inheritance:

The following custom probability distributions are implemented using SciPy's distribution framework:
    - :py:obj:`~scistanpy.model.components.custom_distributions.custom_scipy_dists.dirichlet`
    - :py:obj:`~scistanpy.model.components.custom_distributions.custom_scipy_dists.expdirichlet`
    - :py:obj:`~scistanpy.model.components.custom_distributions.custom_scipy_dists.expexponential`
    - :py:obj:`~scistanpy.model.components.custom_distributions.custom_scipy_dists.explomax`
    - :py:obj:`~scistanpy.model.components.custom_distributions.custom_scipy_dists.multinomial`
    - :py:obj:`~scistanpy.model.components.custom_distributions.custom_scipy_dists.multinomial_logit`
    - :py:obj:`~scistanpy.model.components.custom_distributions.custom_scipy_dists.multinomial_log_theta`

The distributions in this module are designed to work within SciPy's distribution
framework while providing enhanced functionality for advanced probabilistic modeling
scenarios commonly encountered in SciStanPy applications. They will not normally be used directly by users; instead, they are used internally by SciStanPy model components such as :py:class:`~scistanpy.model.components.parameters.Parameter` for sampling from prior distributions.

Custom-Built SciPy Distributions
--------------------------------
Some distributions are extensions of existing SciPy distributions while others are built from the ground up to provide specific features. First, the distribution built from the ground up:

.. autoclass:: scistanpy.model.components.custom_distributions.custom_scipy_dists.CustomDirichlet
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: logpdf, pdf, mean, var, cov, entropy

.. autoclass:: scistanpy.model.components.custom_distributions.custom_scipy_dists.ExpDirichlet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.custom_distributions.custom_scipy_dists.CustomMultinomial
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.custom_distributions.custom_scipy_dists.MultinomialLogit
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: softmax_p, logpmf, pmf, mean, var, cov, entropy, rvs

.. autoclass:: scistanpy.model.components.custom_distributions.custom_scipy_dists.MultinomialLogTheta
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: exp_p, logpmf, pmf, mean, var, cov, entropy, rvs

Transforms of Existing SciPy Distributions
------------------------------------------
Other distributions are implemented as transforms of existing SciPy distributions to provide additional flexibility. Transformations are applied to the base distribution using the following classes:

.. autoclass:: scistanpy.model.components.custom_distributions.custom_scipy_dists.TransformedScipyDist
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.custom_distributions.custom_scipy_dists.LogUnivariateScipyTransform
   :members:
   :undoc-members:
   :show-inheritance:

Distribution Instances
----------------------
The above classes are used to create the following distribution instances that can be used directly in SciStanPy models:

.. autodata:: scistanpy.model.components.custom_distributions.custom_scipy_dists.dirichlet
   :no-value:

.. autodata:: scistanpy.model.components.custom_distributions.custom_scipy_dists.expdirichlet
   :no-value:

.. autodata:: scistanpy.model.components.custom_distributions.custom_scipy_dists.expexponential
   :no-value:

.. autodata:: scistanpy.model.components.custom_distributions.custom_scipy_dists.explomax
   :no-value:

.. autodata:: scistanpy.model.components.custom_distributions.custom_scipy_dists.multinomial
   :no-value:

.. autodata:: scistanpy.model.components.custom_distributions.custom_scipy_dists.multinomial_logit
   :no-value:

.. autodata:: scistanpy.model.components.custom_distributions.custom_scipy_dists.multinomial_log_theta
   :no-value: