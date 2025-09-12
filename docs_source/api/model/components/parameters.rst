Parameters API Reference
========================

.. automodule:: scistanpy.model.components.parameters
   :undoc-members:
   :show-inheritance:

Base Classes
------------
.. autoclass:: scistanpy.model.components.parameters.ParameterMeta
   :members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.parameters.Parameter
   :members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.parameters.ContinuousDistribution
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF

.. autoclass:: scistanpy.model.components.parameters.DiscreteDistribution
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF

Univariate Continuous Distributions
-----------------------------------

.. autoclass:: scistanpy.model.components.parameters.Normal
   :members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.parameters.HalfNormal
   :members:
   :show-inheritance:
   :exclude-members: write_dist_args, CDF, SF, LOG_CDF, LOG_SF

.. autoclass:: scistanpy.model.components.parameters.UnitNormal
   :members:
   :show-inheritance:
   :exclude-members: write_dist_args, CDF, SF, LOG_CDF, LOG_SF

.. autoclass:: scistanpy.model.components.parameters.LogNormal
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF

.. autoclass:: scistanpy.model.components.parameters.Beta
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF

.. autoclass:: scistanpy.model.components.parameters.Gamma
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF

.. autoclass:: scistanpy.model.components.parameters.InverseGamma
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF

.. autoclass:: scistanpy.model.components.parameters.Exponential
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF, get_supporting_functions

.. autoclass:: scistanpy.model.components.parameters.ExpExponential
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF

.. autoclass:: scistanpy.model.components.parameters.Lomax
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF, write_dist_args

.. autoclass:: scistanpy.model.components.parameters.ExpLomax
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF, get_supporting_functions

Univariate Discrete Distributions
---------------------------------
.. autoclass:: scistanpy.model.components.parameters.Binomial
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF

.. autoclass:: scistanpy.model.components.parameters.Poisson
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF

Multivariate Continuous Distributions
-------------------------------------
.. autoclass:: scistanpy.model.components.parameters.Dirichlet
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF

.. autoclass:: scistanpy.model.components.parameters.ExpDirichlet
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF, get_raw_stan_parameter_declaration,
      get_right_side, get_supporting_functions, get_transformation_assignment

Multivariate Discrete Distributions
-----------------------------------
.. autoclass:: scistanpy.model.components.parameters.Multinomial
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF

.. autoclass:: scistanpy.model.components.parameters.MultinomialLogit
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF

.. autoclass:: scistanpy.model.components.parameters.MultinomialLogTheta
   :members:
   :show-inheritance:
   :exclude-members: CDF, SF, LOG_CDF, LOG_SF, get_supporting_functions, get_right_side,
      write_dist_args

Utilities
---------
.. autoclass:: scistanpy.model.components.parameters.ClassOrInstanceMethod
   :members: __get__
   :show-inheritance: