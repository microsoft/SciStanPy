Stan Model API Reference
========================

.. automodule:: scistanpy.model.stan.stan_model
   :undoc-members:
   :show-inheritance:

Stan code generation is supported by four main classes, each described below. Users
of SciStanPy will primarily interact with the :py:class:`~scistanpy.model.stan.stan_model.StanModel` class--the others classes are used internally to generate the Stan code.

Quickly navigate to a section:

- :py:class:`~scistanpy.model.stan.stan_model.StanCodeBase`
- :py:class:`~scistanpy.model.stan.stan_model.StanForLoop`
- :py:class:`~scistanpy.model.stan.stan_model.StanProgram`
- :py:class:`~scistanpy.model.stan.stan_model.StanModel`

Abstract Base for Stan Code Components
--------------------------------------
.. autoclass:: scistanpy.model.stan.stan_model.StanCodeBase
   :members:
   :undoc-members:
   :show-inheritance:

For Loop Representation in Stan Code
------------------------------------
.. autoclass:: scistanpy.model.stan.stan_model.StanForLoop
   :members:
   :undoc-members:
   :show-inheritance:

Base Stan Program Structure
---------------------------
.. autoclass:: scistanpy.model.stan.stan_model.StanProgram
   :members:
   :undoc-members:
   :show-inheritance:

Stan Model Representation
-------------------------
.. autoclass:: scistanpy.model.stan.stan_model.StanModel
   :members:
   :undoc-members:
   :show-inheritance: