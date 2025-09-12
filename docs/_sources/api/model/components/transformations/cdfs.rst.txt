CDFs API Reference
==================

.. automodule:: scistanpy.model.components.transformations.cdfs
   :undoc-members:
   :show-inheritance:

Base Class
----------
All CDF-like classes inherit from a single base class:

.. autoclass:: scistanpy.model.components.transformations.cdfs.CDFLike
   :members:
   :show-inheritance:

Concrete Classes
----------------
Each of the four CDF-like operations is implemented as a separate subclass of the base class:

.. autoclass:: scistanpy.model.components.transformations.cdfs.CDF
   :members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.cdfs.LogCDF
   :members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.cdfs.SurvivalFunction
   :members:
   :show-inheritance:

.. autoclass:: scistanpy.model.components.transformations.cdfs.LogSurvivalFunction
   :members:
   :show-inheritance: