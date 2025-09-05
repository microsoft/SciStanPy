SciStanPy API Reference
=======================

At the root level, SciStanPy exposes the :py:mod:`scistanpy.operations`, :py:mod:`scistanpy.utils`, :py:mod:`scistanpy.model.components.parameters`, and :py:mod:`scistanpy.model.results` modules, as well as the :py:class:`scistanpy.model.model.Model` and :py:class:`scistanpy.model.components.constants.Constant` classes. These components will be sufficient for most users of SciStanPy; however, links to relevant documentation for the entire public API is provided below for reference:

.. toctree::
   :maxdepth: 1

   custom_types
   defaults
   exceptions
   model/index
   plotting/index
   operations
   utils

There are also a few objects defined at the root level itself. These include the following:

.. autofunction:: scistanpy.manual_seed

.. autodata:: scistanpy.RNG