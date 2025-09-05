Stan Integration
=================

This page documents the `scistanpy.model.stan` package and its
implementation modules. The package provides small utilities and the
implementation for translating SciStanPy models into Stan programs,
assembling Stan includes shipped with the package, and compiling/running
Stan-based inference. The package's top-level symbol is
`STAN_INCLUDE_PATHS`, a list of directories that contain Stan function
snippets to be included when compiling generated Stan programs.

Submodules
----------

.. toctree::
   :maxdepth: 1

   stan_functions
   stan_model

Top-level module
----------------

.. automodule:: scistanpy.model.stan
   :members:
   :undoc-members:
   :show-inheritance:

Implementation notes
--------------------

- STAN_INCLUDE_PATHS: a list of absolute paths (usually including the
  package directory) used by the Stan code assembly step to locate
  bundled Stan function snippets and headers. The symbol is defined in
  `scistanpy.model.stan.__init__` and is intended to be consumed by the
  Stan program builder in `stan_model`.

- `stan_functions`: a small collection of Stan code fragments and
  utilities that are included into generated Stan programs when
  required by model components. These are plain Stan snippets stored
  alongside the Python package.

- `stan_model`: the implementation that assembles a complete Stan
  program from a SciStanPy `Model`, optionally compiles it, and offers
  helpers to run Stan inference using a chosen Stan frontend. The exact
  frontend integration (e.g., CmdStanPy, CmdStan, etc.) is handled in
  the implementation and may evolve; consult the `stan_model` page for
  the current API and runtime requirements.

Minimal usage examples
----------------------

Include bundled Stan snippets when compiling a model (used inside the
library):

.. code-block:: python

   from scistanpy.model.stan import STAN_INCLUDE_PATHS

   # STAN_INCLUDE_PATHS is passed to the Stan program builder so it can
   # discover and include local Stan function snippets.

Convert a SciStanPy model to a Stan owner object via the library API:

.. code-block:: python

   # Typical higher-level usage (see stan_model page for details):
   stan_prog = model.to_stan(output_dir='./stan_cache', force_compile=False)
   results = stan_prog.sample(data={'y': observed_y}, chains=4)

Further reading
---------------

For details on the Stan program snippets included with SciStanPy, see
`stan_functions`. For the Stan program generation, compilation and the
inference wrappers consult `stan_model`.
