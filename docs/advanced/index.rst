Advanced Usage
==============

This section aggregates in–depth guides for power users who want to:
- Optimize performance and memory usage
- Extend SciStanPy with custom components
- Integrate with larger scientific / data ecosystems
- Build and manage advanced backends and workflows

What Is Currently Implemented
-----------------------------
The core implemented interfaces you can rely on today:

- Model construction via scistanpy.Model
- Maximum Likelihood estimation: Model.mle(...)
- Stan MCMC sampling: Model.mcmc(...)
- Prior predictive dashboard: Model.prior_predictive()
- Results analysis classes: MLEInferenceRes, SampleResults (summaries, diagnostics)
- Utility layer: scistanpy.utils (lazy imports, stable_sigmoid, chunking, dask context)
- Stan code generation pipeline (automatic when calling Model.to_stan() / Model.mcmc())

Planned / Conceptual Topics
---------------------------
Some advanced guides contain forward-looking patterns (e.g. alternative backends,
custom inference engines, distributed workflows). Treat those examples as design
patterns rather than guaranteed current APIs.

Prerequisites
-------------
Before exploring these advanced guides you should be comfortable with:
- Basic model definition (parameters, observables)
- Running Model.mle() and Model.mcmc()
- Interpreting SampleResults diagnostics (r_hat, ESS, divergences)
- Python packaging & virtual environments (for extension work)

Choosing a Path
---------------
Use this quick map to pick the right guide:

- Need speed or lower memory footprint?  -> Performance Optimization
- Want to plug SciStanPy into external tooling? -> Integration Patterns
- Creating new distributions or transformations? -> Creating Custom Distributions
- Adding domain / backend extensions? -> Extending SciStanPy
- Understanding internal multi-backend design? -> Backend Implementation Details

Quick Reference Snippets
------------------------

.. code-block:: python

   import scistanpy as ssp
   # Compile & run MLE
   mle_res = model.mle()
   # Run Stan MCMC
   mcmc_res = model.mcmc(chains=4, iter_sampling=1000)
   # Diagnostics
   mcmc_res.diagnose()
   # Utilities
   import numpy as np
   stable = ssp.utils.stable_sigmoid(np.array([-50, 0, 50]))

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Advanced Guides

   performance_optimization
   integration_patterns
   extending_scistanpy
   custom_distributions
   backend_details

Best Practices Snapshot
-----------------------
1. Start simple (MLE) before scaling to full MCMC.
2. Profile first, optimize second (avoid premature micro-optimizations).
3. Encapsulate domain-specific math in reusable transformed parameters.
4. Keep Stan-facing shapes & names stable—renaming affects cached builds.
5. Use scistanpy.utils.get_chunk_shape for large posterior arrays before writing custom chunk logic.
6. Validate new extensions with statistical & numerical tests (shape, gradients, stability).

Disclaimer
----------
Some illustrative examples in the advanced guides may reference experimental or future
APIs. Always cross-check with the actual scistanpy.* modules (or help(obj)) if unsure.
