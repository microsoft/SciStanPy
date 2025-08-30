User Guide
==========

Core usage documentation for currently implemented public APIs:

Implemented:
- Model.draw(n)
- Model.prior_predictive()
- Model.mle(...)
- Model.mcmc(chains=..., iter_warmup=..., iter_sampling=...)
- SampleResults.diagnose()

Not yet implemented (and therefore not documented here): variational inference,
posterior predictive helpers, automatic LOO/WAIC, model.sample(), model.variational(),
distributed / GPU execution controls, custom sampler plugin APIs.

Contents
--------

.. toctree::
   :maxdepth: 1

   modeling_basics
   distributions
   transformations
   inference_methods
   model_diagnostics
   advanced_topics

Accuracy Note
-------------
If you find references to unsupported functions (posterior_predictive, loo, waic,
variational, model.sample, model.diagnose) please open a documentation issue.
