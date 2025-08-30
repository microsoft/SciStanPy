Examples
========

Curated examples illustrating core SciStanPy usage with the currently
implemented public APIs:

Implemented inference & utilities used here:
- Model.draw(n)
- Model.mle(...)
- Model.mcmc(chains=..., iter_warmup=..., iter_sampling=...)
- Model.prior_predictive()
- SampleResults.diagnose()

Deliberately excluded (not yet implemented): posterior_predictive(), loo(), waic(),
variational(), model.sample(), model.predict(), model averaging helpers.

Contents
--------

.. toctree::
   :maxdepth: 1

   parameter_estimation
   regression_models
   time_series
   hierarchical_models
   model_comparison
   experimental_design

Accuracy Note
-------------
If you notice an example using an unsupported method (e.g. posterior_predictive,
loo, waic, variational), please open a documentation issue so it can be fixed.
