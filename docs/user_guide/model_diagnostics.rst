Model Diagnostics and Validation
================================

This guide covers essential techniques for validating Bayesian models and ensuring reliable results.

Why Model Checking Matters
--------------------------

Model validation is crucial because:

- **Verify assumptions**: Check if your model captures the data-generating process
- **Detect problems**: Identify issues before making scientific conclusions
- **Build confidence**: Ensure results are trustworthy
- **Improve models**: Guide model refinement and development

The diagnostic workflow should be an integral part of every Bayesian analysis.

Prior Predictive Checks
-----------------------
.. code-block:: python
   prior = model.prior_predictive()
   # Inspect ranges / plausibility
Convergence Diagnostics (MCMC)
------------------------------
.. code-block:: python
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)
   sample_failures, variable_failures = results.diagnose()
   print(sample_failures)
   print(variable_failures.keys())  # e.g. r_hat, ess_bulk
If r_hat or effective sample size tests fail, increase warmup / sampling or
reparameterize.
Model Fit Assessment
--------------------
Posterior predictive utilities and information criteria (LOO/WAIC) are not
yet implemented. Interim approach:
- Perform prior predictive check
- Examine parameter plausibility & convergence diagnostics
- Perform sensitivity by modest prior adjustments
Sensitivity (Manual)
--------------------
.. code-block:: python
   base = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)
   alt_model = build_model_with_alternative_prior()
   alt = alt_model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)
   print(base['param'].mean(), alt['param'].mean())
Accuracy Note
-------------
Removed unsupported: model.diagnose (use results.diagnose), posterior_predictive,
loo(), waic(), marginal_likelihood(), Bayes factors. Focused on available
diagnostics from SampleResults.diagnose().
