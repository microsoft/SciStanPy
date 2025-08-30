Inference Methods
=================

This guide covers the different inference methods available in SciStanPy and when to use each approach.

Overview of Inference Methods
-----------------------------

SciStanPy currently provides:
- Maximum Likelihood Estimation: model.mle(...)
- Hamiltonian Monte Carlo via Stan: model.mcmc(...)
(Variational inference and other approximations are not yet implemented.)
Markov Chain Monte Carlo (MCMC)
-------------------------------
.. code-block:: python
   # Basic MCMC
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)
   # Diagnostics
   sample_failures, variable_failures = results.diagnose()
   if sample_failures:
       # Increase iterations or adjust model
       results = model.mcmc(chains=4, iter_warmup=1000, iter_sampling=2000)
Understanding Output
--------------------
.. code-block:: python
   slope_mean = results['slope'].mean()
   slope_sd   = results['slope'].std()
   ci = np.percentile(results['slope'], [2.5, 97.5])
   print(slope_mean, slope_sd, ci)
Maximum Likelihood Estimation (MLE)
-----------------------------------
.. code-block:: python
   mle_res = model.mle(epochs=10000, early_stop=10)
   print(mle_res['estimates'])
Choosing a Method
-----------------
Use mle() for quick sanity checks / starting values; use mcmc() for final
inference and uncertainty quantification.
Diagnosing Inference Problems
-----------------------------
.. code-block:: python
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)
   sample_failures, variable_failures = results.diagnose()
   print(sample_failures, variable_failures)
   # Example: increase iterations if ess_bulk failed
   if 'ess_bulk' in variable_failures:
       results = model.mcmc(chains=4, iter_warmup=1000, iter_sampling=2000)
Accuracy Note
-------------
Removed unsupported: model.sample, variational(), adapt_delta / advanced sampler
tuning, posterior_predictive, LOO/WAIC, GPU / parallel chain arguments.

This comprehensive guide covers all aspects of inference in SciStanPy, from basic usage to advanced debugging techniques.
   models = {
       'model1': model1,
       'model2': model2,
       'model3': model3
   }

   results = {
       'model1': mcmc_results1,
       'model2': mcmc_results2,
       'model3': mcmc_results3
   }

   # Compute information criteria for all models
   comparison = {}
   for name, (model, result) in zip(models.keys(), zip(models.values(), results.values())):
       loo = model.loo(result)
       comparison[name] = {
           'elpd_loo': loo['elpd_loo'],
           'se_elpd_loo': loo['se_elpd_loo'],
           'looic': loo['looic']
       }

   # Display comparison
   print("Model comparison (higher ELPD is better):")
   for name, metrics in comparison.items():
       print(f"  {name}: ELPD = {metrics['elpd_loo']:.2f} ± {metrics['se_elpd_loo']:.2f}")

Inference Method Selection Guide
-------------------------------

**Start with MLE when:**
- You need quick parameter estimates
- Computational resources are limited
- Model is simple with well-behaved likelihood
- You're doing initial exploration

**Use MCMC when:**
- You need full uncertainty quantification
- Publishing results requiring proper statistical inference
- Model is complex or hierarchical
- You have sufficient computational resources

**Consider VI when:**
- Dataset is large (>10,000 observations)
- You need fast approximate inference
- Doing exploratory analysis
- Real-time or interactive applications

**Always do predictive checks:**
- Prior predictive checks before fitting
- Posterior predictive checks after fitting
- Cross-validation for model selection

Computational Considerations
---------------------------

**Parallelization:**

.. code-block:: python

   # Use multiple cores for MCMC chains
   mcmc_results = model.sample(
       n_chains=4,              # Run 4 chains in parallel
       parallel_chains=True     # Enable parallelization
   )

**GPU Acceleration:**

.. code-block:: python

   # Use GPU for VI (if available)
   vi_results = model.variational(
       backend='pytorch',       # Use PyTorch backend
       device='cuda'           # Use GPU
   )

**Memory Management:**

.. code-block:: python

   # For large models, control memory usage
   mcmc_results = model.sample(
       n_samples=1000,         # Smaller sample size
       save_warmup=False,      # Don't save warmup samples
       output_format='zarr'    # Use efficient storage
   )

This comprehensive guide provides the foundation for choosing and applying appropriate inference methods in your scientific modeling workflow.
