Development
===========

Resources for contributors: architecture overview, contribution workflow, testing,
release process, and internal design notes. Content here reflects currently
implemented public APIs only:

- Model.mle(...)
- Model.mcmc(...)
- Model.draw(n)
- Model.prior_predictive()
- SampleResults.diagnose(...)

Planned / unimplemented items (e.g. variational inference, posterior predictive
helpers, WAIC/LOO wrappers, GPU acceleration, streaming/distributed inference)
are intentionally excluded.

Contents
--------

.. toctree::
   :maxdepth: 1

   architecture
   contributing
   testing
   release_process

Accuracy Note
-------------
If you spot references to non‑existent functions (e.g. variational(), sample(),
posterior_predictive()), please open a documentation issue so it can be removed.
