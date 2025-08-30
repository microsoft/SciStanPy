Frequently Asked Questions
==========================

This FAQ addresses common questions about SciStanPy usage, capabilities, and best practices.

General Questions
-----------------

**What is SciStanPy?**

SciStanPy is a Python framework designed to make Bayesian statistical modeling accessible to scientists and researchers. It allows you to express complex probabilistic models using intuitive Python syntax while automatically generating efficient computational code.

**Who should use SciStanPy?**

SciStanPy is designed for scientists, researchers, and analysts who:

- Have basic Python programming skills
- Work with experimental or observational data
- Need uncertainty quantification in their analyses
- Want to build probabilistic models without deep statistical programming knowledge

**How does SciStanPy differ from other Bayesian libraries?**

SciStanPy focuses specifically on scientific applications with:

- **Scientific syntax**: Express models in terms familiar to scientists
- **Multi-backend support**: Automatically uses NumPy, PyTorch, or Stan as appropriate
- **Built-in validation**: Comprehensive error checking for common modeling mistakes
- **Domain-specific features**: Specialized distributions and transformations for scientific modeling

**Do I need to know Stan to use SciStanPy?**

No! SciStanPy automatically generates Stan code for you. However, understanding basic Bayesian concepts will help you build better models.

Installation and Setup
----------------------

**What Python versions does SciStanPy support?**

SciStanPy supports Python 3.8 and later versions.

**Why is my installation failing?**

Common installation issues:

.. code-block:: bash

   # Try upgrading pip first
   pip install --upgrade pip

   # Install with specific requirements
   pip install scistanpy --no-cache-dir

   # For C++ compiler issues (Windows)
   # Install Visual Studio Build Tools

**Can I use SciStanPy without Stan?**

Yes! You can use NumPy and PyTorch backends for many operations. However, full MCMC sampling requires Stan.

**How do I install optional dependencies?**

.. code-block:: bash

   # For GPU support
   pip install torch[cuda]

   # For plotting
   pip install matplotlib seaborn plotly

   # For Jupyter notebooks
   pip install jupyter ipywidgets

Model Building
--------------

**How do I choose the right distribution?**

Consider your data characteristics:

- **Continuous, unbounded**: Normal, Student's t
- **Continuous, positive**: LogNormal, Gamma, Exponential
- **Continuous, bounded [0,1]**: Beta
- **Count data**: Poisson, Negative Binomial
- **Binary**: Bernoulli, Binomial
- **Categorical**: Multinomial

**What if my distribution isn't available?**

SciStanPy provides many distributions, but you can also:

- Use transformations of existing distributions
- Create custom distributions
- Request new distributions in GitHub discussions

**How do I handle missing data?**

.. code-block:: python

   # Exclude missing values
   clean_data = data[~np.isnan(data)]

   # Or model missingness explicitly
   observed_mask = ~np.isnan(data)
   likelihood = ssp.parameters.Normal(mu=predictions[observed_mask], sigma=noise)
   likelihood.observe(clean_data)

**Should I center and scale my data?**

Generally yes, especially for:

- Regression models with multiple predictors
- When predictors have very different scales
- To improve MCMC sampling efficiency

.. code-block:: python

   # Center and scale predictors
   X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

Prior Specification
------------------

**How do I choose priors?**

**Weakly informative priors** are usually best:

.. code-block:: python

   # For standardized coefficients
   beta = ssp.parameters.Normal(mu=0, sigma=1)

   # For positive scale parameters
   sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

**What if I have no prior information?**

Use weakly informative priors rather than flat priors:

.. code-block:: python

   # Better than flat priors
   intercept = ssp.parameters.Normal(mu=0, sigma=10)  # Allows wide range
   slope = ssp.parameters.Normal(mu=0, sigma=2.5)     # Reasonable effect sizes

**How do I incorporate expert knowledge?**

.. code-block:: python

   # Literature values suggest mean around 5, rarely above 20
   parameter = ssp.parameters.LogNormal(mu=np.log(5), sigma=0.7)

**Can I use flat (uniform) priors?**

Generally not recommended. Weakly informative priors are usually better for:

- Computational stability
- Regularization
- Avoiding improper posteriors

Inference and Sampling
----------------------

**Which inference method should I use?**

Currently implemented:
- Maximum Likelihood: `model.mle(...)`
- Hamiltonian Monte Carlo (Stan): `model.mcmc(...)`
Use MLE for quick checks / point estimates; use MCMC for full posterior inference.

(Previous references to `variational()` removed – VI not yet implemented.)

**Why isn't my model converging?**

.. code-block:: python

   res = model.mcmc(chains=4, iter_sampling=1000, iter_warmup=500)
   sample_fails, var_fails = res.diagnose()

   # If many failures:
   res = model.mcmc(chains=4, iter_sampling=2000, iter_warmup=1000)

**How many samples do I need?**

Aim for effective sample sizes adequate for your parameters; increase `iter_sampling` if diagnostics flag low ESS.

**What if I get divergent transitions?**

Adjust priors / reparameterize or increase warmup; re-run `model.mcmc`.

Model Checking
--------------

Removed non-existent helpers (`model.posterior_predictive`, `model.loo`, `model.get_stan_code`).

Current checks:
- Prior structure: `model.draw(n)` or `model.prior_predictive()`
- MCMC diagnostics: `SampleResults.diagnose()`

**How do I validate my model?**

.. code-block:: python

   prior_samples = model.draw(200)
   mcmc_res = model.mcmc(chains=4, iter_sampling=1000, iter_warmup=500)
   sample_failures, variable_failures = mcmc_res.diagnose()

Performance and Troubleshooting
-------------------------------

Replace `variational()` and `model.sample`:

.. code-block:: python

   # Quick check via MLE
   mle_res = model.mle(epochs=20000, early_stop=10)

   # Then full MCMC
   mcmc_res = model.mcmc(chains=4, iter_sampling=1000)

Memory tips: lower `iter_sampling`, fewer `chains`, or run shorter warmup.

Advanced Usage
--------------

**How do I create custom distributions?**

(General pattern remains conceptual—ensure you define new components consistent with existing component architecture.)

Removed saving/loading examples using `model.save` (not implemented).

Best Practices
--------------

Updated to avoid references to unsupported APIs.

Need More Help?
--------------

Clarified search terms: use `mle`, `mcmc`, `prior_predictive`, `diagnose`.

Note:
    Prior version referenced non-existent functions (`variational`, `posterior_predictive`,
    `loo`, `model.sample`). These have been removed or reworded to reflect current API.
