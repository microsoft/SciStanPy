Getting Help and Support
========================

This guide helps you find the right resources when you need assistance with SciStanPy.

Quick Troubleshooting Checklist
-------------------------------

Before seeking help, try these common solutions:

**Installation Issues:**

.. code-block:: bash

   # Update to latest version
   pip install --upgrade scistanpy

   # Clean install
   pip uninstall scistanpy
   pip install scistanpy

   # Check dependencies
   pip check

**Model Issues:**

.. code-block:: python

   # Generate prior predictive visualization (interactive Panel layout)
   pp_view = model.prior_predictive()

   # Draw prior samples (dictionary or xarray)
   prior_draws = model.draw(100)

   # Run a small MCMC test (use small iterations just to verify)
   test_res = model.mcmc(chains=2, iter_sampling=100, iter_warmup=100)

**Convergence Problems:**

.. code-block:: python

   # After running MCMC
   results = model.mcmc(chains=4, iter_sampling=1000, iter_warmup=500)

   # Run diagnostics on the SampleResults object
   sample_failures, variable_failures = results.diagnose(silent=False)

   # Inspect failed diagnostics programmatically
   print(variable_failures.keys())  # e.g. 'r_hat', 'ess_bulk', etc.

   # Increase sampling if needed
   results = model.mcmc(chains=4, iter_sampling=2000, iter_warmup=1000)

Documentation Resources
-----------------------

**Primary Documentation**

- **User Guide**: Comprehensive tutorials and modeling examples
- **API Reference**: Complete function and class documentation
- **Examples Gallery**: Real-world scientific applications
- **Best Practices**: Guidelines for effective modeling

**Specialized Guides**

- **Getting Started**: Installation and first models
- **Advanced Topics**: Custom distributions and performance optimization
- **Troubleshooting**: Common issues and solutions

**Interactive Resources**

- **Jupyter Notebooks**: Downloadable tutorial notebooks
- **Code Examples**: Copy-paste examples for common tasks
- **Video Tutorials**: Step-by-step modeling walkthroughs

Community Support Channels
--------------------------

GitHub Discussions
~~~~~~~~~~~~~~~~~

**Best for:** General questions, feature requests, sharing examples

- Browse existing discussions: https://github.com/microsoft/SciStanPy/discussions
- Search before posting to avoid duplicates
- Use clear, descriptive titles
- Include minimal reproducible examples

**Discussion Categories:**

- **General**: General questions and discussions
- **Help**: Specific troubleshooting requests
- **Ideas**: Feature requests and suggestions
- **Show and Tell**: Share your models and applications

GitHub Issues
~~~~~~~~~~~~

**Best for:** Bug reports, documentation issues

- Check existing issues first: https://github.com/microsoft/SciStanPy/issues
- Use issue templates when available
- Provide complete error messages and tracebacks
- Include system information (OS, Python version, SciStanPy version)

**Creating Effective Bug Reports:**

.. code-block:: python

   # Include this information in bug reports
   import scistanpy as ssp
   import sys
   import platform

   print(f"SciStanPy version: {ssp.__version__}")
   print(f"Python version: {sys.version}")
   print(f"Platform: {platform.platform()}")

   # Include minimal code that reproduces the issue
   # ...your code here...

Stack Overflow
~~~~~~~~~~~~~

**Best for:** Programming questions with broader context

- Tag questions with ``scistanpy`` and ``bayesian-statistics``
- Follow Stack Overflow guidelines for question quality
- Include complete, minimal examples
- Accept answers that solve your problem

Asking Effective Questions
-------------------------

**Provide Context**

- Describe your scientific problem, not just the technical issue
- Explain what you're trying to accomplish
- Include relevant domain knowledge

**Create Minimal Examples**

.. code-block:: python

   # Good: Minimal reproducible example
   import scistanpy as ssp
   import numpy as np

   # Simplified version of your problem
   data = np.array([1, 2, 3, 4, 5])
   param = ssp.parameters.Normal(mu=0, sigma=1)
   likelihood = ssp.parameters.Normal(mu=param, sigma=0.1)
   likelihood.observe(data)

   model = ssp.Model(likelihood)
   # This is where the error occurs...

**Include Complete Error Messages**

.. code-block:: python

   # Include the full traceback
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
     File "scistanpy/...", line XXX, in function_name
       ...
   ValueError: Specific error message here

**Be Specific About Expected vs Actual Behavior**

- "I expected X to happen"
- "Instead, Y happened"
- "Here's the complete code and error message"

Professional Support Options
----------------------------

**Consulting Services**

For organizations requiring dedicated support:

- Custom model development
- Performance optimization consulting
- Training workshops and seminars
- Code review and best practices guidance

Contact: scistanpy-support@microsoft.com

**Training and Workshops**

- Introductory workshops for research groups
- Advanced modeling techniques
- Domain-specific applications
- Best practices for scientific computing

**Commercial Licensing**

For commercial applications requiring additional support or licensing terms.

Self-Help Resources
------------------

**Debugging Your Models**

.. code-block:: python

   # Minimal debugging pattern
   prior_draws = model.draw(10)
   # Run short MCMC
   quick = model.mcmc(chains=2, iter_sampling=100, iter_warmup=100)
   quick.diagnose()

**Common Error Messages and Solutions**

Removed references to `model.get_stan_code()` (not implemented). Stan code is generated internally; if compilation fails, reduce model complexity and check parameter shapes.

...existing code...

**Performance Optimization**

.. code-block:: python

   # Use mle first for a quick sanity check
   mle_res = model.mle(epochs=5000, early_stop=5)

   # Then run full MCMC only after validating basics
   mcmc_res = model.mcmc(chains=4, iter_sampling=1000, iter_warmup=500)

Community Guidelines
-------------------

**Be Respectful**

- Treat all community members with respect
- Use inclusive language
- Be patient with beginners

**Search Before Asking**

- Check documentation first
- Search existing discussions and issues
- Look for similar questions on Stack Overflow

**Provide Value**

- Share solutions when you find them
- Contribute examples and tutorials
- Help others with questions you can answer

**Follow Up**

- Mark questions as resolved when solved
- Update posts if you find additional information
- Thank those who help you

Getting Involved
---------------

**Contributing to Documentation**

- Fix typos and unclear explanations
- Add examples for your scientific domain
- Translate documentation to other languages
- Create tutorial notebooks

**Code Contributions**

- Fix bugs and improve performance
- Add new distributions and transformations
- Improve error messages and validation
- Write tests for edge cases

**Community Building**

- Share your models and applications
- Write blog posts about your experiences
- Present at conferences and meetups
- Mentor new users

Note:
    Documentation formerly mentioning functions like `model.sample`, `model.variational`,
    `model.validate`, `model.summary`, or `model.posterior_predictive` has been updated
    to reflect only currently implemented APIs.

Remember: The SciStanPy community is here to help you succeed in your scientific modeling endeavors. Don't hesitate to ask questions and share your experiences!
- Mark questions as resolved when solved
- Update posts if you find additional information
- Thank those who help you

Getting Involved
---------------

**Contributing to Documentation**

- Fix typos and unclear explanations
- Add examples for your scientific domain
- Translate documentation to other languages
- Create tutorial notebooks

**Code Contributions**

- Fix bugs and improve performance
- Add new distributions and transformations
- Improve error messages and validation
- Write tests for edge cases

**Community Building**

- Share your models and applications
- Write blog posts about your experiences
- Present at conferences and meetups
- Mentor new users

Remember: The SciStanPy community is here to help you succeed in your scientific modeling endeavors. Don't hesitate to ask questions and share your experiences!
