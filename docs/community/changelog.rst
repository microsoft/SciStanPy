Changelog
=========

All notable changes to SciStanPy are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~
- Initial advanced documentation structure (performance, extension guides)
- Additional diagnostic reporting refinements in SampleResults
- Utility enhancements (stable sigmoid, chunk shape helper)

Changed
~~~~~~~
- Clarified documentation to reflect currently implemented APIs only
- Improved error handling messages in utilities and results diagnostics

Fixed
~~~~~
- Documentation references to non-existent methods (`variational`, `model.sample`, `model.validate`)
- Minor consistency issues in prior predictive and diagnostics examples

[1.0.0] - 2024-XX-XX
--------------------
Added
~~~~~
**Core Features**
- Model specification via Python components
- Automatic Stan code generation + HMC sampling (`Model.mcmc`)
- Maximum Likelihood estimation (`Model.mle`)
- Prior sampling (`Model.draw`) and interactive prior predictive (`Model.prior_predictive`)
- Diagnostics (`SampleResults.diagnose`, sample/variable test evaluation)

**Distributions & Transformations**
- Foundational continuous / discrete distributions and transformation operators (as implemented in parameters & transformation modules)

**Utilities**
- Lazy import helpers, stable sigmoid, Dask context manager, chunk shape calculation, Spearman correlation optimization

(Items such as variational inference, WAIC/LOO interfaces, model saving/loading, GPU-specific acceleration are future roadmap items and not part of this release.)

Changed
~~~~~~~
- Initial public release

Removed
~~~~~~~
- Prior draft claims of unsupported features (VI, WAIC, LOO helper methods)

Development History
------------------

Pre-1.0 Development
~~~~~~~~~~~~~~~~~~

**2024-Q3: Beta Development**
- Core architecture implementation
- Distribution library development
- Transformation system design
- Stan integration and code generation
- Initial documentation and examples

**2024-Q2: Alpha Development**
- Project conception and design
- Prototype development
- User interface design
- Scientific domain research
- Community feedback integration

**2024-Q1: Planning Phase**
- Requirements gathering from scientific community
- Technical architecture planning
- Stakeholder interviews and feedback
- Competitive analysis and positioning

Migration Guide
---------------

**From Other Bayesian Libraries**

If you're migrating from PyMC or other Bayesian libraries:

.. code-block:: python

   # PyMC style
   with pm.Model() as model:
       mu = pm.Normal('mu', mu=0, sigma=1)
       sigma = pm.HalfNormal('sigma', sigma=1)
       obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)

   # SciStanPy equivalent
   mu = ssp.parameters.Normal(mu=0, sigma=1)
   sigma = ssp.parameters.LogNormal(mu=0, sigma=1)
   likelihood = ssp.parameters.Normal(mu=mu, sigma=sigma)
   likelihood.observe(data)
   model = ssp.Model(likelihood)

**From Stan**

If you're migrating from Stan:

.. code-block:: stan

   // Stan code
   data {
     int<lower=0> N;
     real y[N];
   }
   parameters {
     real mu;
     real<lower=0> sigma;
   }
   model {
     mu ~ normal(0, 1);
     sigma ~ lognormal(0, 1);
     y ~ normal(mu, sigma);
   }

.. code-block:: python

   # SciStanPy equivalent
   mu = ssp.parameters.Normal(mu=0, sigma=1)
   sigma = ssp.parameters.LogNormal(mu=0, sigma=1)
   likelihood = ssp.parameters.Normal(mu=mu, sigma=sigma)
   likelihood.observe(y)
   model = ssp.Model(likelihood)

Breaking Changes
---------------

No breaking changes (initial scope defined).

Future breaking changes will be clearly documented with migration guides.

Deprecation Policy
-----------------

SciStanPy follows semantic versioning:

- **Major versions** (x.0.0): May include breaking changes
- **Minor versions** (1.x.0): New features, backward compatible
- **Patch versions** (1.0.x): Bug fixes, backward compatible

**Deprecation Timeline:**
- Features are marked deprecated for at least one minor version
- Breaking changes are introduced only in major versions
- Migration guides are provided for all breaking changes

Security Updates
---------------

Security vulnerabilities are addressed immediately:

- **Critical**: Patched within 24 hours
- **High**: Patched within 1 week
- **Medium/Low**: Included in next scheduled release

Report security issues to: security@scistanpy.org

Release Schedule
---------------

**Regular Releases:**
- **Major**: Annually (breaking changes, major features)
- **Minor**: Quarterly (new features, enhancements)
- **Patch**: As needed (bug fixes, security updates)

**Long-Term Support (LTS):**
- LTS versions are supported for 2 years
- Security updates provided for 3 years
- Version 1.0 is an LTS release

Contributors
-----------

**Core Team:**
- Lead Developer: [Name]
- Scientific Advisor: [Name]
- Documentation: [Name]
- Community Manager: [Name]

**Community Contributors:**
- [List of community contributors]
- [Domain experts and advisors]
- [Beta testers and early adopters]

**Acknowledgments:**
- Stan Development Team for the underlying inference engine
- PyTorch Team for automatic differentiation capabilities
- NumPy/SciPy communities for numerical computing foundations
- Scientific community for feedback and requirements

Roadmap
-------

**Version 1.1 (Planned - 2024-QX):**
- Enhanced time series modeling
- Improved GPU support
- Additional custom distributions
- Performance optimizations

**Version 1.2 (Planned - 2024-QX):**
- Interactive visualization tools
- Model averaging and ensemble methods
- Streaming data support
- Cloud computing integration

**Version 2.0 (Planned - 2025):**
- Next-generation user interface
- Advanced meta-learning capabilities
- Expanded domain-specific modules
- Enhanced collaboration features

See the `roadmap <https://github.com/microsoft/SciStanPy/blob/main/ROADMAP.md>`_ for detailed plans.

Getting Involved
---------------

**Ways to Contribute:**
- Report bugs and request features on GitHub
- Contribute code improvements and new features
- Add examples from your scientific domain
- Improve documentation and tutorials
- Help answer questions in community forums

**Development Process:**
- All changes go through pull request review
- Comprehensive testing required for all features
- Documentation updates required for user-facing changes
- Community input sought for major design decisions

See `CONTRIBUTING.md <https://github.com/microsoft/SciStanPy/blob/main/CONTRIBUTING.md>`_ for detailed contribution guidelines.

Note:
    Earlier draft changelog entries referencing unimplemented features (e.g. variational
    inference, automatic WAIC/LOO methods, GPU VI) have been pruned to reflect the
    actual codebase.
- Cloud computing integration

**Version 2.0 (Planned - 2025):**
- Next-generation user interface
- Advanced meta-learning capabilities
- Expanded domain-specific modules
- Enhanced collaboration features

See the `roadmap <https://github.com/microsoft/SciStanPy/blob/main/ROADMAP.md>`_ for detailed plans.

Getting Involved
---------------

**Ways to Contribute:**
- Report bugs and request features on GitHub
- Contribute code improvements and new features
- Add examples from your scientific domain
- Improve documentation and tutorials
- Help answer questions in community forums

**Development Process:**
- All changes go through pull request review
- Comprehensive testing required for all features
- Documentation updates required for user-facing changes
- Community input sought for major design decisions

See `CONTRIBUTING.md <https://github.com/microsoft/SciStanPy/blob/main/CONTRIBUTING.md>`_ for detailed contribution guidelines.
