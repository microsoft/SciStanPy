Regression Modeling Examples
============================

This section demonstrates various regression approaches for scientific data analysis using SciStanPy.

Example 1: Linear Regression with Measurement Error
---------------------------------------------------

When both x and y variables have measurement uncertainty, ordinary least squares can be biased.

**Scientific Context**

Calibrating a spectrophotometer where both concentration and absorbance measurements have known uncertainties.

**Implementation**

.. code-block:: python

   import scistanpy as ssp
   import numpy as np
   import matplotlib.pyplot as plt

   # Experimental data with known measurement uncertainties
   true_concentrations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # mM
   concentration_errors = np.array([0.05, 0.05, 0.1, 0.1, 0.15])  # mM

   observed_concentrations = true_concentrations + np.random.normal(0, concentration_errors)
   observed_absorbances = np.array([0.12, 0.31, 0.48, 0.67, 0.82])  # AU
   absorbance_errors = np.array([0.01, 0.015, 0.02, 0.025, 0.03])  # AU

   # Model true concentrations as latent variables
   true_conc = ssp.parameters.Normal(
       mu=observed_concentrations,
       sigma=concentration_errors
   )

   # Beer's law: A = ε * c * l (assuming l=1 cm)
   extinction_coeff = ssp.parameters.Normal(mu=0.15, sigma=0.05)  # AU/mM
   intercept = ssp.parameters.Normal(mu=0, sigma=0.02)  # Background

   # Predicted absorbances
   predicted_abs = intercept + extinction_coeff * true_conc

   # Measurement noise
   measurement_noise = ssp.parameters.LogNormal(mu=np.log(0.01), sigma=0.3)

   # Combined uncertainty
   total_sigma = ssp.operations.sqrt(absorbance_errors**2 + measurement_noise**2)

   # Likelihood
   likelihood = ssp.parameters.Normal(mu=predicted_abs, sigma=total_sigma)
   likelihood.observe(observed_absorbances)

   # Inference
   model = ssp.Model(likelihood)
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=2000)

   # Results
   print(f"Extinction coefficient: {results['extinction_coeff'].mean():.4f} ± {results['extinction_coeff'].std():.4f} AU/mM")
   print(f"Intercept: {results['intercept'].mean():.4f} ± {results['intercept'].std():.4f} AU")

Example 2: Polynomial Regression with Model Selection
-----------------------------------------------------

Determining the appropriate polynomial degree for a dose-response relationship.

**Scientific Context**

Analyzing dose-response data where the relationship might be linear, quadratic, or cubic.

**Implementation**

.. code-block:: python

   # Dose-response data
   doses = np.array([0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])  # μM
   responses = np.array([0.05, 0.12, 0.45, 0.78, 1.2, 1.8, 2.1])  # Response units
   response_errors = np.full_like(responses, 0.1)

   def create_polynomial_model(degree):
       """Create polynomial model of specified degree."""

       # Standardize doses for numerical stability
       dose_mean = doses.mean()
       dose_std = doses.std()
       doses_std = (doses - dose_mean) / dose_std

       # Polynomial coefficients
       coefficients = ssp.parameters.Normal(mu=0, sigma=1, shape=(degree + 1,))

       # Polynomial prediction
       prediction = coefficients[0]  # Intercept
       for i in range(1, degree + 1):
           prediction = prediction + coefficients[i] * (doses_std ** i)

       # Observation noise
       sigma = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.3)

       # Likelihood
       likelihood = ssp.parameters.Normal(mu=prediction, sigma=sigma)
       likelihood.observe(responses)

       return ssp.Model(likelihood)

   # Compare models of different degrees
   models = {}
   results = {}
   loos = {}

   for degree in [1, 2, 3, 4]:
       print(f"Fitting degree {degree} polynomial...")
       models[degree] = create_polynomial_model(degree)
       results[degree] = models[degree].mcmc(chains=4, iter_warmup=500, iter_sampling=1000)
       # LOO not implemented; retain results for manual comparison.

   # Model comparison
   print("\nModel Comparison (LOO-CV):")
   for degree in [1, 2, 3, 4]:
       elpd = loos[degree]['elpd_loo']
       se = loos[degree]['se_elpd_loo']
       print(f"Degree {degree}: ELPD = {elpd:.1f} ± {se:.1f}")

   # Select best model
   best_degree = max(loos.keys(), key=lambda d: loos[d]['elpd_loo'])
   print(f"\nBest model: Degree {best_degree}")

Example 3: Robust Regression with Outliers
------------------------------------------

Using Student's t distribution to handle outliers in the data.

**Scientific Context**

Analyzing reaction kinetics data where some measurements may be outliers due to experimental errors.

**Implementation**

.. code-block:: python

   # Kinetics data with potential outliers
   time_points = np.array([0, 5, 10, 15, 20, 30, 45, 60])  # minutes
   concentrations = np.array([10.0, 8.1, 6.8, 5.2, 4.1, 2.8, 1.7, 1.2])  # mM

   # One outlier at t=15 min
   concentrations[3] = 7.5  # Should be ~5.2, but measured as 7.5

   # Standard (non-robust) model
   def normal_regression():
       # First-order decay: C(t) = C0 * exp(-k*t)
       log_C0 = ssp.parameters.Normal(mu=np.log(10), sigma=0.2)
       k = ssp.parameters.LogNormal(mu=np.log(0.03), sigma=0.5)
       sigma = ssp.parameters.LogNormal(mu=np.log(0.2), sigma=0.3)

       C0 = ssp.operations.exp(log_C0)
       predicted = C0 * ssp.operations.exp(-k * time_points)

       likelihood = ssp.parameters.Normal(mu=predicted, sigma=sigma)
       likelihood.observe(concentrations)

       return ssp.Model(likelihood)

   # Robust model with Student's t
   def robust_regression():
       # Same model structure but with t-distribution
       log_C0 = ssp.parameters.Normal(mu=np.log(10), sigma=0.2)
       k = ssp.parameters.LogNormal(mu=np.log(0.03), sigma=0.5)
       sigma = ssp.parameters.LogNormal(mu=np.log(0.2), sigma=0.3)
       nu = ssp.parameters.Exponential(rate=0.1)  # Degrees of freedom

       C0 = ssp.operations.exp(log_C0)
       predicted = C0 * ssp.operations.exp(-k * time_points)

       likelihood = ssp.parameters.StudentT(nu=nu, mu=predicted, sigma=sigma)
       likelihood.observe(concentrations)

       return ssp.Model(likelihood)

   # Fit both models
   normal_model = normal_regression()
   robust_model = robust_regression()

   normal_results = normal_model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)
   robust_results = robust_model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)

   # Compare parameter estimates
   print("Parameter Estimates:")
   print(f"Normal model - k: {normal_results['k'].mean():.4f} ± {normal_results['k'].std():.4f}")
   print(f"Robust model - k: {robust_results['k'].mean():.4f} ± {robust_results['k'].std():.4f}")

   # Information criteria helpers not implemented; compare via domain criteria
   print("\nModel comparison via robustness (qualitative).")

Example 4: Hierarchical Regression
----------------------------------

Modeling dose-response relationships across multiple experimental conditions.

**Scientific Context**

Drug dose-response curves measured under different pH conditions, where we expect similar shapes but different parameters.

**Implementation**

.. code-block:: python

   # Multi-condition dose-response data
   n_conditions = 4
   pH_values = np.array([6.0, 6.5, 7.0, 7.5])

   # Doses (same for all conditions)
   doses = np.array([0.01, 0.1, 1.0, 10.0, 100.0])  # μM
   log_doses = np.log10(doses)

   # Simulated responses (EC50 varies with pH)
   np.random.seed(42)
   true_responses = {}
   for i, pH in enumerate(pH_values):
       # EC50 increases with pH
       EC50 = 1.0 * (10 ** (0.3 * (pH - 7.0)))
       Hill = 1.2
       Emax = 100

       responses = Emax / (1 + (EC50 / doses) ** Hill)
       responses += np.random.normal(0, 5, len(responses))  # Add noise
       true_responses[i] = responses

   # Hierarchical dose-response model
   # Global parameters (across all conditions)
   global_log_EC50 = ssp.parameters.Normal(mu=0, sigma=1)  # log10(EC50)
   global_Hill = ssp.parameters.Normal(mu=1, sigma=0.5)
   global_Emax = ssp.parameters.Normal(mu=100, sigma=20)

   # Variability parameters
   sigma_EC50 = ssp.parameters.LogNormal(mu=np.log(0.3), sigma=0.5)
   sigma_Hill = ssp.parameters.LogNormal(mu=np.log(0.2), sigma=0.5)
   sigma_Emax = ssp.parameters.LogNormal(mu=np.log(10), sigma=0.5)

   # Condition-specific parameters
   condition_log_EC50 = ssp.parameters.Normal(
       mu=global_log_EC50,
       sigma=sigma_EC50,
       shape=(n_conditions,)
   )
   condition_Hill = ssp.parameters.Normal(
       mu=global_Hill,
       sigma=sigma_Hill,
       shape=(n_conditions,)
   )
   condition_Emax = ssp.parameters.Normal(
       mu=global_Emax,
       sigma=sigma_Emax,
       shape=(n_conditions,)
   )

   # Observation noise
   obs_sigma = ssp.parameters.LogNormal(mu=np.log(5), sigma=0.3)

   # Likelihood for each condition
   likelihoods = []
   for i in range(n_conditions):
       # Hill equation
       EC50_i = 10 ** condition_log_EC50[i]
       predicted_i = condition_Emax[i] / (
           1 + (EC50_i / doses) ** condition_Hill[i]
       )

       likelihood_i = ssp.parameters.Normal(mu=predicted_i, sigma=obs_sigma)
       likelihood_i.observe(true_responses[i])
       likelihoods.append(likelihood_i)

   # Combined model
   model = ssp.Model(likelihoods)
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=2000)

   # Results
   print("Population parameters:")
   print(f"Global log EC50: {results['global_log_EC50'].mean():.2f} ± {results['global_log_EC50'].std():.2f}")
   print(f"Global Hill: {results['global_Hill'].mean():.2f} ± {results['global_Hill'].std():.2f}")
   print(f"Global Emax: {results['global_Emax'].mean():.1f} ± {results['global_Emax'].std():.1f}")

   print("\nCondition-specific EC50 values:")
   for i, pH in enumerate(pH_values):
       ec50_samples = 10 ** results['condition_log_EC50'][:, i]
       print(f"pH {pH}: {ec50_samples.mean():.2f} ± {ec50_samples.std():.2f} μM")

Example 5: Nonlinear Regression with Custom Functions
-----------------------------------------------------

Fitting complex mechanistic models with custom mathematical functions.

**Scientific Context**

Enzyme inhibition kinetics following a competitive inhibition model.

**Implementation**

.. code-block:: python

   # Competitive inhibition data
   substrate_concs = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])  # mM
   inhibitor_concs = np.array([0, 0.5, 1.0, 2.0])  # mM

   # Measured velocities (synthetic data)
   np.random.seed(123)
   true_Vmax = 10.0  # μM/min
   true_Km = 1.0     # mM
   true_Ki = 0.8     # mM

   velocities = {}
   for I in inhibitor_concs:
       # Competitive inhibition equation
       apparent_Km = true_Km * (1 + I / true_Ki)
       v = true_Vmax * substrate_concs / (apparent_Km + substrate_concs)
       v += np.random.normal(0, 0.3, len(v))  # Add noise
       velocities[I] = v

   # Custom transformation for competitive inhibition
   class CompetitiveInhibition(ssp.operations.CustomTransformation):
       """Competitive inhibition kinetics."""

       def __init__(self, substrate, inhibitor, Vmax, Km, Ki):
           self.substrate = substrate
           self.inhibitor = inhibitor
           super().__init__(Vmax=Vmax, Km=Km, Ki=Ki)

       def run_np_torch_op(self, Vmax, Km, Ki):
           # Apparent Km increases with inhibitor
           apparent_Km = Km * (1 + self.inhibitor / Ki)
           return Vmax * self.substrate / (apparent_Km + self.substrate)

       def write_stan_operation(self, Vmax: str, Km: str, Ki: str) -> str:
           return f"({Vmax} .* substrate) ./ (({Km} .* (1 + inhibitor ./ {Ki})) + substrate)"

   # Model parameters
   Vmax = ssp.parameters.LogNormal(mu=np.log(10), sigma=0.3)
   Km = ssp.parameters.LogNormal(mu=np.log(1), sigma=0.5)
   Ki = ssp.parameters.LogNormal(mu=np.log(1), sigma=0.5)
   sigma = ssp.parameters.LogNormal(mu=np.log(0.3), sigma=0.3)

   # Create model for all conditions
   all_predictions = []
   all_observations = []

   for I in inhibitor_concs:
       # Predicted velocities for this inhibitor concentration
       predicted = CompetitiveInhibition(
           substrate=substrate_concs,
           inhibitor=I,
           Vmax=Vmax,
           Km=Km,
           Ki=Ki
       )
       all_predictions.extend([predicted[i] for i in range(len(substrate_concs))])
       all_observations.extend(velocities[I])

   # Combined likelihood
   likelihood = ssp.parameters.Normal(
       mu=all_predictions,
       sigma=sigma
   )
   likelihood.observe(all_observations)

   # Inference
   model = ssp.Model(likelihood)
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=2000)

   # Results
   print("Kinetic parameters:")
   print(f"Vmax: {results['Vmax'].mean():.2f} ± {results['Vmax'].std():.2f} μM/min")
   print(f"Km: {results['Km'].mean():.2f} ± {results['Km'].std():.2f} mM")
   print(f"Ki: {results['Ki'].mean():.2f} ± {results['Ki'].std():.2f} mM")

   # Compare to true values
   print(f"\nTrue values: Vmax={true_Vmax}, Km={true_Km}, Ki={true_Ki}")

Regression Diagnostics and Validation
-------------------------------------
.. note::
   posterior_predictive() not implemented; residual & fit checks require
   custom simulation using parameter draws from results.

Best Practices for Scientific Regression
----------------------------------------

1. **Account for measurement error** in both predictors and responses when known
2. **Use appropriate error distributions** (normal, t, lognormal) based on data characteristics
3. **Consider hierarchical structure** when data has natural groupings
4. **Validate model assumptions** using posterior predictive checks and residual analysis
5. **Compare models** using information criteria when uncertain about functional form
6. **Use robust methods** when outliers are suspected
7. **Transform variables** when necessary to meet model assumptions
8. **Report uncertainty** in all parameter estimates and predictions

Accuracy Note
-------------
Removed unsupported: model.sample, loo, posterior_predictive. Use model.mcmc
and custom posterior simulation instead.

These regression examples demonstrate the flexibility of SciStanPy for handling complex scientific data with appropriate uncertainty quantification.
       # 1. Residuals vs fitted
       axes[0, 0].scatter(post_pred.mean(axis=0), residuals)
       axes[0, 0].axhline(0, color='red', linestyle='--')
       axes[0, 0].set_xlabel('Fitted values')
       axes[0, 0].set_ylabel('Residuals')
       axes[0, 0].set_title('Residuals vs Fitted')

       # 2. Q-Q plot of residuals
       from scipy import stats
       stats.probplot(residuals, dist="norm", plot=axes[0, 1])
       axes[0, 1].set_title('Q-Q Plot of Residuals')

       # 3. Residuals vs predictor
       axes[1, 0].scatter(x_data, residuals)
       axes[1, 0].axhline(0, color='red', linestyle='--')
       axes[1, 0].set_xlabel('Predictor')
       axes[1, 0].set_ylabel('Residuals')
       axes[1, 0].set_title('Residuals vs Predictor')

       # 4. Posterior predictive check
       axes[1, 1].hist(post_pred.flatten(), alpha=0.7, density=True,
                       label='Posterior predictive')
       axes[1, 1].hist(y_data, alpha=0.7, density=True, label='Observed')
       axes[1, 1].set_xlabel('Response')
       axes[1, 1].set_ylabel('Density')
       axes[1, 1].set_title('Posterior Predictive Check')
       axes[1, 1].legend()

       plt.tight_layout()
       return fig

   # Example usage
   # fig = regression_diagnostics(model, results, doses, responses)

Best Practices for Scientific Regression
----------------------------------------

1. **Account for measurement error** in both predictors and responses when known
2. **Use appropriate error distributions** (normal, t, lognormal) based on data characteristics
3. **Consider hierarchical structure** when data has natural groupings
4. **Validate model assumptions** using posterior predictive checks and residual analysis
5. **Compare models** using information criteria when uncertain about functional form
6. **Use robust methods** when outliers are suspected
7. **Transform variables** when necessary to meet model assumptions
8. **Report uncertainty** in all parameter estimates and predictions

These regression examples demonstrate the flexibility of SciStanPy for handling complex scientific data with appropriate uncertainty quantification.
       axes[1, 1].legend()

       plt.tight_layout()
       return fig

This comprehensive collection of regression examples demonstrates the flexibility and power of SciStanPy for scientific data analysis across various modeling scenarios.
