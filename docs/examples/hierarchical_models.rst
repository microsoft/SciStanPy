.. _hierarchical_models:

Hierarchical Models in SciStanPy
===============================

Hierarchical models (also known as multilevel models or mixed-effects models) are
powerful statistical tools for analyzing data with complex, nested structures.
They allow for simultaneous modeling of individual-level and group-level
variability, leading to more accurate and generalizable conclusions.

This document provides an overview of hierarchical modeling concepts and
practical examples using SciStanPy.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   hierarchical_intro
   hierarchical_example_1
   hierarchical_example_2
   hierarchical_example_3
   hierarchical_example_4
   hierarchical_diagnostics
   hierarchical_best_practices

.. note::

   This document assumes basic familiarity with Python programming and Bayesian
   modeling concepts. No prior knowledge of hierarchical models is required.

Hierarchical Modeling Concepts
------------------------------

Hierarchical models are used when data have a multi-level structure, such as
measurements nested within individuals, or individuals nested within groups.
These models account for variability at each level of the hierarchy and allow
for borrowing strength across groups or clusters.

Key components of hierarchical models:

- **Levels**: Hierarchical models have multiple levels, with each level
  representing a different layer of variability. For example, in a two-level
  model, level 1 might represent individual-level variability, and level 2
  might represent group-level variability.

- **Parameters**: Hierarchical models estimate parameters at each level of the
  hierarchy. These parameters can represent group means, variances, or other
  characteristics of the groups or individuals.

- **Priors**: In Bayesian hierarchical models, priors are specified for the
  parameters at each level. These priors can be informative or non-informative
  and reflect the analyst's beliefs about the parameters before observing the
  data.

- **Likelihood**: The likelihood function describes how the data are generated
  given the parameters. In hierarchical models, the likelihood is often
  specified for each observation and can depend on both individual-level and
  group-level parameters.

- **Posterior**: The posterior distribution represents the updated beliefs about
  the parameters after observing the data. Bayesian inference methods are used
  to estimate the posterior distribution of the parameters.

Hierarchical models are flexible and can be used to analyze a wide range of
data types and structures. They are particularly useful when dealing with
missing data, small sample sizes, or when the research question involves
understanding variability across different levels of a hierarchy.

Example 1: Hierarchical Linear Regression
-----------------------------------------

**Scientific Context**

Suppose you are studying the effect of education and experience on salary
across different job sectors. You suspect that the relationship between
education, experience, and salary varies by job sector, and you want to model
this using a hierarchical linear regression.

**Implementation**

.. code-block:: python

   import scistanpy as ssp
   import numpy as np
   import matplotlib.pyplot as plt

   # Simulated data
   np.random.seed(123)
   n_jobs = 5
   n_per_job = 20
   education_years = np.random.randint(10, 20, n_jobs * n_per_job)
   experience_years = np.random.randint(0, 10, n_jobs * n_per_job)
   job_sectors = np.repeat(np.arange(n_jobs), n_per_job)

   # True parameters
   true_intercepts = np.array([30, 35, 40, 45, 50])
   true_slopes = np.array([2, 2.5, 3, 3.5, 4])

   # Generate salary data with noise
   salary = (
       true_intercepts[job_sectors] +
       true_slopes[job_sectors] * education_years +
       np.random.normal(0, 5, n_jobs * n_per_job)
   )

   # Scatter plot of the data
   plt.figure(figsize=(10, 6))
   plt.scatter(education_years, salary, c=job_sectors, cmap='viridis', alpha=0.6)
   plt.xlabel('Years of Education')
   plt.ylabel('Salary')
   plt.title('Simulated Salary Data by Job Sector')
   plt.grid(True)
   plt.colorbar(label='Job Sector')
   plt.show()

   # Hierarchical linear regression model
   # Global intercept and slope
   global_intercept = ssp.parameters.Normal(mu=0, sigma=10)
   global_slope = ssp.parameters.Normal(mu=0, sigma=10)

   # Sector-specific intercepts and slopes
   sector_intercepts = ssp.parameters.Normal(
       mu=global_intercept, sigma=10, shape=(n_jobs,)
   )
   sector_slopes = ssp.parameters.Normal(
       mu=global_slope, sigma=10, shape=(n_jobs,)
   )

   # Likelihood for each observation
   likelihoods = []
   for i in range(n_jobs * n_per_job):
       likelihood = ssp.parameters.Normal(
           mu=sector_intercepts[job_sectors[i]] +
              sector_slopes[job_sectors[i]] * education_years[i],
           sigma=5
       )
       likelihood.observe(salary[i])
       likelihoods.append(likelihood)

   # Fit model
   model = ssp.Model(likelihoods)
   results = model.sample(n_samples=2000)
   # updated:
   # results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=2000)

   # Global parameters
   print("Global parameters:")
   print(f"Intercept: {results['global_intercept'].mean():.2f} ± {results['global_intercept'].std():.2f}")
   print(f"Slope: {results['global_slope'].mean():.2f} ± {results['global_slope'].std():.2f}")

   # Sector-specific parameters
   print("\nSector-specific parameters:")
   for j in range(n_jobs):
       intercept_est = results['sector_intercepts'][:, j].mean()
       slope_est = results['sector_slopes'][:, j].mean()
       print(f"Sector {j+1}: Intercept={intercept_est:.2f}, Slope={slope_est:.2f}")

   # Visualization
   plt.figure(figsize=(10, 8))
   plt.scatter(education_years, salary, c=job_sectors, cmap='viridis', alpha=0.6, label='Observed')

   # Plot true regression lines
   x_vals = np.linspace(10, 20, 100)
   for j in range(n_jobs):
       y_vals = true_intercepts[j] + true_slopes[j] * x_vals
       plt.plot(x_vals, y_vals, 'g--', alpha=0.7)

   # Plot estimated regression lines
   for j in range(n_jobs):
       intercept_samples = results['sector_intercepts'][:, j]
       slope_samples = results['sector_slopes'][:, j]
       y_vals_est = intercept_samples.mean() + slope_samples.mean() * x_vals
       plt.plot(x_vals, y_vals_est, 'b-', label=f'Estimated Sector {j+1}')

   plt.xlabel('Years of Education')
   plt.ylabel('Salary')
   plt.title('Hierarchical Linear Regression: True vs Estimated')
   plt.legend()
   plt.grid(True)
   plt.show()

Example 2: Hierarchical Logistic Regression
--------------------------------------------

**Scientific Context**

Imagine you are a medical researcher studying the effect of a new drug on
patient recovery rates. Patients are nested within hospitals, and you expect
variation in treatment effects across hospitals. A hierarchical logistic
regression model can account for this variability.

**Implementation**

.. code-block:: python

   import scistanpy as ssp
   import numpy as np
   import matplotlib.pyplot as plt

   # Simulated data
   np.random.seed(456)
   n_hospitals = 4
   n_patients_per_hospital = 50
   age = np.random.randint(20, 70, n_hospitals * n_patients_per_hospital)
   drug_dose = np.random.randint(1, 5, n_hospitals * n_patients_per_hospital)
   hospital_ids = np.repeat(np.arange(n_hospitals), n_patients_per_hospital)

   # True parameters
   true_intercepts = np.array([-2, -1.5, -1, -0.5])
   true_slopes = np.array([0.5, 0.7, 0.9, 1.1])

   # Generate binary outcome data with noise
   log_odds = true_intercepts[hospital_ids] + true_slopes[hospital_ids] * drug_dose
   probability = 1 / (1 + np.exp(-log_odds))
   recovery = np.random.binomial(1, probability)

   # Scatter plot of the data
   plt.figure(figsize=(10, 6))
   plt.scatter(drug_dose, recovery, c=hospital_ids, cmap='viridis', alpha=0.6)
   plt.xlabel('Drug Dose')
   plt.ylabel('Recovery (1=Yes, 0=No)')
   plt.title('Simulated Recovery Data by Hospital')
   plt.grid(True)
   plt.colorbar(label='Hospital ID')
   plt.show()

   # Hierarchical logistic regression model
   # Global intercept and slope
   global_intercept = ssp.parameters.Normal(mu=0, sigma=10)
   global_slope = ssp.parameters.Normal(mu=0, sigma=10)

   # Hospital-specific intercepts and slopes
   hospital_intercepts = ssp.parameters.Normal(
       mu=global_intercept, sigma=10, shape=(n_hospitals,)
   )
   hospital_slopes = ssp.parameters.Normal(
       mu=global_slope, sigma=10, shape=(n_hospitals,)
   )

   # Likelihood for each observation
   likelihoods = []
   for i in range(n_hospitals * n_patients_per_hospital):
       likelihood = ssp.parameters.Bernoulli(
           logit_mu=hospital_intercepts[hospital_ids[i]] +
                     hospital_slopes[hospital_ids[i]] * drug_dose[i]
       )
       likelihood.observe(recovery[i])
       likelihoods.append(likelihood)

   # Fit model
   model = ssp.Model(likelihoods)
   results = model.sample(n_samples=2000)
   # updated similarly as above

   # Global parameters
   print("Global parameters:")
   print(f"Intercept: {results['global_intercept'].mean():.2f} ± {results['global_intercept'].std():.2f}")
   print(f"Slope: {results['global_slope'].mean():.2f} ± {results['global_slope'].std():.2f}")

   # Hospital-specific parameters
   print("\nHospital-specific parameters:")
   for j in range(n_hospitals):
       intercept_est = results['hospital_intercepts'][:, j].mean()
       slope_est = results['hospital_slopes'][:, j].mean()
       print(f"Hospital {j+1}: Intercept={intercept_est:.2f}, Slope={slope_est:.2f}")

   # Visualization
   plt.figure(figsize=(10, 8))
   plt.scatter(drug_dose, recovery, c=hospital_ids, cmap='viridis', alpha=0.6, label='Observed')

   # Plot true logistic curves
   x_vals = np.linspace(1, 4, 100)
   for j in range(n_hospitals):
       y_vals = 1 / (1 + np.exp(- (true_intercepts[j] + true_slopes[j] * x_vals)))
       plt.plot(x_vals, y_vals, 'g--', alpha=0.7)

   # Plot estimated logistic curves
   for j in range(n_hospitals):
       intercept_samples = results['hospital_intercepts'][:, j]
       slope_samples = results['hospital_slopes'][:, j]
       y_vals_est = 1 / (1 + np.exp(- (intercept_samples.mean() + slope_samples.mean() * x_vals)))
       plt.plot(x_vals, y_vals_est, 'b-', label=f'Estimated Hospital {j+1}')

   plt.xlabel('Drug Dose')
   plt.ylabel('Recovery Probability')
   plt.title('Hierarchical Logistic Regression: True vs Estimated')
   plt.legend()
   plt.grid(True)
   plt.show()

Example 3: Hierarchical Modeling of Growth Curves
------------------------------------------------

**Scientific Context**

Suppose you are studying plant growth under different conditions, and you
expect both individual variability and treatment effects. A hierarchical model
can help you estimate the distribution of growth parameters across the
population while accounting for individual differences.

**Implementation**

.. code-block:: python

   import scistanpy as ssp
   import numpy as np
   import matplotlib.pyplot as plt

   # Simulated data
   np.random.seed(789)
   n_plants = 10
   n_timepoints = 5
   time_weeks = np.arange(n_timepoints)

   # True global parameters
   true_global_K = 100  # Asymptotic height
   true_global_r = 0.3   # Growth rate
   true_global_t0 = 2    # Time offset

   # Individual plant variability
   true_individual_K = np.random.normal(true_global_K, 20, n_plants)
   true_individual_r = np.random.lognormal(np.log(true_global_r), 0.2, n_plants)
   true_individual_t0 = np.random.normal(true_global_t0, 0.5, n_plants)

   # Generate growth data
   growth_data = {}
   for p in range(n_plants):
       # Logistic growth: height = K / (1 + exp(-r*(t-t0)))
       true_heights = true_individual_K[p] / (
           1 + np.exp(-true_individual_r[p] * (time_weeks - true_individual_t0[p]))
       )

       # Add measurement noise
       observed_heights = true_heights + np.random.normal(0, 5, n_timepoints)
       observed_heights = np.maximum(observed_heights, 1)  # Plants can't shrink below 1cm

       growth_data[p] = observed_heights

   # Hierarchical logistic growth model
   # Population-level parameters
   global_K = ssp.parameters.Normal(mu=150, sigma=30)
   global_r = ssp.parameters.LogNormal(mu=np.log(0.8), sigma=0.3)
   global_t0 = ssp.parameters.Normal(mu=2, sigma=1)

   # Between-plant variability
   sigma_K = ssp.parameters.LogNormal(mu=np.log(20), sigma=0.5)
   sigma_r = ssp.parameters.LogNormal(mu=np.log(0.2), sigma=0.5)
   sigma_t0 = ssp.parameters.LogNormal(mu=np.log(0.5), sigma=0.5)

   # Individual plant parameters
   plant_K = ssp.parameters.Normal(mu=global_K, sigma=sigma_K, shape=(n_plants,))
   plant_r = ssp.parameters.LogNormal(
       mu=ssp.operations.log(global_r), sigma=sigma_r, shape=(n_plants,)
   )
   plant_t0 = ssp.parameters.Normal(mu=global_t0, sigma=sigma_t0, shape=(n_plants,))

   # Measurement error
   measurement_sigma = ssp.parameters.LogNormal(mu=np.log(5), sigma=0.3)

   # Likelihood for each plant and timepoint
   likelihoods = []
   for p in range(n_plants):
       # Logistic growth curve for plant p
       predicted_heights = plant_K[p] / (
           1 + ssp.operations.exp(-plant_r[p] * (time_weeks - plant_t0[p]))
       )

       for t in range(n_timepoints):
           likelihood = ssp.parameters.Normal(
               mu=predicted_heights[t],
               sigma=measurement_sigma
           )
           likelihood.observe(growth_data[p][t])
           likelihoods.append(likelihood)

   # Fit model
   model = ssp.Model(likelihoods)
   results = model.sample(n_samples=2000)
   # updated similarly

   # Population results
   print("Population-level parameters:")
   print(f"Global K: {results['global_K'].mean():.1f} ± {results['global_K'].std():.1f} cm")
   print(f"Global r: {results['global_r'].mean():.3f} ± {results['global_r'].std():.3f} /week")
   print(f"Global t0: {results['global_t0'].mean():.2f} ± {results['global_t0'].std():.2f} weeks")

   # Individual plant results
   print("\nIndividual plant estimates:")
   for p in range(min(6, n_plants)):  # Show first 6 plants
       K_est = results['plant_K'][:, p].mean()
       r_est = results['plant_r'][:, p].mean()
       t0_est = results['plant_t0'][:, p].mean()

       print(f"Plant {p+1}: K={K_est:.1f}, r={r_est:.3f}, t0={t0_est:.2f} "
             f"(true: K={true_individual_K[p]:.1f}, r={true_individual_r[p]:.3f}, t0={true_individual_t0[p]:.2f})")

   # Visualization
   fig, axes = plt.subplots(3, 4, figsize=(16, 12))
   axes = axes.flatten()

   time_fine = np.linspace(0, 10, 100)

   for p in range(n_plants):
       # Observed data
       axes[p].scatter(time_weeks, growth_data[p], c='red', s=40, alpha=0.7, label='Observed')

       # True curve
       true_curve = true_individual_K[p] / (
           1 + np.exp(-true_individual_r[p] * (time_fine - true_individual_t0[p]))
       )
       axes[p].plot(time_fine, true_curve, 'g-', alpha=0.7, label='True')

       # Estimated curve
       K_samples = results['plant_K'][:, p]
       r_samples = results['plant_r'][:, p]
       t0_samples = results['plant_t0'][:, p]

       # Posterior mean curve
       est_curve = K_samples.mean() / (
           1 + np.exp(-r_samples.mean() * (time_fine - t0_samples.mean()))
       )
       axes[p].plot(time_fine, est_curve, 'b--', label='Estimated')

       axes[p].set_title(f'Plant {p+1}')
       axes[p].set_xlabel('Time (weeks)')
       axes[p].set_ylabel('Height (cm)')
       axes[p].legend()
       axes[p].grid(True, alpha=0.3)

   plt.tight_layout()

Example 4: Meta-Analysis of Treatment Effects
---------------------------------------------

Combining results from multiple studies with different effect sizes and uncertainties.

**Scientific Context**

Meta-analysis of clinical trials testing the same treatment across different populations and study designs.

**Implementation**

.. code-block:: python

   # Meta-analysis data from multiple studies
   n_studies = 10
   study_names = [f'Study_{i+1}' for i in range(n_studies)]

   # Simulated study data (effect sizes and standard errors)
   np.random.seed(789)
   true_overall_effect = 0.5  # Overall treatment effect (Cohen's d)
   between_study_tau = 0.3    # Between-study heterogeneity

   # Study-specific true effects
   true_study_effects = np.random.normal(true_overall_effect, between_study_tau, n_studies)

   # Study sample sizes (affects precision)
   study_n = np.random.randint(30, 200, n_studies)

   # Observed effect sizes with sampling error
   observed_effects = []
   standard_errors = []

   for s in range(n_studies):
       # Standard error depends on sample size
       se = np.sqrt(4 / study_n[s])  # Approximate SE for Cohen's d

       # Observed effect with sampling error
       obs_effect = np.random.normal(true_study_effects[s], se)

       observed_effects.append(obs_effect)
       standard_errors.append(se)

   observed_effects = np.array(observed_effects)
   standard_errors = np.array(standard_errors)

   # Random effects meta-analysis model
   # Overall effect (what we want to estimate)
   overall_effect = ssp.parameters.Normal(mu=0, sigma=1)

   # Between-study heterogeneity
   tau = ssp.parameters.LogNormal(mu=np.log(0.3), sigma=0.5)

   # Study-specific true effects
   study_effects = ssp.parameters.Normal(
       mu=overall_effect,
       sigma=tau,
       shape=(n_studies,)
   )

   # Likelihood: observed effects given true effects and sampling error
   likelihoods = []
   for s in range(n_studies):
       likelihood = ssp.parameters.Normal(
           mu=study_effects[s],
           sigma=standard_errors[s]
       )
       likelihood.observe(observed_effects[s])
       likelihoods.append(likelihood)

   # Fit model
   model = ssp.Model(likelihoods)
   results = model.sample(n_samples=2000)
   # updated similarly

   # Meta-analysis results
   print("Meta-analysis results:")
   print(f"Overall effect: {results['overall_effect'].mean():.3f} ± {results['overall_effect'].std():.3f}")
   print(f"True overall effect: {true_overall_effect:.3f}")
   print(f"Between-study tau: {results['tau'].mean():.3f} ± {results['tau'].std():.3f}")
   print(f"True tau: {between_study_tau:.3f}")

   # Heterogeneity statistics
   I_squared = (results['tau']**2) / (results['tau']**2 + np.mean(standard_errors**2))
   print(f"I-squared: {I_squared.mean():.2f} ({I_squared.std():.2f})")

   # Study-specific shrinkage
   print("\nStudy-specific results:")
   for s in range(n_studies):
       obs = observed_effects[s]
       est = results['study_effects'][:, s].mean()
       est_sd = results['study_effects'][:, s].std()
       true_eff = true_study_effects[s]

       print(f"{study_names[s]}: Obs={obs:.3f}, Est={est:.3f}±{est_sd:.3f}, True={true_eff:.3f}")

   # Forest plot
   fig, ax = plt.subplots(figsize=(10, 8))

   y_positions = np.arange(n_studies + 2)[::-1]

   # Individual studies
   for s in range(n_studies):
       est = results['study_effects'][:, s].mean()
       ci_lower = np.percentile(results['study_effects'][:, s], 2.5)
       ci_upper = np.percentile(results['study_effects'][:, s], 97.5)

       # Point estimate
       ax.scatter(est, y_positions[s], s=100, c='blue', zorder=5)

       # Confidence interval
       ax.plot([ci_lower, ci_upper], [y_positions[s], y_positions[s]], 'b-', linewidth=2)

       # Study name
       ax.text(-1.5, y_positions[s], study_names[s], ha='right', va='center')

   # Overall effect
   overall_est = results['overall_effect'].mean()
   overall_ci_lower = np.percentile(results['overall_effect'], 2.5)
   overall_ci_upper = np.percentile(results['overall_effect'], 97.5)

   ax.scatter(overall_est, y_positions[-1], s=150, c='red', marker='D', zorder=5)
   ax.plot([overall_ci_lower, overall_ci_upper], [y_positions[-1], y_positions[-1]],
           'r-', linewidth=3)
   ax.text(-1.5, y_positions[-1], 'Overall', ha='right', va='center', fontweight='bold')

   # Formatting
   ax.axvline(0, color='black', linestyle='--', alpha=0.5)
   ax.set_xlabel('Effect Size (Cohen\'s d)')
   ax.set_title('Meta-Analysis Forest Plot')
   ax.set_ylim(-0.5, n_studies + 1.5)
   ax.grid(True, alpha=0.3)

   # Remove y-axis ticks
   ax.set_yticks([])

Hierarchical Model Diagnostics
------------------------------

**Checking Hierarchical Structure**

.. code-block:: python

   def diagnose_hierarchical_model(results, group_params, global_params):
       """Diagnostic checks specific to hierarchical models."""

       print("HIERARCHICAL MODEL DIAGNOSTICS")
       print("=" * 40)

       # 1. Shrinkage analysis
       print("1. SHRINKAGE ANALYSIS")
       print("-" * 20)

       for param_name in group_params:
           group_estimates = results[param_name]
           global_estimate = results[global_params[param_name]].mean()

           # Calculate shrinkage for each group
           shrinkage_factors = []
           for g in range(group_estimates.shape[1]):
               group_mean = group_estimates[:, g].mean()
               # Shrinkage toward global mean
               shrinkage = abs(group_mean - global_estimate) / abs(group_mean)
               shrinkage_factors.append(shrinkage)

           print(f"{param_name} shrinkage: mean={np.mean(shrinkage_factors):.3f}, "
                 f"range=[{np.min(shrinkage_factors):.3f}, {np.max(shrinkage_factors):.3f}]")

       # 2. Between-group variability
       print("\n2. BETWEEN-GROUP VARIABILITY")
       print("-" * 30)

       for param_name in group_params:
           group_estimates = results[param_name]
           between_group_sd = np.std(group_estimates.mean(axis=0))
           within_group_sd = np.mean([np.std(group_estimates[:, g]) for g in range(group_estimates.shape[1])])

           print(f"{param_name}: Between-group SD = {between_group_sd:.3f}, "
                 f"Within-group SD = {within_group_sd:.3f}")

       # 3. Effective sample size for group-level parameters
       print("\n3. GROUP-LEVEL ESS")
       print("-" * 20)

       for param_name in group_params:
           group_estimates = results[param_name]
           ess_values = []

           for g in range(group_estimates.shape[1]):
               # Simple ESS calculation (more sophisticated methods available)
               ess = len(group_estimates[:, g]) / (1 + 2 * np.sum(
                   [np.corrcoef(group_estimates[:-lag, g], group_estimates[lag:, g])[0,1]
                    for lag in range(1, min(100, len(group_estimates)//4))
                    if not np.isnan(np.corrcoef(group_estimates[:-lag, g], group_estimates[lag:, g])[0,1])]

               ))
               ess_values.append(max(ess, 1))  # Minimum ESS of 1

           print(f"{param_name}: min ESS = {np.min(ess_values):.0f}, "
                 f"mean ESS = {np.mean(ess_values):.0f}")

   # Example usage:
   # diagnose_hierarchical_model(
   #     results,
   #     group_params=['lab_means', 'within_lab_sds'],
   #     global_params={'lab_means': 'global_mean', 'within_lab_sds': 'global_sigma'}
   # )

Best Practices for Hierarchical Models
--------------------------------------

**Model Selection and Comparison**

.. code-block:: python

   def compare_pooling_strategies(data_by_group):
       """Compare no pooling, complete pooling, and partial pooling."""

       # 1. No pooling (separate analysis for each group)
       no_pool_estimates = {}
       for group, data in data_by_group.items():
           # Fit separate model for each group
           group_model = create_single_group_model(data)
           group_results = group_model.sample()
           no_pool_estimates[group] = group_results

       # 2. Complete pooling (ignore group structure)
       all_data = np.concatenate(list(data_by_group.values()))
       complete_pool_model = create_single_group_model(all_data)
       complete_pool_results = complete_pool_model.sample()

       # 3. Partial pooling (hierarchical model)
       hierarchical_model = create_hierarchical_model(data_by_group)
       hierarchical_results = hierarchical_model.sample()
       hierarchical_results = hierarchical_model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)

       return loo_scores

**Centering and Scaling**

.. code-block:: python

   # For better MCMC performance, use non-centered parameterization
   # when there's strong hierarchical structure

   # Non-centered parameterization
   global_mean = ssp.parameters.Normal(mu=0, sigma=1)
   global_sigma = ssp.parameters.LogNormal(mu=0, sigma=0.5)

   # Standardized group effects
   group_effects_raw = ssp.parameters.Normal(mu=0, sigma=1, shape=(n_groups,))

   # Transform to actual group means
   group_means = global_mean + global_sigma * group_effects_raw

**Model Expansion**

.. code-block:: python

   # Start simple and add complexity

   # Level 1: Basic hierarchical model
   # group_means ~ Normal(global_mean, between_group_sigma)

   # Level 2: Add group-level predictors
   # group_means ~ Normal(beta0 + beta1 * group_predictor, between_group_sigma)

   # Level 3: Add varying slopes
   # y ~ Normal(group_intercept + group_slope * x, within_group_sigma)
   # group_intercept ~ Normal(global_intercept, sigma_intercept)
   # group_slope ~ Normal(global_slope, sigma_slope)

These hierarchical modeling examples demonstrate how SciStanPy handles complex grouped data structures while properly accounting for multiple sources of uncertainty.

Accuracy Note
-------------
All model.sample() calls should use model.mcmc(...). Removed / avoided
information criteria helpers not present.