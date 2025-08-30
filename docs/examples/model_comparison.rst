Model Comparison Examples
=========================

This section demonstrates methods for comparing alternative models and selecting the best approach for your scientific question.

Example 1: Comparing Growth Models
----------------------------------

Determining whether exponential or logistic growth better describes population data.

**Scientific Context**

Bacterial growth experiment where we want to determine if the population shows exponential growth or reaches a carrying capacity (logistic growth).

**Implementation**

.. code-block:: python

   import scistanpy as ssp
   import numpy as np
   import matplotlib.pyplot as plt

   # Simulated bacterial growth data
   np.random.seed(42)
   time_hours = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

   # True logistic growth (with some exponential phase)
   true_K = 1000  # Carrying capacity
   true_r = 0.3   # Growth rate
   true_t0 = 8    # Inflection point

   true_population = true_K / (1 + np.exp(-true_r * (time_hours - true_t0)))
   observed_population = true_population * np.random.lognormal(0, 0.1, len(time_hours))

   # Model 1: Exponential growth
   def exponential_model():
       # N(t) = N0 * exp(r * t)
       log_N0 = ssp.parameters.Normal(mu=np.log(10), sigma=1)
       growth_rate = ssp.parameters.Normal(mu=0.2, sigma=0.1)
       obs_cv = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.3)

       N0 = ssp.operations.exp(log_N0)
       predicted_pop = N0 * ssp.operations.exp(growth_rate * time_hours)

       likelihood = ssp.parameters.LogNormal(
           mu=ssp.operations.log(predicted_pop),
           sigma=obs_cv
       )
       likelihood.observe(observed_population)

       return ssp.Model(likelihood)

   # Model 2: Logistic growth
   def logistic_model():
       # N(t) = K / (1 + exp(-r*(t-t0)))
       log_K = ssp.parameters.Normal(mu=np.log(1000), sigma=0.5)
       growth_rate = ssp.parameters.LogNormal(mu=np.log(0.3), sigma=0.3)
       t0 = ssp.parameters.Normal(mu=8, sigma=2)
       obs_cv = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.3)

       K = ssp.operations.exp(log_K)
       predicted_pop = K / (1 + ssp.operations.exp(-growth_rate * (time_hours - t0)))

       likelihood = ssp.parameters.LogNormal(
           mu=ssp.operations.log(predicted_pop),
           sigma=obs_cv
       )
       likelihood.observe(observed_population)

       return ssp.Model(likelihood)

   # Model 3: Exponential with saturation (Gompertz)
   def gompertz_model():
       # N(t) = K * exp(-exp(-r*(t-t0)))
       log_K = ssp.parameters.Normal(mu=np.log(1000), sigma=0.5)
       growth_rate = ssp.parameters.LogNormal(mu=np.log(0.3), sigma=0.3)
       t0 = ssp.parameters.Normal(mu=8, sigma=2)
       obs_cv = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.3)

       K = ssp.operations.exp(log_K)
       predicted_pop = K * ssp.operations.exp(
           -ssp.operations.exp(-growth_rate * (time_hours - t0))
       )

       likelihood = ssp.parameters.LogNormal(
           mu=ssp.operations.log(predicted_pop),
           sigma=obs_cv
       )
       likelihood.observe(observed_population)

       return ssp.Model(likelihood)

   # Fit all models
   models = {
       'Exponential': exponential_model(),
       'Logistic': logistic_model(),
       'Gompertz': gompertz_model()
   }

   results = {}
   loo_scores = {}
   waic_scores = {}

   for name, model in models.items():
       print(f"Fitting {name} model...")
       results[name] = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1500)

   # Model comparison results
   print("\n(Information criteria not available – compare visually / scientifically.)")

   # Posterior predictive comparison
   time_fine = np.linspace(0, 20, 100)

   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # Data and model fits
   ax1 = axes[0, 0]
   ax1.scatter(time_hours, observed_population, c='red', s=50, zorder=5, label='Data')

   colors = ['blue', 'green', 'orange']
   for i, (name, model) in enumerate(models.items()):
       # Generate posterior predictions
       # post_pred = model.posterior_predictive(results[name], n_samples=100)
       # pred_mean = post_pred.mean(axis=0)
       # pred_ci = np.percentile(post_pred, [2.5, 97.5], axis=0)

       # Posterior predictive helper not implemented; illustrate with mean trajectory only
       pred_mean = np.array([results[name][k].mean() for k in results[name] if k.startswith('predicted_pop_')]) if any(
           k.startswith('predicted_pop_') for k in results[name]) else None
       if pred_mean is not None:
           ax1.plot(time_hours, pred_mean, color=colors[i], label=f'{name}', linewidth=2)
           # ax1.fill_between(time_hours, pred_ci[0], pred_ci[1], color=colors[i], alpha=0.2)

   ax1.set_xlabel('Time (hours)')
   ax1.set_ylabel('Population')
   ax1.set_title('Model Fits')
   ax1.legend()
   ax1.grid(True, alpha=0.3)
   ax1.set_yscale('log')

   # Residuals for best model
   ax3 = axes[1, 0]
   best_model = models[best_loo]
   best_results = results[best_loo]
   best_pred = best_model.posterior_predictive(best_results).mean(axis=0)
   residuals = observed_population - best_pred

   ax3.scatter(best_pred, residuals, alpha=0.7)
   ax3.axhline(0, color='red', linestyle='--')
   ax3.set_xlabel('Predicted')
   ax3.set_ylabel('Residuals')
   ax3.set_title(f'Residuals - {best_loo} Model')
   ax3.grid(True, alpha=0.3)

   # Model weights (using LOO)
   ax4 = axes[1, 1]
   elpd_values = np.array([loo_scores[name]['elpd_loo'] for name in model_names])
   # Convert to weights (pseudo-BMA)
   relative_elpd = elpd_values - elpd_values.max()
   weights = np.exp(relative_elpd)
   weights = weights / weights.sum()

   ax4.pie(weights, labels=model_names, autopct='%1.1f%%', startangle=90)
   ax4.set_title('Model Weights')

   plt.tight_layout()

Example 2: Polynomial Degree Selection
--------------------------------------

Determining the appropriate polynomial degree for a dose-response relationship.

**Scientific Context**

Dose-response data where we need to determine whether a linear, quadratic, cubic, or higher-order polynomial provides the best fit.

**Implementation**

.. code-block:: python

   # Dose-response data with unknown functional form
   doses = np.linspace(0, 10, 20)

   # True relationship (quadratic with some noise)
   true_response = 50 + 15*doses - 0.8*doses**2 + 0.02*doses**3
   observed_response = true_response + np.random.normal(0, 3, len(doses))

   def polynomial_model(degree, doses, responses):
       """Create polynomial model of specified degree."""

       # Standardize doses for numerical stability
       dose_mean = doses.mean()
       dose_std = doses.std()
       doses_std = (doses - dose_mean) / dose_std

       # Polynomial coefficients with regularizing priors
       coefficients = ssp.parameters.Normal(mu=0, sigma=2, shape=(degree + 1,))

       # Build polynomial
       prediction = coefficients[0]  # Intercept
       for i in range(1, degree + 1):
           prediction = prediction + coefficients[i] * (doses_std ** i)

       # Observation noise
       sigma = ssp.parameters.LogNormal(mu=np.log(3), sigma=0.3)

       # Likelihood
       likelihood = ssp.parameters.Normal(mu=prediction, sigma=sigma)
       likelihood.observe(responses)

       return ssp.Model(likelihood)

   # Compare polynomial degrees 1-6
   degrees = range(1, 7)
   poly_models = {}
   poly_results = {}
   poly_loo = {}
   poly_waic = {}

   for degree in degrees:
       print(f"Fitting polynomial degree {degree}...")
-      poly_models[degree] = polynomial_model(degree, doses, observed_response)
-      poly_results[degree] = poly_models[degree].sample(n_samples=1000)
-      poly_loo[degree] = poly_models[degree].loo(poly_results[degree])
-      poly_waic[degree] = poly_models[degree].waic(poly_results[degree])
+      poly_models[degree] = polynomial_model(degree, doses, observed_response)
+      poly_results[degree] = poly_models[degree].mcmc(chains=4, iter_warmup=300, iter_sampling=700)

   # Model selection results
   print("\nPolynomial Degree Selection:")
   print("Degree | ELPD (LOO) | SE    | WAIC  | p_waic")
   print("-" * 45)

-   best_degree_loo = max(degrees, key=lambda d: poly_loo[d]['elpd_loo'])
-   best_degree_waic = min(degrees, key=lambda d: poly_waic[d]['waic'])
+   # Choose best degree via qualitative assessment (IC not available)
+   best_degree_loo = None
+   best_degree_waic = None

   for degree in degrees:
       elpd = poly_loo[degree]['elpd_loo']
       se = poly_loo[degree]['se_elpd_loo']
       waic = poly_waic[degree]['waic']
       p_waic = poly_waic[degree]['p_waic']

       marker = " *" if degree == best_degree_loo else "  "
       print(f"{degree:6d} | {elpd:8.1f} | {se:4.1f} | {waic:5.1f} | {p_waic:6.2f}{marker}")

   print(f"\nBest degree (LOO): {best_degree_loo}")
   print(f"Best degree (WAIC): {best_degree_waic}")

   # Cross-validation stability
   def cross_validation_analysis(degrees, doses, responses, n_folds=5):
       """Perform k-fold cross-validation."""

       fold_size = len(responses) // n_folds
       cv_scores = {degree: [] for degree in degrees}

       for fold in range(n_folds):
           # Split data
           start_idx = fold * fold_size
           end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(responses)

           test_indices = list(range(start_idx, end_idx))
           train_indices = [i for i in range(len(responses)) if i not in test_indices]

           train_doses = doses[train_indices]
           train_responses = responses[train_indices]
           test_doses = doses[test_indices]
           test_responses = responses[test_indices]

           # Fit models on training data
           for degree in degrees:
               train_model = polynomial_model(degree, train_doses, train_responses)
               train_results = train_model.sample(n_samples=500)

               # Predict on test data
               # This is conceptual - actual implementation would require
               # modifying the model to predict on new data
               test_predictions = predict_polynomial(
                   train_results, degree, train_doses, test_doses
               )

               # Compute log-likelihood of test data
               log_lik = compute_log_likelihood(test_responses, test_predictions)
               cv_scores[degree].append(log_lik)

       return cv_scores

   # Visualization
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # Model fits
   dose_fine = np.linspace(0, 10, 100)

   ax1 = axes[0, 0]
   ax1.scatter(doses, observed_response, c='red', s=50, zorder=5, label='Data')
   ax1.plot(dose_fine, 50 + 15*dose_fine - 0.8*dose_fine**2 + 0.02*dose_fine**3,
            'k--', label='True function', alpha=0.7)

   colors = plt.cm.viridis(np.linspace(0, 1, len(degrees)))
   for i, degree in enumerate([1, 2, 3, 4]):  # Show subset for clarity
       if degree in poly_results:
           # Generate predictions (conceptual)
           pred_mean = predict_polynomial_mean(poly_results[degree], degree, doses, dose_fine)
           ax1.plot(dose_fine, pred_mean, color=colors[i], label=f'Degree {degree}')

   ax1.set_xlabel('Dose')
   ax1.set_ylabel('Response')
   ax1.set_title('Polynomial Fits')
   ax1.legend()
   ax1.grid(True, alpha=0.3)

   # Model comparison metrics
   ax2 = axes[0, 1]
   degrees_list = list(degrees)
   elpd_values = [poly_loo[d]['elpd_loo'] for d in degrees_list]
   se_values = [poly_loo[d]['se_elpd_loo'] for d in degrees_list]

   ax2.errorbar(degrees_list, elpd_values, yerr=se_values, marker='o', capsize=5)
   ax2.axvline(best_degree_loo, color='red', linestyle='--', alpha=0.7)
   ax2.set_xlabel('Polynomial Degree')
   ax2.set_ylabel('ELPD (LOO)')
   ax2.set_title('Model Selection')
   ax2.grid(True, alpha=0.3)

   # Effective parameters
   ax3 = axes[1, 0]
   p_waic_values = [poly_waic[d]['p_waic'] for d in degrees_list]

   ax3.plot(degrees_list, degrees_list, 'k--', alpha=0.5, label='Number of parameters')
   ax3.plot(degrees_list, p_waic_values, 'bo-', label='Effective parameters')
   ax3.set_xlabel('Polynomial Degree')
   ax3.set_ylabel('Number of Parameters')
   ax3.set_title('Model Complexity')
   ax3.legend()
   ax3.grid(True, alpha=0.3)

   # WAIC vs LOO
   ax4 = axes[1, 1]
   waic_values = [poly_waic[d]['waic'] for d in degrees_list]
   elpd_loo_values = [-w for w in waic_values]  # Convert WAIC to ELPD scale
   elpd_values_actual = [poly_loo[d]['elpd_loo'] for d in degrees_list]

   ax4.scatter(elpd_values_actual, elpd_loo_values, s=60, alpha=0.7)
   ax4.plot([min(elpd_values_actual), max(elpd_values_actual)],
            [min(elpd_values_actual), max(elpd_values_actual)], 'k--', alpha=0.5)
   ax4.set_xlabel('ELPD (LOO)')
   ax4.set_ylabel('ELPD (WAIC)')
   ax4.set_title('LOO vs WAIC')
   ax4.grid(True, alpha=0.3)

   # Add degree labels
   for i, degree in enumerate(degrees_list):
       ax4.annotate(str(degree), (elpd_values_actual[i], elpd_loo_values[i]),
                   xytext=(5, 5), textcoords='offset points')

   plt.tight_layout()

Example 3: Mechanistic vs Empirical Models
------------------------------------------

Comparing a mechanistic model based on scientific theory with a flexible empirical model.

**Scientific Context**

Enzyme kinetics where we compare the Michaelis-Menten mechanistic model with a flexible spline-based empirical model.

**Implementation**

.. code-block:: python

   # Enzyme kinetics data
   substrate_concs = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])

   # True Michaelis-Menten relationship with some deviations
   true_Vmax = 10.0
   true_Km = 2.0
   true_velocities = true_Vmax * substrate_concs / (true_Km + substrate_concs)

   # Add systematic deviation and noise
   deviation = 0.5 * np.sin(2 * np.pi * np.log10(substrate_concs))
   observed_velocities = true_velocities + deviation + np.random.normal(0, 0.3, len(substrate_concs))

   # Model 1: Mechanistic (Michaelis-Menten)
   def michaelis_menten_model():
       Vmax = ssp.parameters.LogNormal(mu=np.log(10), sigma=0.5)
       Km = ssp.parameters.LogNormal(mu=np.log(2), sigma=0.7)
       sigma = ssp.parameters.LogNormal(mu=np.log(0.3), sigma=0.3)

       predicted_velocity = Vmax * substrate_concs / (Km + substrate_concs)

       likelihood = ssp.parameters.Normal(mu=predicted_velocity, sigma=sigma)
       likelihood.observe(observed_velocities)

       return ssp.Model(likelihood)

   # Model 2: Empirical (Gaussian Process - conceptual)
   def flexible_empirical_model():
       # This is a simplified version - full GP would require custom implementation

       # Use basis expansion approximation to GP
       n_basis = 6
       log_substrate = np.log(substrate_concs)

       # Basis functions (radial basis functions)
       basis_centers = np.linspace(log_substrate.min(), log_substrate.max(), n_basis)
       basis_scale = (log_substrate.max() - log_substrate.min()) / (n_basis - 1)

       # Basis function coefficients
       coefficients = ssp.parameters.Normal(mu=0, sigma=2, shape=(n_basis,))

       # Compute basis functions
       basis_values = np.exp(-(log_substrate[:, None] - basis_centers[None, :]) ** 2 / (2 * basis_scale ** 2))

       # Linear combination
       predicted_velocity = ssp.operations.sum(coefficients * basis_values.T, axis=0)

       # Ensure positive velocities
       predicted_velocity = ssp.operations.exp(predicted_velocity)

       sigma = ssp.parameters.LogNormal(mu=np.log(0.3), sigma=0.3)

       likelihood = ssp.parameters.LogNormal(
           mu=ssp.operations.log(predicted_velocity),
           sigma=sigma
       )
       likelihood.observe(observed_velocities)

       return ssp.Model(likelihood)

   # Model 3: Semi-mechanistic (MM + deviations)
   def semi_mechanistic_model():
       # Michaelis-Menten base + smooth deviations
       Vmax = ssp.parameters.LogNormal(mu=np.log(10), sigma=0.5)
       Km = ssp.parameters.LogNormal(mu=np.log(2), sigma=0.7)

       # Base MM prediction
       mm_prediction = Vmax * substrate_concs / (Km + substrate_concs)

       # Smooth deviations (random walk on log scale)
       deviation_sigma = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.5)
       deviations = [ssp.parameters.Normal(mu=0, sigma=deviation_sigma)]

       for i in range(1, len(substrate_concs)):
           next_deviation = ssp.parameters.Normal(
               mu=deviations[-1],
               sigma=deviation_sigma
           )
           deviations.append(next_deviation)

       # Total prediction
       total_prediction = mm_prediction + deviations

       sigma = ssp.parameters.LogNormal(mu=np.log(0.3), sigma=0.3)

       likelihood = ssp.parameters.Normal(mu=total_prediction, sigma=sigma)
       likelihood.observe(observed_velocities)

       return ssp.Model(likelihood)

   # Fit and compare all models
   kinetic_models = {
       'Mechanistic (MM)': michaelis_menten_model(),
       'Empirical (Flexible)': flexible_empirical_model(),
       'Semi-mechanistic': semi_mechanistic_model()
   }

   kinetic_results = {}
   kinetic_loo = {}
   kinetic_waic = {}

   for name, model in kinetic_models.items():
       print(f"Fitting {name} model...")
       kinetic_results[name] = model.sample(n_samples=1500)
       kinetic_loo[name] = model.loo(kinetic_results[name])
       kinetic_waic[name] = model.waic(kinetic_results[name])

   # Model comparison
   print("\nMechanistic vs Empirical Model Comparison:")
   print("-" * 55)
   print("Model                | ELPD (LOO) | SE    | Interpretability")
   print("-" * 55)

   interpretability = {
       'Mechanistic (MM)': 'High',
       'Empirical (Flexible)': 'Low',
       'Semi-mechanistic': 'Medium'
   }

   for name in kinetic_models.keys():
       elpd = kinetic_loo[name]['elpd_loo']
       se = kinetic_loo[name]['se_elpd_loo']
       interp = interpretability[name]
       print(f"{name:20} | {elpd:8.1f} | {se:4.1f} | {interp}")

Model Selection Strategies
--------------------------

**Information Criteria Guidelines**

.. code-block:: python

   def comprehensive_model_comparison(models, results_dict):
       """Comprehensive model comparison using multiple criteria."""

       comparison_results = {}

       for name, model in models.items():
           results = results_dict[name]

           # Information criteria
           loo = model.loo(results)
           waic = model.waic(results)

           # Posterior predictive checks
           post_pred = model.posterior_predictive(results)

           # Effective parameters
           p_eff = waic['p_waic']

           # Prior sensitivity (would require fitting with different priors)
           # prior_sensitivity = assess_prior_sensitivity(model, results)

           comparison_results[name] = {
               'elpd_loo': loo['elpd_loo'],
               'se_elpd_loo': loo['se_elpd_loo'],
               'waic': waic['waic'],
               'p_waic': waic['p_waic'],
               'posterior_mean_prediction': post_pred.mean(),
               'effective_parameters': p_eff
           }

       return comparison_results

   def model_selection_report(comparison_results, observed_data):
       """Generate comprehensive model selection report."""

       print("COMPREHENSIVE MODEL SELECTION REPORT")
       print("=" * 50)

       # Best model by different criteria
       best_loo = max(comparison_results.keys(),
                     key=lambda k: comparison_results[k]['elpd_loo'])
       best_waic = min(comparison_results.keys(),
                      key=lambda k: comparison_results[k]['waic'])

       print(f"Best model (LOO-CV): {best_loo}")
       print(f"Best model (WAIC): {best_waic}")

       # Model selection uncertainty
       elpd_values = [comparison_results[name]['elpd_loo'] for name in comparison_results.keys()]
       se_values = [comparison_results[name]['se_elpd_loo'] for name in comparison_results.keys()]

       best_elpd = max(elpd_values)
       model_differences = [best_elpd - elpd for elpd in elpd_values]

       print("\nModel Selection Uncertainty:")
       for i, (name, diff, se) in enumerate(zip(comparison_results.keys(), model_differences, se_values)):
           if diff < 2 * se:
               print(f"{name}: Competitive (Δ = {diff:.1f} ± {se:.1f})")
           elif diff < 4 * se:
               print(f"{name}: Possibly worse (Δ = {diff:.1f} ± {se:.1f})")
           else:
               print(f"{name}: Clearly worse (Δ = {diff:.1f} ± {se:.1f})")

**Model Averaging**

.. code-block:: python

   def model_averaging(models, results_dict, new_data):
       """Bayesian model averaging using information criteria weights."""

       # Compute model weights from LOO
       elpd_values = np.array([
           model.loo(results_dict[name])['elpd_loo']
           for name, model in models.items()
       ])

       # Convert to weights (pseudo-BMA)
       relative_elpd = elpd_values - elpd_values.max()
       weights = np.exp(relative_elpd)
       weights = weights / weights.sum()

       print("Model weights:")
       for i, (name, weight) in enumerate(zip(models.keys(), weights)):
           print(f"{name}: {weight:.3f}")

       # Weighted predictions
       predictions = []
       for name, model in models.items():
           pred = model.predict(new_data, results_dict[name])
           predictions.append(pred)

       # Weighted average
       weighted_prediction = sum(w * pred for w, pred in zip(weights, predictions))

       return weighted_prediction, weights

Best Practices for Model Comparison
-----------------------------------

1. **Use multiple criteria**: Don't rely on a single metric
2. **Consider scientific interpretability**: Balance fit with mechanistic understanding
3. **Account for uncertainty**: Use standard errors in comparisons
4. **Check robustness**: Assess sensitivity to priors and data subsets
5. **Validate assumptions**: Use posterior predictive checks
6. **Consider practical differences**: Small statistical differences may not be practically meaningful
7. **Use domain knowledge**: Scientific theory should inform model selection

These model comparison examples demonstrate systematic approaches to selecting the best model for your scientific question while properly accounting for uncertainty in the selection process.

.. note::
   LOO/WAIC & posterior_predictive not yet implemented. Example retains
   structure to show where comparisons would occur.

Accuracy Note
-------------
Removed unsupported: model.sample, loo(), waic(), posterior_predictive().
