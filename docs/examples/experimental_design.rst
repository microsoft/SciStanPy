.. _experimental_design:

==============================
Experimental Design Examples
==============================

Bayesian methods offer powerful tools for designing experiments that optimize information gain while respecting practical constraints. This section presents examples of experimental design scenarios, illustrating how to apply SciStanPy for planning and analysis.

Example 1: Basic A/B Test
--------------------------

**Scientific Context**

You want to test two formulations of a drug (A and B) to see which one leads to a higher response rate in patients.

**Implementation**

.. code-block:: python

   import scistanpy as ssp
   import numpy as np
   import matplotlib.pyplot as plt

   # Simulated patient response data
   np.random.seed(0)
   n_A = 50
   n_B = 50
   true_response_A = 0.65
   true_response_B = 0.80
   sd_response = 0.1

   # Generate synthetic data
   data_A = np.random.binomial(1, true_response_A, n_A)
   data_B = np.random.binomial(1, true_response_B, n_B)

   # Model for A/B testing
   class ABTestModel(ssp.Model):
       def __init__(self, data_A, data_B):
           super().__init__(default_data={"data_A": data_A, "data_B": data_B})

           # Priors for response rates
           self.response_A = ssp.parameters.Beta(alpha=1, beta=1)
           self.response_B = ssp.parameters.Beta(alpha=1, beta=1)

           # Likelihoods
           self.likelihood_A = ssp.parameters.Bernoulli(logit_p=self.response_A)
           self.likelihood_B = ssp.parameters.Bernoulli(logit_p=self.response_B)

           self.likelihood_A.observe(data_A)
           self.likelihood_B.observe(data_B)

       def contrast(self):
           """Contrast response rates between groups."""
           return self.response_B - self.response_A

   # Fit the model
   model = ABTestModel(data_A, data_B)
   results = model.mcmc(chains=4, iter_warmup=300, iter_sampling=700)

   # Print estimated response rates
   print(f"Group A: {results['response_A'].mean():.3f} (95% CI: [{np.percentile(results['response_A'], 2.5):.3f}, {np.percentile(results['response_A'], 97.5):.3f}])")
   print(f"Group B: {results['response_B'].mean():.3f} (95% CI: [{np.percentile(results['response_B'], 2.5):.3f}, {np.percentile(results['response_B'], 97.5):.3f}])")

   # Plot the posterior distributions
   import seaborn as sns
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   sns.histplot(results['response_A'], bins=30, ax=axes[0], color='skyblue', stat="density")
   sns.histplot(results['response_B'], bins=30, ax=axes[1], color='salmon', stat="density")
   axes[0].set_title("Posterior of Response Rate - Group A")
   axes[1].set_title("Posterior of Response Rate - Group B")
   for ax in axes:
       ax.set_xlabel("Response Rate")
       ax.set_ylabel("Density")
       ax.grid(True, alpha=0.3)

   plt.tight_layout()

Example 2: Factorial Experiment Design
---------------------------------------

**Scientific Context**

You are testing the effects of two factors (temperature and pressure) on the yield of a chemical reaction. Each factor has two levels (low and high), resulting in four experimental conditions.

**Implementation**

.. code-block:: python

   # Simulated yield data
   np.random.seed(1)
   n_per_condition = 30
   true_effects = {
       "low_temp_low_pres": 5,
       "low_temp_high_pres": 7,
       "high_temp_low_pres": 6,
       "high_temp_high_pres": 9
   }
   sd_yield = 1.5

   # Generate synthetic data
   data = {}
   for condition, effect in true_effects.items():
       data[condition] = np.random.normal(effect, sd_yield, n_per_condition)

   # Factorial model
   class FactorialModel(ssp.Model):
       def __init__(self, data):
           super().__init__(default_data=data)

           # Priors for group means
           self.means = {key: ssp.parameters.Normal(mu=0, sigma=10) for key in data.keys()}

           # Likelihoods
           self.likelihoods = {
               key: ssp.parameters.Normal(mu=self.means[key], sigma=sd_yield)
               for key in data.keys()
           }

           for key, value in data.items():
               self.likelihoods[key].observe(value)

       def contrasts(self):
           """Compute contrasts between conditions."""
           return {
               "high_temp_high_pres_vs_low": self.means["high_temp_high_pres"] - self.means["low_temp_low_pres"],
               "high_temp_low_pres_vs_low": self.means["high_temp_low_pres"] - self.means["low_temp_low_pres"],
               "high_temp_high_pres_vs_high": self.means["high_temp_high_pres"] - self.means["high_temp_low_pres"],
               "low_temp_high_pres_vs_low": self.means["low_temp_high_pres"] - self.means["low_temp_low_pres"]
           }

   # Fit the model
   factorial_model = FactorialModel(data)
   factorial_results = factorial_model.mcmc(chains=4, iter_warmup=300, iter_sampling=700)

   # Print estimated effects
   for condition, samples in factorial_results['means'].items():
       print(f"{condition}: {samples.mean():.3f} (95% CI: [{np.percentile(samples, 2.5):.3f}, {np.percentile(samples, 97.5):.3f}])")

   # Contrasts
   contrasts = factorial_results['contrasts']()
   for contrast, samples in contrasts.items():
       print(f"{contrast}: {samples.mean():.3f} (95% CI: [{np.percentile(samples, 2.5):.3f}, {np.percentile(samples, 97.5):.3f}])")

   # Visualization
   fig, ax = plt.subplots(figsize=(8, 6))
   sns.boxplot(data=[v for v in data.values()], ax=ax)
   ax.set_xticklabels(data.keys())
   ax.set_ylabel("Yield")
   ax.set_title("Yield by Experimental Condition")
   ax.grid(True, alpha=0.3)

Example 3: Sequential Experimental Design
-----------------------------------------

Adaptive design where experimental conditions are chosen based on accumulating data.

**Scientific Context**

Dose-finding study where we adaptively select the next dose to test based on previous results.

**Implementation**

.. code-block:: python

   # Sequential dose-finding design
   class AdaptiveDesign:
       """Adaptive experimental design for dose-finding."""

       def __init__(self, dose_range=(0.1, 100), target_response=50):
           self.dose_range = dose_range
           self.target_response = target_response
           self.doses_tested = []
           self.responses_observed = []
           self.models_fitted = []

       def dose_response_model(self, doses, responses):
           """Fit dose-response model to current data."""

           if len(doses) < 2:
               # Not enough data for full model, use simple prior
               log_EC50 = ssp.parameters.Normal(mu=np.log(10), sigma=1)
               hill = ssp.parameters.LogNormal(mu=np.log(1), sigma=0.5)
           else:
               # Informative priors based on current data
               empirical_EC50 = np.median(doses)
               log_EC50 = ssp.parameters.Normal(mu=np.log(empirical_EC50), sigma=0.5)
               hill = ssp.parameters.LogNormal(mu=np.log(1), sigma=0.3)

           sigma = ssp.parameters.LogNormal(mu=np.log(5), sigma=0.3)

           # Hill equation
           EC50 = 10 ** log_EC50
           predicted_responses = 100 / (1 + (EC50 / doses) ** hill)

           likelihood = ssp.parameters.Normal(mu=predicted_responses, sigma=sigma)
           likelihood.observe(responses)

           return ssp.Model(likelihood)

       def expected_utility(self, candidate_dose, n_simulations=100):
           """Calculate expected utility of testing a candidate dose."""

           if not self.responses_observed:
               # No data yet, use uniform utility
               return 1.0

           # Fit current model
           current_model = self.dose_response_model(
               np.array(self.doses_tested),
               np.array(self.responses_observed)
           )
           current_results = current_model.mcmc(chains=2, iter_warmup=200, iter_sampling=300)

           utilities = []

           for sim in range(n_simulations):
               # Sample parameters from current posterior
               idx = np.random.randint(len(current_results['log_EC50']))
               log_EC50_sim = current_results['log_EC50'][idx]
               hill_sim = current_results['hill'][idx]
               sigma_sim = current_results['sigma'][idx]

               # Simulate response at candidate dose
               EC50_sim = 10 ** log_EC50_sim
               expected_response = 100 / (1 + (EC50_sim / candidate_dose) ** hill_sim)
               simulated_response = np.random.normal(expected_response, sigma_sim)

               # Calculate utility (information gain about target dose)
               # Utility is higher when we're closer to target response
               utility = 1 / (1 + abs(simulated_response - self.target_response))

               # Bonus for exploring unexplored regions
               min_distance_to_tested = min([abs(np.log(candidate_dose) - np.log(d))
                                           for d in self.doses_tested], default=1)
               exploration_bonus = min_distance_to_tested / 2

               total_utility = utility + exploration_bonus
               utilities.append(total_utility)

           return np.mean(utilities)

       def select_next_dose(self, n_candidates=20):
           """Select the next dose to test."""

           log_dose_min, log_dose_max = np.log(self.dose_range)
           candidate_doses = np.logspace(log_dose_min, log_dose_max, n_candidates)

           utilities = []
           for dose in candidate_doses:
               utility = self.expected_utility(dose)
               utilities.append(utility)

           best_idx = np.argmax(utilities)
           return candidate_doses[best_idx], utilities

       def run_experiment(self, dose):
           """Simulate running an experiment at the specified dose."""

           # Simulate "true" dose-response relationship
           true_EC50 = 5.0
           true_hill = 1.5
           true_sigma = 8.0

           true_response = 100 / (1 + (true_EC50 / dose) ** true_hill)
           observed_response = np.random.normal(true_response, true_sigma)
           observed_response = np.clip(observed_response, 0, 100)

           return observed_response

       def adaptive_experiment(self, max_experiments=10):
           """Run adaptive experimental sequence."""

           print("Starting adaptive dose-finding experiment...")
           print(f"Target response: {self.target_response}%")
           print("-" * 50)

           for experiment in range(max_experiments):
               # Select next dose
               next_dose, utilities = self.select_next_dose()

               # Run experiment
               response = self.run_experiment(next_dose)

               # Update data
               self.doses_tested.append(next_dose)
               self.responses_observed.append(response)

               print(f"Experiment {experiment+1}: Dose = {next_dose:.2f}, Response = {response:.1f}%")

               # Check stopping criterion
               if len(self.responses_observed) >= 3:
                   # Fit model and check if we're close to target
                   model = self.dose_response_model(
                       np.array(self.doses_tested),
                       np.array(self.responses_observed)
                   )
                   results = model.mcmc(chains=4, iter_warmup=300, iter_sampling=700)

                   # Predict response at current best estimate of target dose
                   log_EC50_mean = results['log_EC50'].mean()
                   hill_mean = results['hill'].mean()

                   # Find dose giving target response
                   target_dose_estimate = (10 ** log_EC50_mean) * (
                       (100 / self.target_response - 1) ** (1 / hill_mean)
                   )

                   # Check if we have sufficient precision
                   log_EC50_sd = results['log_EC50'].std()
                   if log_EC50_sd < 0.2:  # Good precision
                       print(f"\nStopping: Sufficient precision achieved")
                       print(f"Estimated target dose: {target_dose_estimate:.2f}")
                       break

           return self.doses_tested, self.responses_observed

   # Run adaptive experiment
   np.random.seed(456)
   adaptive_design = AdaptiveDesign(target_response=50)
   final_doses, final_responses = adaptive_design.adaptive_experiment()

   # Compare with fixed design
   def fixed_design_experiment(n_doses=10):
       """Run experiment with fixed, evenly spaced doses."""

       doses = np.logspace(np.log10(0.1), np.log10(100), n_doses)
       responses = []

       for dose in doses:
           response = adaptive_design.run_experiment(dose)  # Use same simulation
           responses.append(response)

       return doses, responses

   fixed_doses, fixed_responses = fixed_design_experiment(len(final_doses))

   # Visualization of adaptive vs fixed design
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # Dose-response curves
   dose_fine = np.logspace(-1, 2, 100)
   true_response_fine = 100 / (1 + (5.0 / dose_fine) ** 1.5)

   ax1 = axes[0, 0]
   ax1.semilogx(dose_fine, true_response_fine, 'k-', linewidth=2, label='True curve')
   ax1.semilogx(final_doses, final_responses, 'ro', markersize=8, label='Adaptive design')
   ax1.axhline(50, color='red', linestyle='--', alpha=0.7, label='Target response')
   ax1.set_xlabel('Dose')
   ax1.set_ylabel('Response (%)')
   ax1.set_title('Adaptive Design Results')
   ax1.legend()
   ax1.grid(True, alpha=0.3)

   ax2 = axes[0, 1]
   ax2.semilogx(dose_fine, true_response_fine, 'k-', linewidth=2, label='True curve')
   ax2.semilogx(fixed_doses, fixed_responses, 'bo', markersize=8, label='Fixed design')
   ax2.axhline(50, color='red', linestyle='--', alpha=0.7, label='Target response')
   ax2.set_xlabel('Dose')
   ax2.set_ylabel('Response (%)')
   ax2.set_title('Fixed Design Results')
   ax2.legend()
   ax2.grid(True, alpha=0.3)

   # Dose selection sequence (adaptive only)
   ax3 = axes[1, 0]
   ax3.semilogx(range(1, len(final_doses)+1), final_doses, 'ro-')
   ax3.set_xlabel('Experiment number')
   ax3.set_ylabel('Dose selected')
   ax3.set_title('Adaptive Dose Selection Sequence')
   ax3.grid(True, alpha=0.3)

   # Precision comparison
   ax4 = axes[1, 1]

   # Fit models to both datasets
   adaptive_model = adaptive_design.dose_response_model(
       np.array(final_doses), np.array(final_responses)
   )
   adaptive_results = adaptive_model.sample(n_samples=1000)

   fixed_model = adaptive_design.dose_response_model(
       np.array(fixed_doses), np.array(fixed_responses)
   )
   fixed_results = fixed_model.sample(n_samples=1000)

   # Compare EC50 estimates
   ax4.hist(10 ** adaptive_results['log_EC50'], bins=30, alpha=0.7,
            density=True, label='Adaptive design')
   ax4.hist(10 ** fixed_results['log_EC50'], bins=30, alpha=0.7,
            density=True, label='Fixed design')
   ax4.axvline(5.0, color='black', linestyle='--', label='True EC50')
   ax4.set_xlabel('EC50 estimate')
   ax4.set_ylabel('Density')
   ax4.set_title('Parameter Estimation Precision')
   ax4.legend()
   ax4.grid(True, alpha=0.3)

   plt.tight_layout()

   # Summary statistics
   adaptive_ec50_est = (10 ** adaptive_results['log_EC50'])
   fixed_ec50_est = (10 ** fixed_results['log_EC50'])

   print(f"\nDesign Comparison:")
   print(f"True EC50: 5.0")
   print(f"Adaptive design - EC50: {adaptive_ec50_est.mean():.2f} ± {adaptive_ec50_est.std():.2f}")
   print(f"Fixed design - EC50: {fixed_ec50_est.mean():.2f} ± {fixed_ec50_est.std():.2f}")
   print(f"Adaptive precision gain: {fixed_ec50_est.std() / adaptive_ec50_est.std():.1f}x")

Best Practices for Experimental Design
--------------------------------------

**General Principles**

1. **Define clear objectives**: What parameters do you want to estimate?
2. **Quantify prior knowledge**: Use informative priors based on literature
3. **Consider constraints**: Budget, time, ethical considerations
4. **Plan for uncertainty**: Use robust designs that work across parameter ranges
5. **Validate designs**: Simulate experiments before implementation

**Design Optimization Strategies**

.. code-block:: python

   def design_optimization_checklist():
       """Checklist for experimental design optimization."""

       checklist = {
           'Objectives': [
               'Primary parameters identified',
               'Precision requirements specified',
               'Decision criteria defined'
           ],
           'Prior Knowledge': [
               'Literature review completed',
               'Expert opinions collected',
               'Prior distributions specified',
               'Sensitivity analysis performed'
           ],
           'Constraints': [
               'Budget limitations considered',
               'Time constraints identified',
               'Resource availability confirmed',
               'Ethical approval obtained'
           ],
           'Design Features': [
               'Dose/condition range optimized',
               'Sample sizes determined',
               'Randomization scheme planned',
               'Control conditions included'
           ],
           'Validation': [
               'Simulation studies completed',
               'Power analysis performed',
               'Robustness checked',
               'Alternative designs compared'
           ],
           'Implementation': [
               'Protocol finalized',
               'Data collection procedures defined',
               'Quality control measures established',
               'Analysis plan pre-specified'
           ]
       }

       return checklist

**Common Design Types**

- **D-optimal**: Minimize determinant of parameter covariance matrix
- **A-optimal**: Minimize trace of parameter covariance matrix
- **E-optimal**: Minimize maximum eigenvalue of covariance matrix
- **T-optimal**: Optimize for specific parameter or function of parameters
- **Robust**: Perform well across range of possible parameter values
- **Sequential/Adaptive**: Update design based on accumulating data

**Simulation-Based Design**

.. code-block:: python

   def simulate_design_performance(design_function, true_parameters, n_replications=100):
       """Evaluate design performance through simulation."""

       results = []

       for rep in range(n_replications):
           # Generate design
           experimental_design = design_function()

           # Simulate experiment
           simulated_data = simulate_experiment(experimental_design, true_parameters)

           # Analyze simulated data
           model = create_analysis_model(experimental_design, simulated_data)
           analysis_results = model.sample()

           # Extract performance metrics
           parameter_estimates = extract_parameters(analysis_results)
           results.append(parameter_estimates)

       # Summarize performance
       performance_summary = {
           'bias': np.mean(results, axis=0) - true_parameters,
           'variance': np.var(results, axis=0),
           'mse': np.mean((results - true_parameters)**2, axis=0),
           'coverage': calculate_coverage(results, true_parameters)
       }

       return performance_summary

These experimental design examples demonstrate how Bayesian methods can optimize experiments to maximize information gain while respecting practical constraints.

Accuracy Note
-------------
Updated model.sample -> model.mcmc; other advanced adaptive utilities left
conceptual. No posterior predictive helpers included.