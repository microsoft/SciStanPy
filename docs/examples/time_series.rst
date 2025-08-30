Time Series Analysis Examples
=============================

This section demonstrates time series modeling approaches for scientific data using SciStanPy.

Example 1: Random Walk with Measurement Error
---------------------------------------------

Modeling a physical process that evolves as a random walk with noisy observations.

**Scientific Context**

Tracking the position of a particle undergoing Brownian motion with measurement uncertainty.

**Implementation**

.. code-block:: python

   import scistanpy as ssp
   import numpy as np
   import matplotlib.pyplot as plt

   # Simulated particle position data
   np.random.seed(42)
   n_timepoints = 100
   time = np.arange(n_timepoints)

   # True underlying random walk
   true_innovation_std = 0.1
   true_positions = np.cumsum(np.random.normal(0, true_innovation_std, n_timepoints))

   # Noisy observations
   measurement_std = 0.05
   observed_positions = true_positions + np.random.normal(0, measurement_std, n_timepoints)

   # Random walk model
   # Initial position
   initial_position = ssp.parameters.Normal(mu=0, sigma=1)

   # Innovation noise
   innovation_sigma = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.5)

   # Measurement noise
   measurement_sigma = ssp.parameters.LogNormal(mu=np.log(0.05), sigma=0.3)

   # Build the time series
   positions = [initial_position]

   for t in range(1, n_timepoints):
       # Next position follows previous position + innovation
       next_position = ssp.parameters.Normal(
           mu=positions[-1],
           sigma=innovation_sigma
       )
       positions.append(next_position)

   # Observation model
   likelihoods = []
   for t, pos in enumerate(positions):
       likelihood = ssp.parameters.Normal(mu=pos, sigma=measurement_sigma)
       likelihood.observe(observed_positions[t])
       likelihoods.append(likelihood)

   # Combined model
   model = ssp.Model(likelihoods)
   results = model.mcmc(chains=2, iter_warmup=500, iter_sampling=1000)

   # Extract smoothed positions
   smoothed_positions = np.array([
       results[f'positions_{t}'].mean() for t in range(n_timepoints)
   ])
   position_uncertainty = np.array([
       results[f'positions_{t}'].std() for t in range(n_timepoints)
   ])

   # Plot results
   plt.figure(figsize=(12, 6))
   plt.plot(time, true_positions, 'g-', label='True positions', alpha=0.7)
   plt.scatter(time, observed_positions, c='red', s=20, label='Observations', alpha=0.6)
   plt.plot(time, smoothed_positions, 'b-', label='Smoothed positions')
   plt.fill_between(time,
                    smoothed_positions - 1.96*position_uncertainty,
                    smoothed_positions + 1.96*position_uncertainty,
                    alpha=0.3, label='95% CI')
   plt.xlabel('Time')
   plt.ylabel('Position')
   plt.title('Random Walk with Measurement Error')
   plt.legend()
   plt.grid(True, alpha=0.3)

Example 2: Autoregressive Process (AR)
--------------------------------------

Modeling temporal dependencies with autoregressive structure.

**Scientific Context**

Daily temperature measurements where today's temperature depends on recent temperatures.

**Implementation**

.. code-block:: python

   # Temperature time series data
   n_days = 365
   dates = np.arange(n_days)

   # Simulate AR(2) process for temperature anomalies
   np.random.seed(123)
   true_phi1, true_phi2 = 0.7, -0.2  # AR coefficients
   true_sigma = 2.0  # Innovation noise

   temp_anomalies = np.zeros(n_days)
   temp_anomalies[0] = np.random.normal(0, true_sigma)
   temp_anomalies[1] = true_phi1 * temp_anomalies[0] + np.random.normal(0, true_sigma)

   for t in range(2, n_days):
       temp_anomalies[t] = (true_phi1 * temp_anomalies[t-1] +
                           true_phi2 * temp_anomalies[t-2] +
                           np.random.normal(0, true_sigma))

   # Add seasonal component
   seasonal = 10 * np.sin(2 * np.pi * dates / 365)
   temperatures = 15 + seasonal + temp_anomalies  # Base temp 15°C

   # Add measurement noise
   observed_temps = temperatures + np.random.normal(0, 0.5, n_days)

   # AR(2) model
   # Priors for AR coefficients (stationary constraint)
   phi1 = ssp.parameters.Normal(mu=0.5, sigma=0.3)
   phi2 = ssp.parameters.Normal(mu=0, sigma=0.3)

   # Seasonal component
   seasonal_amplitude = ssp.parameters.Normal(mu=10, sigma=2)
   base_temp = ssp.parameters.Normal(mu=15, sigma=3)

   # Innovation and observation noise
   innovation_sigma = ssp.parameters.LogNormal(mu=np.log(2), sigma=0.5)
   obs_sigma = ssp.parameters.LogNormal(mu=np.log(0.5), sigma=0.3)

   # Model structure
   seasonal_component = seasonal_amplitude * ssp.operations.sin(2 * np.pi * dates / 365)

   # Initial conditions
   temp_anom = [
       ssp.parameters.Normal(mu=0, sigma=innovation_sigma),  # t=0
       ssp.parameters.Normal(mu=0, sigma=innovation_sigma)   # t=1
   ]

   # AR process for t >= 2
   for t in range(2, n_days):
       ar_mean = phi1 * temp_anom[t-1] + phi2 * temp_anom[t-2]
       next_anom = ssp.parameters.Normal(mu=ar_mean, sigma=innovation_sigma)
       temp_anom.append(next_anom)

   # Predicted temperatures
   predicted_temps = [
       base_temp + seasonal_component[t] + temp_anom[t]
       for t in range(n_days)
   ]

   # Likelihood
   likelihoods = []
   for t in range(n_days):
       likelihood = ssp.parameters.Normal(mu=predicted_temps[t], sigma=obs_sigma)
       likelihood.observe(observed_temps[t])
       likelihoods.append(likelihood)

   # Fit model
   model = ssp.Model(likelihoods)
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1500)

   # Results
   print("AR coefficients:")
   print(f"φ₁: {results['phi1'].mean():.3f} ± {results['phi1'].std():.3f} (true: {true_phi1})")
   print(f"φ₂: {results['phi2'].mean():.3f} ± {results['phi2'].std():.3f} (true: {true_phi2})")
   print(f"Innovation σ: {results['innovation_sigma'].mean():.3f} ± {results['innovation_sigma'].std():.3f} (true: {true_sigma})")

Example 3: State-Space Model with Missing Data
----------------------------------------------

Handling missing observations in a dynamic system.

**Scientific Context**

Monitoring bacterial growth where some measurements are missing due to equipment failures.

**Implementation**

.. code-block:: python

   # Bacterial growth data with missing observations
   n_hours = 48
   time_hours = np.arange(n_hours)

   # True exponential growth (log scale)
   true_growth_rate = 0.1  # per hour
   true_initial_log_pop = np.log(1000)  # log(cells/mL)
   true_noise = 0.05

   # Simulate true population
   true_log_pop = true_initial_log_pop + true_growth_rate * time_hours
   true_log_pop += np.random.normal(0, true_noise, n_hours)

   # Observed data with missing values
   np.random.seed(456)
   missing_prob = 0.3  # 30% missing
   observed_mask = np.random.random(n_hours) > missing_prob
   observed_indices = np.where(observed_mask)[0]
   observed_log_pop = true_log_pop[observed_mask]

   print(f"Observed {len(observed_indices)} out of {n_hours} time points")

   # State-space model
   # Initial state
   log_pop_0 = ssp.parameters.Normal(mu=np.log(1000), sigma=0.2)

   # Growth parameters
   growth_rate = ssp.parameters.Normal(mu=0.1, sigma=0.05)
   process_noise = ssp.parameters.LogNormal(mu=np.log(0.05), sigma=0.3)
   obs_noise = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.3)

   # State evolution (log population)
   log_populations = [log_pop_0]

   for t in range(1, n_hours):
       # Population grows exponentially in linear space
       # In log space: log(P(t)) = log(P(t-1)) + growth_rate * dt + noise
       expected_log_pop = log_populations[-1] + growth_rate * 1.0  # dt = 1 hour
       next_log_pop = ssp.parameters.Normal(mu=expected_log_pop, sigma=process_noise)
       log_populations.append(next_log_pop)

   # Observations (only for observed time points)
   likelihoods = []
   for i, obs_idx in enumerate(observed_indices):
       likelihood = ssp.parameters.Normal(
           mu=log_populations[obs_idx],
           sigma=obs_noise
       )
       likelihood.observe(observed_log_pop[i])
       likelihoods.append(likelihood)

   # Model
   model = ssp.Model(likelihoods)
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=2000)

   # Extract state estimates
   estimated_log_pop = np.array([
       results[f'log_populations_{t}'].mean() for t in range(n_hours)
   ])
   log_pop_ci_lower = np.array([
       np.percentile(results[f'log_populations_{t}'], 2.5) for t in range(n_hours)
   ])
   log_pop_ci_upper = np.array([
       np.percentile(results[f'log_populations_{t}'], 97.5) for t in range(n_hours)
   ])

   # Plot results
   plt.figure(figsize=(12, 6))
   plt.plot(time_hours, true_log_pop, 'g-', label='True log population', alpha=0.7)
   plt.scatter(time_hours[observed_mask], observed_log_pop,
               c='red', s=30, label='Observations', zorder=5)
   plt.plot(time_hours, estimated_log_pop, 'b-', label='Estimated log population')
   plt.fill_between(time_hours, log_pop_ci_lower, log_pop_ci_upper,
                    alpha=0.3, label='95% CI')
   plt.xlabel('Time (hours)')
   plt.ylabel('Log Population')
   plt.title('Bacterial Growth with Missing Data')
   plt.legend()
   plt.grid(True, alpha=0.3)

   # Parameter estimates
   print(f"\nParameter estimates:")
   print(f"Growth rate: {results['growth_rate'].mean():.4f} ± {results['growth_rate'].std():.4f} /hr")
   print(f"True growth rate: {true_growth_rate:.4f} /hr")

Example 4: Periodic Time Series with Trend
------------------------------------------

Decomposing a time series into trend, seasonal, and noise components.

**Scientific Context**

Analyzing atmospheric CO₂ concentrations with long-term trend and seasonal cycle.

**Implementation**

.. code-block:: python

   # Simulate CO2-like data
   n_months = 120  # 10 years of monthly data
   time_months = np.arange(n_months)
   time_years = time_months / 12

   # Components
   baseline_co2 = 350  # ppm
   trend_rate = 2.0    # ppm/year
   seasonal_amplitude = 6  # ppm

   # True time series
   trend = baseline_co2 + trend_rate * time_years
   seasonal = seasonal_amplitude * np.sin(2 * np.pi * time_months / 12 - np.pi/2)
   noise = np.random.normal(0, 1, n_months)
   true_co2 = trend + seasonal + noise

   # Add measurement error
   observed_co2 = true_co2 + np.random.normal(0, 0.5, n_months)

   # Trend + seasonal model
   # Trend parameters
   baseline = ssp.parameters.Normal(mu=350, sigma=10)
   trend_slope = ssp.parameters.Normal(mu=2, sigma=1)

   # Seasonal parameters
   seasonal_amp = ssp.parameters.Normal(mu=6, sigma=2)
   seasonal_phase = ssp.parameters.Normal(mu=-np.pi/2, sigma=0.5)

   # Noise parameters
   process_noise = ssp.parameters.LogNormal(mu=np.log(1), sigma=0.3)
   obs_noise = ssp.parameters.LogNormal(mu=np.log(0.5), sigma=0.3)

   # Model components
   trend_component = baseline + trend_slope * time_years
   seasonal_component = seasonal_amp * ssp.operations.sin(
       2 * np.pi * time_months / 12 + seasonal_phase
   )

   # For more realistic modeling, add AR(1) structure to residuals
   ar_coeff = ssp.parameters.Normal(mu=0.3, sigma=0.2)

   # First residual
   residuals = [ssp.parameters.Normal(mu=0, sigma=process_noise)]

   # Subsequent residuals with AR(1) structure
   for t in range(1, n_months):
       ar_mean = ar_coeff * residuals[-1]
       next_residual = ssp.parameters.Normal(mu=ar_mean, sigma=process_noise)
       residuals.append(next_residual)

   # Total predicted CO2
   predicted_co2 = [
       trend_component[t] + seasonal_component[t] + residuals[t]
       for t in range(n_months)
   ]

   # Likelihood
   likelihoods = []
   for t in range(n_months):
       likelihood = ssp.parameters.Normal(mu=predicted_co2[t], sigma=obs_noise)
       likelihood.observe(observed_co2[t])
       likelihoods.append(likelihood)

   # Fit model
   model = ssp.Model(likelihoods)
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1500)

   # Decompose the time series
   estimated_trend = results['baseline'].mean() + results['trend_slope'].mean() * time_years
   estimated_seasonal = results['seasonal_amp'].mean() * np.sin(
       2 * np.pi * time_months / 12 + results['seasonal_phase'].mean()
   )
   estimated_total = np.array([
       results[f'predicted_co2_{t}'].mean() for t in range(n_months)
   ])

   # Plot decomposition
   fig, axes = plt.subplots(4, 1, figsize=(12, 10))

   # Original data
   axes[0].plot(time_years, observed_co2, 'k-', alpha=0.7, label='Observed')
   axes[0].plot(time_years, estimated_total, 'r-', label='Fitted')
   axes[0].set_ylabel('CO₂ (ppm)')
   axes[0].set_title('Time Series Decomposition')
   axes[0].legend()
   axes[0].grid(True, alpha=0.3)

   # Trend
   axes[1].plot(time_years, true_co2 - seasonal - noise, 'g-', alpha=0.7, label='True trend')
   axes[1].plot(time_years, estimated_trend, 'b-', label='Estimated trend')
   axes[1].set_ylabel('Trend (ppm)')
   axes[1].legend()
   axes[1].grid(True, alpha=0.3)

   # Seasonal
   axes[2].plot(time_years, seasonal, 'g-', alpha=0.7, label='True seasonal')
   axes[2].plot(time_years, estimated_seasonal, 'b-', label='Estimated seasonal')
   axes[2].set_ylabel('Seasonal (ppm)')
   axes[2].legend()
   axes[2].grid(True, alpha=0.3)

   # Residuals
   residual_estimates = estimated_total - estimated_trend - estimated_seasonal
   axes[3].plot(time_years, noise, 'g-', alpha=0.7, label='True residuals')
   axes[3].plot(time_years, residual_estimates, 'b-', label='Estimated residuals')
   axes[3].set_ylabel('Residuals (ppm)')
   axes[3].set_xlabel('Time (years)')
   axes[3].legend()
   axes[3].grid(True, alpha=0.3)

   plt.tight_layout()

   # Parameter estimates
   print("Parameter estimates:")
   print(f"Trend rate: {results['trend_slope'].mean():.2f} ± {results['trend_slope'].std():.2f} ppm/year")
   print(f"Seasonal amplitude: {results['seasonal_amp'].mean():.2f} ± {results['seasonal_amp'].std():.2f} ppm")
   print(f"AR coefficient: {results['ar_coeff'].mean():.3f} ± {results['ar_coeff'].std():.3f}")

Example 5: Nonlinear Growth Model
---------------------------------

Modeling population growth with time-varying carrying capacity.

**Scientific Context**

Bacterial population growth in a bioreactor with changing environmental conditions.

**Implementation**

.. code-block:: python

   # Time-varying logistic growth
   n_days = 30
   time_days = np.arange(n_days)

   # Simulate logistic growth with time-varying carrying capacity
   true_r = 0.3  # growth rate
   K_base = 1000  # base carrying capacity
   K_variation = 200  # carrying capacity variation

   # Time-varying carrying capacity
   K_t = K_base + K_variation * np.sin(2 * np.pi * time_days / 10)

   # Simulate logistic growth
   true_pop = np.zeros(n_days)
   true_pop[0] = 50  # initial population

   for t in range(1, n_days):
       dt = 0.1  # integration step
       for _ in range(10):  # sub-steps for numerical integration
           growth = true_r * true_pop[t-1] * (1 - true_pop[t-1] / K_t[t])
           true_pop[t-1] += growth * dt
       true_pop[t] = true_pop[t-1]

   # Add observation noise
   observed_pop = true_pop * np.random.lognormal(0, 0.1, n_days)

   # Time-varying logistic growth model
   # Parameters
   initial_pop = ssp.parameters.LogNormal(mu=np.log(50), sigma=0.3)
   growth_rate = ssp.parameters.LogNormal(mu=np.log(0.3), sigma=0.3)

   # Carrying capacity parameters
   K_baseline = ssp.parameters.LogNormal(mu=np.log(1000), sigma=0.2)
   K_amplitude = ssp.parameters.LogNormal(mu=np.log(200), sigma=0.5)
   K_period = ssp.parameters.LogNormal(mu=np.log(10), sigma=0.3)
   K_phase = ssp.parameters.Normal(mu=0, sigma=1)

   # Observation noise
   obs_cv = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.3)

   # Time-varying carrying capacity
   carrying_capacity = K_baseline + K_amplitude * ssp.operations.sin(
       2 * np.pi * time_days / K_period + K_phase
   )

   # Discrete-time logistic growth
   populations = [initial_pop]

   for t in range(1, n_days):
       # Logistic growth equation
       prev_pop = populations[-1]
       growth_factor = 1 + growth_rate * (1 - prev_pop / carrying_capacity[t])
       next_pop = prev_pop * growth_factor
       populations.append(next_pop)

   # Likelihood with log-normal observation model
   likelihoods = []
   for t in range(n_days):
       # Log-normal observation model (multiplicative noise)
       likelihood = ssp.parameters.LogNormal(
           mu=ssp.operations.log(populations[t]),
           sigma=obs_cv
       )
       likelihood.observe(observed_pop[t])
       likelihoods.append(likelihood)

   # Fit model
   model = ssp.Model(likelihoods)
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=2000)

   # Extract estimates
   estimated_pop = np.array([
       results[f'populations_{t}'].mean() for t in range(n_days)
   ])
   pop_ci_lower = np.array([
       np.percentile(results[f'populations_{t}'], 2.5) for t in range(n_days)
   ])
   pop_ci_upper = np.array([
       np.percentile(results[f'populations_{t}'], 97.5) for t in range(n_days)
   ])

   estimated_K = (results['K_baseline'].mean() +
                  results['K_amplitude'].mean() *
                  np.sin(2 * np.pi * time_days / results['K_period'].mean() +
                         results['K_phase'].mean()))

   # Plot results
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

   # Population trajectory
   ax1.plot(time_days, true_pop, 'g-', label='True population', alpha=0.7)
   ax1.scatter(time_days, observed_pop, c='red', s=30, label='Observations', alpha=0.7)
   ax1.plot(time_days, estimated_pop, 'b-', label='Estimated population')
   ax1.fill_between(time_days, pop_ci_lower, pop_ci_upper, alpha=0.3, label='95% CI')
   ax1.set_ylabel('Population')
   ax1.set_title('Nonlinear Growth Model')
   ax1.legend()
   ax1.grid(True, alpha=0.3)

   # Carrying capacity
   ax2.plot(time_days, K_t, 'g-', label='True carrying capacity', alpha=0.7)
   ax2.plot(time_days, estimated_K, 'b-', label='Estimated carrying capacity')
   ax2.set_ylabel('Carrying capacity')
   ax2.set_xlabel('Time (days)')
   ax2.legend()
   ax2.grid(True, alpha=0.3)

   plt.tight_layout()

   # Parameter estimates
   print("Parameter estimates:")
   print(f"Growth rate: {results['growth_rate'].mean():.3f} ± {results['growth_rate'].std():.3f}")
   print(f"K baseline: {results['K_baseline'].mean():.1f} ± {results['K_baseline'].std():.1f}")
   print(f"K amplitude: {results['K_amplitude'].mean():.1f} ± {results['K_amplitude'].std():.1f}")

Time Series Forecasting
-----------------------

.. note::
   Forecasting & posterior predictive helper methods are not yet exposed.
   For now, export posterior draws of latent states/parameters from
   Model.mcmc() and implement custom simulation manually.

**Best Practices for Time Series**

1. **Check for stationarity** and transform if necessary
2. **Handle missing data** appropriately in state-space framework
3. **Validate forecasts** using proper cross-validation methods
4. **Model residual autocorrelation** to capture remaining temporal structure
5. **Use informative priors** based on domain knowledge
6. **Consider seasonal patterns** in your data
7. **Account for measurement error** when observations are noisy
8. **Check model assumptions** with residual analysis

Accuracy Note
-------------
Removed unsupported calls: model.sample, posterior_predictive, loo. Use
model.mcmc(...) and custom simulation instead.

These time series examples demonstrate the flexibility of SciStanPy for modeling complex temporal dynamics in scientific data.
   # forecast_mean = forecasts.mean(axis=0)
   # forecast_ci = np.percentile(forecasts, [2.5, 97.5], axis=0)

**Model Comparison for Time Series**

.. code-block:: python

   def compare_time_series_models(models, results_list, data):
       """Compare time series models using cross-validation."""

       # Time series cross-validation (expanding window)
       scores = {}

       for name, (model, results) in zip(models.keys(), zip(models.values(), results_list)):
           # Compute LOO or time series specific metrics
           loo_score = model.loo(results)
           scores[name] = loo_score['elpd_loo']

       return scores

**Best Practices for Time Series**

1. **Check for stationarity** and transform if necessary
2. **Handle missing data** appropriately in state-space framework
3. **Validate forecasts** using proper cross-validation methods
4. **Model residual autocorrelation** to capture remaining temporal structure
5. **Use informative priors** based on domain knowledge
6. **Consider seasonal patterns** in your data
7. **Account for measurement error** when observations are noisy
8. **Check model assumptions** with residual analysis

These time series examples demonstrate the flexibility of SciStanPy for modeling complex temporal dynamics in scientific data.
