Parameter Estimation Examples
=============================

This section provides real-world examples of parameter estimation across various scientific domains.

Example 1: Enzyme Kinetics
--------------------------

Estimate Michaelis-Menten parameters from enzyme assay data.

**Scientific Background**

Enzyme kinetics follow the Michaelis-Menten equation:

.. math::

   v = \frac{V_{max} \cdot [S]}{K_m + [S]}

Where:
- v: reaction velocity
- [S]: substrate concentration
- V_max: maximum velocity
- K_m: Michaelis constant (substrate concentration at half V_max)

**Implementation**

.. code-block:: python

   import scistanpy as ssp
   import numpy as np
   import matplotlib.pyplot as plt

   # Experimental data
   substrate_conc = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])  # mM
   velocity = np.array([0.8, 1.4, 2.8, 4.1, 5.9, 8.2, 9.1])        # μM/min
   velocity_error = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6])

   # Prior knowledge from literature
   V_max = ssp.parameters.LogNormal(mu=np.log(10), sigma=0.5)  # 3-30 μM/min
   K_m = ssp.parameters.LogNormal(mu=np.log(1), sigma=0.7)     # 0.1-10 mM
   extra_noise = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.3)

   # Michaelis-Menten model
   predicted_velocity = (V_max * substrate_conc) / (K_m + substrate_conc)

   # Combined uncertainty
   total_sigma = ssp.operations.sqrt(velocity_error**2 + extra_noise**2)

   # Likelihood
   likelihood = ssp.parameters.Normal(mu=predicted_velocity, sigma=total_sigma)
   likelihood.observe(velocity)

   # Inference
   model = ssp.Model(likelihood)
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=2000)

   # Results
   print(f"V_max: {results['V_max'].mean():.2f} ± {results['V_max'].std():.2f} μM/min")
   print(f"K_m: {results['K_m'].mean():.2f} ± {results['K_m'].std():.2f} mM")

Example 2: Radioactive Decay
----------------------------

Determine decay constants from counting data.

**Scientific Background**

Radioactive decay follows an exponential law:

.. math::

   N(t) = N_0 e^{-\lambda t}

Where λ is the decay constant and the half-life is t₁/₂ = ln(2)/λ.

**Implementation**

.. code-block:: python

   # Counting data (with Poisson uncertainty)
   time_points = np.array([0, 10, 20, 30, 40, 50, 60])  # minutes
   counts = np.array([1000, 819, 670, 549, 449, 368, 301])

   # Priors
   log_N0 = ssp.parameters.Normal(mu=np.log(1000), sigma=0.1)  # Initial counts
   lambda_decay = ssp.parameters.Exponential(rate=100)         # Decay rate (1/min)

   # Exponential decay model
   N0 = ssp.operations.exp(log_N0)
   expected_counts = N0 * ssp.operations.exp(-lambda_decay * time_points)

   # Poisson likelihood (counting statistics)
   likelihood = ssp.parameters.Poisson(rate=expected_counts)
   likelihood.observe(counts)

   # Inference
   model = ssp.Model(likelihood)
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)

   # Calculate half-life
   half_life = np.log(2) / results['lambda_decay']
   print(f"Half-life: {half_life.mean():.1f} ± {half_life.std():.1f} minutes")

Example 3: Spectral Line Fitting
--------------------------------

Fit Gaussian profiles to spectroscopic data.

**Scientific Background**

Spectral lines often have Gaussian profiles with parameters for amplitude, center wavelength, and width.

**Implementation**

.. code-block:: python

   # Spectroscopic data
   wavelength = np.linspace(500, 520, 200)  # nm
   intensity = np.array([...])  # Measured intensities
   noise_level = 0.05 * intensity.max()

   # Multiple Gaussian peaks
   n_peaks = 3

   # Priors for each peak
   amplitudes = ssp.parameters.LogNormal(mu=np.log(1), sigma=1, shape=(n_peaks,))
   centers = ssp.parameters.Normal(
       mu=np.array([505, 510, 515]),  # Expected positions
       sigma=2,
       shape=(n_peaks,)
   )
   widths = ssp.parameters.LogNormal(mu=np.log(1), sigma=0.5, shape=(n_peaks,))

   # Background
   background = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.5)

   # Gaussian peak model
   predicted_intensity = background
   for i in range(n_peaks):
       gaussian_peak = amplitudes[i] * ssp.operations.exp(
           -0.5 * ((wavelength - centers[i]) / widths[i])**2
       )
       predicted_intensity = predicted_intensity + gaussian_peak

   # Likelihood
   likelihood = ssp.parameters.Normal(
       mu=predicted_intensity,
       sigma=noise_level
   )
   likelihood.observe(intensity)

   # Inference
   model = ssp.Model(likelihood)
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)

   # Extract peak parameters
   for i in range(n_peaks):
       amp_mean = results[f'amplitudes'][i].mean()
       center_mean = results[f'centers'][i].mean()
       width_mean = results[f'widths'][i].mean()
       print(f"Peak {i+1}: λ={center_mean:.2f} nm, A={amp_mean:.3f}, σ={width_mean:.2f}")

Example 4: Thermal Analysis
--------------------------

Analyze temperature-dependent reaction rates using Arrhenius kinetics.

**Scientific Background**

The Arrhenius equation describes temperature dependence of reaction rates:

.. math::

   k = A e^{-E_a / RT}

Where:
- k: rate constant
- A: frequency factor
- E_a: activation energy
- R: gas constant
- T: temperature

**Implementation**

.. code-block:: python

   # Experimental data
   temperatures = np.array([298, 308, 318, 328, 338])  # K
   rate_constants = np.array([0.01, 0.025, 0.058, 0.125, 0.26])  # s⁻¹
   rate_errors = 0.1 * rate_constants  # 10% relative error

   R = 8.314  # J/(mol·K)

   # Priors (typical values for organic reactions)
   log_A = ssp.parameters.Normal(mu=np.log(1e8), sigma=2)  # Frequency factor
   Ea = ssp.parameters.Normal(mu=50000, sigma=10000)       # Activation energy (J/mol)
   extra_noise = ssp.parameters.LogNormal(mu=np.log(0.05), sigma=0.3)

   # Arrhenius model
   A = ssp.operations.exp(log_A)
   predicted_rates = A * ssp.operations.exp(-Ea / (R * temperatures))

   # Likelihood with measurement error
   total_error = ssp.operations.sqrt(rate_errors**2 + (extra_noise * predicted_rates)**2)
   likelihood = ssp.parameters.Normal(mu=predicted_rates, sigma=total_error)
   likelihood.observe(rate_constants)

   # Inference
   model = ssp.Model(likelihood)
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)

   # Results
   A_mean = np.exp(results['log_A'].mean())
   Ea_mean = results['Ea'].mean()
   print(f"Frequency factor: {A_mean:.2e} s⁻¹")
   print(f"Activation energy: {Ea_mean/1000:.1f} ± {results['Ea'].std()/1000:.1f} kJ/mol")

Example 5: Crystallography Unit Cell Refinement
-----------------------------------------------

Refine unit cell parameters from powder diffraction data.

**Scientific Background**

In powder diffraction, peak positions depend on unit cell parameters through Bragg's law and crystal geometry.

**Implementation**

.. code-block:: python

   # Observed peak positions (2θ in degrees)
   observed_peaks = np.array([12.5, 17.8, 22.3, 25.1, 29.6])
   peak_errors = np.array([0.02, 0.02, 0.03, 0.03, 0.04])

   # Miller indices for observed peaks
   hkl_indices = np.array([
       [1, 0, 0],
       [1, 1, 0],
       [2, 0, 0],
       [2, 1, 0],
       [2, 2, 0]
   ])

   # Wavelength (Cu Kα)
   wavelength = 1.5406  # Å

   # Priors for unit cell parameters (orthorhombic)
   a = ssp.parameters.Normal(mu=7.0, sigma=0.2)  # Å
   b = ssp.parameters.Normal(mu=8.5, sigma=0.3)  # Å
   c = ssp.parameters.Normal(mu=5.2, sigma=0.2)  # Å

   # Calculate d-spacings for orthorhombic system
   h, k, l = hkl_indices.T
   d_spacings = 1 / ssp.operations.sqrt(
       (h/a)**2 + (k/b)**2 + (l/c)**2
   )

   # Convert to 2θ using Bragg's law
   theta = ssp.operations.arcsin(wavelength / (2 * d_spacings))
   predicted_peaks = 2 * theta * 180 / np.pi  # Convert to degrees

   # Likelihood
   likelihood = ssp.parameters.Normal(
       mu=predicted_peaks,
       sigma=peak_errors
   )
   likelihood.observe(observed_peaks)

   # Inference
   model = ssp.Model(likelihood)
   results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)

   # Results
   print(f"Unit cell parameters:")
   print(f"a = {results['a'].mean():.3f} ± {results['a'].std():.3f} Å")
   print(f"b = {results['b'].mean():.3f} ± {results['b'].std():.3f} Å")
   print(f"c = {results['c'].mean():.3f} ± {results['c'].std():.3f} Å")

   # Volume and uncertainty
   volume = results['a'] * results['b'] * results['c']
   print(f"Volume = {volume.mean():.1f} ± {volume.std():.1f} Å³")

Model Validation and Diagnostics
-------------------------------

For all parameter estimation problems, validate your results:

**Convergence Diagnostics**

.. code-block:: python

   sample_failures, variable_failures = results.diagnose()
   print(sample_failures, variable_failures.keys())

Accuracy Note
-------------
Removed unsupported: model.sample, posterior_predictive, model.diagnose.
   plt.plot([min(observed_data), max(observed_data)],
            [min(observed_data), max(observed_data)], 'r--')
   plt.xlabel('Observed')
   plt.ylabel('Predicted')
   plt.title('Posterior Predictive Check')

**Convergence Diagnostics**

.. code-block:: python

   # Check R-hat and effective sample size
   diagnostics = model.diagnose(results)
   print(f"Max R-hat: {diagnostics['rhat'].max():.3f}")
   print(f"Min ESS: {diagnostics['ess_bulk'].min():.0f}")

**Parameter Correlations**

.. code-block:: python

   # Check for strong correlations
   import seaborn as sns
   param_samples = np.column_stack([results[param] for param in ['param1', 'param2']])
   correlation_matrix = np.corrcoef(param_samples.T)
   sns.heatmap(correlation_matrix, annot=True)

**Sensitivity Analysis**

.. code-block:: python

   # Try different priors
   model_alt = create_model_with_different_priors()
   results_alt = model_alt.sample()

   # Compare posterior means
   for param in ['param1', 'param2']:
       print(f"{param}: {results[param].mean():.3f} vs {results_alt[param].mean():.3f}")

These examples demonstrate best practices for parameter estimation across diverse scientific applications, emphasizing proper uncertainty quantification and model validation.
