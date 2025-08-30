Extending SciStanPy
===================

This guide covers how to extend SciStanPy's capabilities by adding new components, integrating with external libraries, and customizing the framework for specialized scientific applications.

Extension Architecture
---------------------

SciStanPy's modular design enables extensions at multiple levels:

**Component Level Extensions:**
- Custom probability distributions
- New transformation operations
- Specialized inference methods
- Domain-specific diagnostics

**Backend Extensions:**
- Integration with new computational backends
- Custom sampling algorithms
- External solver integration
- Hardware acceleration

**Framework Extensions:**
- Plugin systems for domain-specific functionality
- Custom model builders
- Automated workflow systems
- Integration with external data sources

Extension Points
---------------

Core Extension Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~

**AbstractModelComponent:**
Base class for all model components provides standard extension points:

.. code-block:: python

   from scistanpy.model.components.abstract_model_component import AbstractModelComponent

   class CustomComponent(AbstractModelComponent):
       """Custom component following SciStanPy patterns."""

       def __init__(self, **kwargs):
           """Initialize with standard SciStanPy interface."""
           super().__init__(**kwargs)
           # Custom initialization logic

       def get_stan_declaration(self) -> str:
           """Generate Stan variable declaration."""
           return f"real {self.name};"

       def get_stan_code(self) -> str:
           """Generate Stan code for this component."""
           return f"{self.name} ~ custom_distribution({self.params});"

**Parameter Extensions:**
Extend the parameter system with new distributions:

.. code-block:: python

   from scistanpy.model.components.parameters import ContinuousDistribution

   class BetaPrimeDistribution(ContinuousDistribution):
       """Beta prime distribution for ratio modeling."""

       SCIPY_DIST = stats.betaprime
       TORCH_DIST = None  # Implement if needed
       STAN_DIST = "beta_prime"

       def __init__(self, alpha, beta, **kwargs):
           super().__init__(alpha=alpha, beta=beta, **kwargs)

**Transformation Extensions:**
Add new mathematical operations:

.. code-block:: python

   from scistanpy.model.components.transformations.transformed_parameters import UnaryTransformedParameter

   class BoxCoxTransform(UnaryTransformedParameter):
       """Box-Cox transformation for normalization."""

       def __init__(self, dist1, lambda_param):
           self.lambda_param = lambda_param
           super().__init__(dist1=dist1)

       def run_np_torch_op(self, dist1):
           if self.lambda_param == 0:
               return torch.log(dist1)
           else:
               return (torch.pow(dist1, self.lambda_param) - 1) / self.lambda_param

       def write_stan_operation(self, dist1: str) -> str:
           if self.lambda_param == 0:
               return f"log({dist1})"
           else:
               return f"(pow({dist1}, {self.lambda_param}) - 1) / {self.lambda_param}"

Domain-Specific Extensions
-------------------------

Astronomy Extensions
~~~~~~~~~~~~~~~~~~~

**Celestial Coordinate Transformations:**

.. code-block:: python

   import numpy as np
   from astropy.coordinates import SkyCoord
   from scistanpy.model.components.transformations.transformed_parameters import BinaryTransformedParameter

   class EquatorialToGalactic(BinaryTransformedParameter):
       """Transform equatorial to galactic coordinates."""

       def __init__(self, ra, dec):
           """Transform right ascension and declination to galactic coordinates."""
           super().__init__(dist1=ra, dist2=dec)

       def run_np_torch_op(self, ra, dec):
           # Convert to astropy SkyCoord and transform
           equatorial = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
           galactic = equatorial.galactic
           return galactic.l.degree, galactic.b.degree

       def write_stan_operation(self, ra: str, dec: str) -> str:
           # Implement galactic transformation in Stan
           return f"equatorial_to_galactic({ra}, {dec})"

**Luminosity Distance Calculation:**

.. code-block:: python

   from astropy.cosmology import Planck18

   class LuminosityDistance(UnaryTransformedParameter):
       """Calculate luminosity distance from redshift."""

       def __init__(self, redshift, cosmology=Planck18):
           self.cosmology = cosmology
           super().__init__(dist1=redshift)

       def run_np_torch_op(self, redshift):
           # Calculate luminosity distance using astropy
           return torch.tensor([
               self.cosmology.luminosity_distance(z).value
               for z in redshift.numpy()
           ])

Chemistry Extensions
~~~~~~~~~~~~~~~~~~~

**Molecular Property Calculations:**

.. code-block:: python

   from rdkit import Chem
   from rdkit.Chem import Descriptors

   class MolecularWeight(AbstractModelComponent):
       """Calculate molecular weight from SMILES strings."""

       def __init__(self, smiles_strings, **kwargs):
           self.smiles = smiles_strings
           super().__init__(**kwargs)

           # Pre-calculate molecular weights
           self.mol_weights = self._calculate_weights()

       def _calculate_weights(self):
           weights = []
           for smiles in self.smiles:
               mol = Chem.MolFromSmiles(smiles)
               if mol is not None:
                   weights.append(Descriptors.MolWt(mol))
               else:
                   weights.append(np.nan)
           return np.array(weights)

       def get_stan_declaration(self) -> str:
           return f"vector[{len(self.mol_weights)}] {self.name};"

**Reaction Rate Models:**

.. code-block:: python

   class ArrheniusRate(BinaryTransformedParameter):
       """Arrhenius equation for temperature-dependent rates."""

       def __init__(self, temperature, activation_energy, pre_exponential_factor):
           self.R = 8.314  # Gas constant
           self.A = pre_exponential_factor
           super().__init__(dist1=temperature, dist2=activation_energy)

       def run_np_torch_op(self, temperature, activation_energy):
           return self.A * torch.exp(-activation_energy / (self.R * temperature))

       def write_stan_operation(self, temp: str, ea: str) -> str:
           return f"{self.A} * exp(-{ea} / ({self.R} * {temp}))"

Biology Extensions
~~~~~~~~~~~~~~~~~

**Population Dynamics Models:**

.. code-block:: python

   class LogisticGrowth(AbstractModelComponent):
       """Logistic population growth model."""

       def __init__(self, time_points, carrying_capacity, growth_rate, initial_population, **kwargs):
           self.time_points = time_points
           super().__init__(
               K=carrying_capacity,
               r=growth_rate,
               N0=initial_population,
               **kwargs
           )

       def population_at_time(self):
           """Calculate population at each time point."""
           K = self.K
           r = self.r
           N0 = self.N0
           t = self.time_points

           return K / (1 + ((K - N0) / N0) * torch.exp(-r * t))

**Phylogenetic Models:**

.. code-block:: python

   import dendropy

   class PhylogeneticDistance(AbstractModelComponent):
       """Calculate phylogenetic distances from tree."""

       def __init__(self, newick_tree, species_names, **kwargs):
           self.tree = dendropy.Tree.get(data=newick_tree, schema="newick")
           self.species = species_names
           super().__init__(**kwargs)

           self.distance_matrix = self._calculate_distances()

       def _calculate_distances(self):
           """Calculate pairwise phylogenetic distances."""
           pdm = self.tree.phylogenetic_distance_matrix()
           n_species = len(self.species)
           distances = np.zeros((n_species, n_species))

           for i, sp1 in enumerate(self.species):
               for j, sp2 in enumerate(self.species):
                   distances[i, j] = pdm.distance(
                       self.tree.find_node_with_taxon_label(sp1),
                       self.tree.find_node_with_taxon_label(sp2)
                   )

           return distances

Physics Extensions
~~~~~~~~~~~~~~~~~

**Quantum Mechanics Models:**

.. code-block:: python

   import qutip

   class QuantumEvolution(UnaryTransformedParameter):
       """Quantum state evolution under Hamiltonian."""

       def __init__(self, initial_state, hamiltonian, time):
           self.H = hamiltonian
           self.psi0 = initial_state
           super().__init__(dist1=time)

       def run_np_torch_op(self, time):
           """Calculate time evolution of quantum state."""
           # This is conceptual - actual implementation would need careful handling
           evolved_states = []
           for t in time:
               U = (-1j * self.H * t).expm()  # Time evolution operator
               psi_t = U * self.psi0
               evolved_states.append(psi_t.norm())
           return torch.tensor(evolved_states)

**Statistical Mechanics:**

.. code-block:: python

   class BoltzmannDistribution(ContinuousDistribution):
       """Boltzmann distribution for energy states."""

       def __init__(self, energy_levels, temperature, **kwargs):
           self.energy_levels = energy_levels
           self.kB = 1.380649e-23  # Boltzmann constant
           super().__init__(temperature=temperature, **kwargs)

       def partition_function(self):
           """Calculate partition function."""
           beta = 1 / (self.kB * self.temperature)
           return torch.sum(torch.exp(-beta * self.energy_levels))

       def log_prob(self, energy):
           """Log probability of energy state."""
           beta = 1 / (self.kB * self.temperature)
           Z = self.partition_function()
           return -beta * energy - torch.log(Z)

Backend Extensions
-----------------

Custom Inference Engines
~~~~~~~~~~~~~~~~~~~~~~~

**Custom MCMC Sampler:**

.. code-block:: python

   from scistanpy.inference.base import InferenceEngine

   class CustomMCMC(InferenceEngine):
       """Custom MCMC implementation."""

       def __init__(self, model, **kwargs):
           super().__init__(model)
           self.step_size = kwargs.get('step_size', 0.01)
           self.n_leapfrog = kwargs.get('n_leapfrog', 10)

       def sample(self, n_samples=1000, **kwargs):
           """Run custom MCMC sampling."""
           # Implement your sampling algorithm
           samples = self._run_custom_mcmc(n_samples)
           return self._format_results(samples)

       def _run_custom_mcmc(self, n_samples):
           """Core MCMC algorithm implementation."""
           # Your custom sampling logic here
           pass

**GPU-Accelerated Sampling:**

.. code-block:: python

   import cupy as cp  # GPU array library

   class GPUSampler(InferenceEngine):
       """GPU-accelerated inference engine."""

       def __init__(self, model, device='cuda'):
           super().__init__(model)
           self.device = device

       def sample(self, n_samples=1000, **kwargs):
           """GPU-accelerated sampling."""
           # Move computations to GPU
           with cp.cuda.Device(0):
               samples = self._gpu_sampling_loop(n_samples)

           # Move results back to CPU
           return {k: cp.asnumpy(v) for k, v in samples.items()}

External Library Integration
---------------------------

TensorFlow Probability Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import tensorflow_probability as tfp

   class TFPDistribution(ContinuousDistribution):
       """Wrapper for TensorFlow Probability distributions."""

       def __init__(self, tfp_dist, **kwargs):
           self.tfp_dist = tfp_dist
           super().__init__(**kwargs)

       def log_prob(self, value):
           """Use TFP log probability calculation."""
           return self.tfp_dist.log_prob(value)

       def sample(self, sample_shape=()):
           """Use TFP sampling."""
           return self.tfp_dist.sample(sample_shape)

JAX Integration
~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jax import jit, grad

   class JAXOptimizedParameter(ContinuousDistribution):
       """JAX-optimized parameter with JIT compilation."""

       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # JIT compile key functions
           self._jit_log_prob = jit(self._log_prob_impl)
           self._jit_sample = jit(self._sample_impl)

       @staticmethod
       def _log_prob_impl(value, params):
           """JIT-compiled log probability calculation."""
           # JAX implementation here
           pass

       def log_prob(self, value):
           """JIT-compiled log probability."""
           return self._jit_log_prob(value, self.params)

Scikit-learn Integration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.base import BaseEstimator, TransformerMixin

   class SciStanPyTransformer(BaseEstimator, TransformerMixin):
       """Scikit-learn compatible transformer using SciStanPy."""

       def __init__(self, model_builder, inference_method='vi'):
           self.model_builder = model_builder
           self.inference_method = inference_method
           self.fitted_model = None

       def fit(self, X, y=None):
           """Fit SciStanPy model using scikit-learn interface."""
           # Build SciStanPy model
           model = self.model_builder(X, y)

           # Run inference
           if self.inference_method == 'vi':
               self.results = model.variational()
           else:
               self.results = model.sample()

           self.fitted_model = model
           return self

       def transform(self, X):
           """Transform data using fitted model."""
           if self.fitted_model is None:
               raise ValueError("Model not fitted yet")

           # Use fitted model for transformation/prediction
           return self.fitted_model.predict(X, self.results)

Workflow Integration
-------------------

Snakemake Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Snakemake workflow file
   rule bayesian_analysis:
       input:
           data="data/experiment_{sample}.csv"
       output:
           results="results/analysis_{sample}.pkl",
           plots="plots/diagnostics_{sample}.png"
       script:
           "scripts/scistanpy_analysis.py"

   # Analysis script
   def run_bayesian_analysis(input_file, output_file, plot_file):
       """Run SciStanPy analysis in Snakemake workflow."""
       # Load data
       data = pd.read_csv(input_file)

       # Build and run model
       model = build_model(data)
       results = model.sample()

       # Save results
       with open(output_file, 'wb') as f:
           pickle.dump(results, f)

       # Generate diagnostics
       fig = model.plot_diagnostics(results)
       fig.savefig(plot_file)

Jupyter Integration
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from IPython.display import display, HTML
   import ipywidgets as widgets

   class InteractiveSciStanPy:
       """Interactive SciStanPy interface for Jupyter."""

       def __init__(self, model):
           self.model = model
           self.setup_widgets()

       def setup_widgets(self):
           """Create interactive widgets for model parameters."""
           self.param_widgets = {}
           for param_name, param in self.model.parameters.items():
               if hasattr(param, 'mu') and hasattr(param, 'sigma'):
                   # Create sliders for normal parameters
                   mu_slider = widgets.FloatSlider(
                       value=param.mu.value if hasattr(param.mu, 'value') else 0,
                       min=-10, max=10, step=0.1,
                       description=f'{param_name}_mu'
                   )
                   sigma_slider = widgets.FloatSlider(
                       value=param.sigma.value if hasattr(param.sigma, 'value') else 1,
                       min=0.1, max=5, step=0.1,
                       description=f'{param_name}_sigma'
                   )
                   self.param_widgets[param_name] = (mu_slider, sigma_slider)

       def interactive_analysis(self):
           """Create interactive analysis interface."""
           @widgets.interact_manual
           def update_analysis(**kwargs):
               # Update model parameters
               self.update_model_parameters(kwargs)

               # Run inference
               results = self.model.sample(n_samples=500)

               # Display results
               self.display_results(results)

           return update_analysis

Plugin System
-------------

Plugin Architecture
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from abc import ABC, abstractmethod

   class SciStanPyPlugin(ABC):
       """Base class for SciStanPy plugins."""

       @abstractmethod
       def get_name(self) -> str:
           """Return plugin name."""
           pass

       @abstractmethod
       def get_version(self) -> str:
           """Return plugin version."""
           pass

       @abstractmethod
       def register_components(self, registry):
           """Register plugin components with SciStanPy."""
           pass

   class AstronomyPlugin(SciStanPyPlugin):
       """Plugin for astronomy-specific functionality."""

       def get_name(self):
           return "SciStanPy-Astronomy"

       def get_version(self):
           return "1.0.0"

       def register_components(self, registry):
           """Register astronomy distributions and transformations."""
           registry.register_distribution('schechter', SchechterFunction)
           registry.register_transformation('luminosity_distance', LuminosityDistance)
           registry.register_coordinate_system('galactic', GalacticCoordinates)

**Plugin Registry:**

.. code-block:: python

   class PluginRegistry:
       """Registry for managing SciStanPy plugins."""

       def __init__(self):
           self.distributions = {}
           self.transformations = {}
           self.inference_engines = {}
           self.loaded_plugins = {}

       def load_plugin(self, plugin_class):
           """Load and register a plugin."""
           plugin = plugin_class()
           plugin.register_components(self)
           self.loaded_plugins[plugin.get_name()] = plugin

       def register_distribution(self, name, dist_class):
           """Register a custom distribution."""
           self.distributions[name] = dist_class

       def get_distribution(self, name):
           """Get registered distribution by name."""
           return self.distributions.get(name)

Testing Extensions
-----------------

Extension Testing Framework
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from scistanpy.testing import ExtensionTestSuite

   class TestCustomDistribution(ExtensionTestSuite):
       """Test suite for custom distributions."""

       def test_parameter_validation(self):
           """Test parameter validation for custom distribution."""
           # Test valid parameters
           dist = CustomDistribution(param1=1.0, param2=2.0)
           assert dist.param1 == 1.0

           # Test invalid parameters
           with pytest.raises(ValueError):
               CustomDistribution(param1=-1.0, param2=2.0)

       def test_stan_code_generation(self):
           """Test Stan code generation."""
           dist = CustomDistribution(param1=1.0, param2=2.0)
           stan_code = dist.get_stan_code()

           # Validate generated Stan code
           assert "custom_distribution" in stan_code
           assert self.validate_stan_syntax(stan_code)

       def test_numerical_accuracy(self):
           """Test numerical accuracy against reference implementation."""
           dist = CustomDistribution(param1=1.0, param2=2.0)

           # Compare against reference
           test_values = [0.1, 0.5, 1.0, 2.0]
           for val in test_values:
               ssp_result = dist.log_prob(val)
               ref_result = reference_implementation(val, 1.0, 2.0)
               assert abs(ssp_result - ref_result) < 1e-10

Documentation for Extensions
---------------------------

Extension Documentation Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rst

   Custom Extension Documentation
   =============================

   Brief description of what the extension provides and its scientific use case.

   Installation
   -----------

   .. code-block:: bash

      pip install scistanpy-extension-name

   Quick Start
   ----------

   .. code-block:: python

      import scistanpy as ssp
      from scistanpy_extension import CustomComponent

      # Example usage
      component = CustomComponent(param1=value1, param2=value2)

   API Reference
   ------------

   .. autoclass:: scistanpy_extension.CustomComponent
      :members:
      :undoc-members:
      :show-inheritance:

   Examples
   -------

   Provide comprehensive examples showing:
   - Basic usage
   - Integration with SciStanPy models
   - Scientific applications
   - Performance considerations

This comprehensive guide provides the foundation for extending SciStanPy across multiple dimensions, from simple custom distributions to complex domain-specific plugins and backend integrations.
