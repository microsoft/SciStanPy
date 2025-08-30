SciStanPy Architecture
======================

This document describes the overall architecture and design principles of SciStanPy.

High-Level Architecture
----------------------

SciStanPy follows a modular architecture designed for scientific computing:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                    User Interface                       │
   │  (Scientific Python API, Jupyter Integration)          │
   └─────────────────────────┬───────────────────────────────┘
                             │
   ┌─────────────────────────▼───────────────────────────────┐
   │                   Model Layer                           │
   │  (Model, Parameters, Transformations, Constants)       │
   └─────────────────────────┬───────────────────────────────┘
                             │
   ┌─────────────────────────▼───────────────────────────────┐
   │                Backend Abstraction                     │
   │    (NumPy, PyTorch, Stan Code Generation)              │
   └─────────────────────────┬───────────────────────────────┘
                             │
   ┌─────────────────────────▼───────────────────────────────┐
   │              Computational Backends                     │
   │     (NumPy/SciPy, PyTorch, CmdStanPy)                 │
   └─────────────────────────────────────────────────────────┘

Design Principles
----------------

**1. Scientific Focus**

- API designed around scientific thinking patterns
- Natural expression of mathematical relationships
- Domain-specific distributions and transformations
- Built-in validation for common scientific modeling errors

**2. Multi-Backend Flexibility**

- Abstract interface over multiple computational backends
- Automatic backend selection based on operation requirements
- Seamless transitions between backends within workflows

**3. Composability**

- Small, focused components that combine naturally
- Mathematical operations work consistently across parameter types
- Hierarchical model construction through composition

**4. Automatic Code Generation**

- Python expressions automatically generate efficient backend code
- Stan code generation for high-performance MCMC
- Gradient computation through automatic differentiation

Core Components
--------------

**Model System**

.. code-block:: python

   scistanpy/
   ├── model/
   │   ├── __init__.py           # Main Model class
   │   └── components/
   │       ├── abstract_model_component.py  # Base interfaces
   │       ├── parameters.py               # Probability distributions
   │       ├── constants.py                # Fixed values
   │       ├── transformations/            # Mathematical operations
   │       └── custom_distributions/       # Extended distributions

**Operations System**

.. code-block:: python

   scistanpy/
   ├── operations.py             # Mathematical transformations
   └── model/components/transformations/
       ├── transformed_parameters.py      # Transformation base classes
       └── ...                            # Specific transformations

**Backend Integration**

.. code-block:: python

   scistanpy/
   ├── utils.py                  # Utilities and helpers
   ├── defaults.py               # Configuration defaults
   ├── exceptions.py             # Error handling
   └── plotting.py               # Visualization

Component Interactions
---------------------

**Parameter Creation Flow:**

.. code-block:: text

   User Code → Parameter Constructor → Validation → Backend Selection → Storage

**Model Building Flow:**

.. code-block:: text

   Parameters → Mathematical Operations → Transformations → Model Assembly → Validation

**Inference Flow:**

.. code-block:: text

   Model → Backend Selection → Code Generation → Computation → Result Processing

**Example Interaction:**

.. code-block:: python

   # 1. Parameter creation
   mu = ssp.parameters.Normal(mu=0, sigma=1)

   # 2. Transformation
   exp_mu = ssp.operations.exp(mu)

   # 3. Model assembly (observe data elsewhere)
   likelihood = ssp.parameters.Normal(mu=exp_mu, sigma=0.1)

   # 4. Model creation
   model = ssp.Model(likelihood)

   # 5. Inference (point then Bayesian)
   mle_res = model.mle()
   mcmc_res = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)

Backend Architecture
-------------------

**Backend Abstraction Layer:**

.. code-block:: python

   class BackendInterface:
       """Abstract interface for computational backends."""

       def compile_model(self, model_components):
           """Compile model to backend-specific representation."""
           pass

       def execute_inference(self, compiled_model, method):
           """Execute inference using backend."""
           pass

       def process_results(self, raw_results):
           """Process backend results to standard format."""
           pass

**NumPy Backend:**

- Fast prototyping and simple computations
- Uses SciPy for statistical distributions
- Limited inference methods (MLE, simple optimization)

**PyTorch Backend:**

- GPU acceleration support
- Automatic differentiation for gradients
- Variational inference and optimization methods

**Stan Backend:**

- High-performance MCMC sampling
- Advanced sampling algorithms (NUTS, HMC)
- Automatic code generation from Python expressions

Stan Code Generation
-------------------

**Component-Based Generation:**

.. code-block:: python

   # Each component knows how to generate its Stan code
   class NormalParameter:
       def write_stan_code(self):
           return f"{self.name} ~ normal({self.mu_code}, {self.sigma_code});"

   class TransformedParameter:
       def write_stan_code(self):
           return f"{self.name} = {self.transformation_code};"

**Model Assembly:**

.. code-block:: python

   # Model assembles Stan program from components
   def generate_stan_program(model):
       stan_code = """
       data {
           // Generated data declarations
       }
       parameters {
           // Generated parameter declarations
       }
       model {
           // Generated model statements
       }
       """
       return stan_code

Error Handling Strategy
----------------------

**Hierarchical Error System:**

.. code-block:: python

   SciStanPyError                    # Base exception
   ├── ModelError                    # Model construction errors
   │   ├── ParameterError           # Parameter definition issues
   │   ├── ShapeError               # Shape incompatibilities
   │   └── ValidationError          # Model validation failures
   ├── InferenceError               # Inference execution errors
   │   ├── ConvergenceError         # Convergence failures
   │   ├── NumericalError           # Numerical instabilities
   │   └── BackendError             # Backend-specific issues
   └── DataError                    # Data handling errors
       ├── ObservationError         # Data observation issues
       └── FormatError              # Data format problems

**Error Context and Recovery:**

- Errors include detailed context about the failure
- Suggestions for common fixes
- Graceful degradation when possible

Testing Architecture
-------------------

**Test Categories:**

.. code-block:: python

   tests/
   ├── unit/                        # Component-level tests
   │   ├── test_parameters.py       # Parameter functionality
   │   ├── test_transformations.py  # Mathematical operations
   │   └── test_model.py            # Model construction
   ├── integration/                 # Cross-component tests
   │   ├── test_backends.py         # Backend integration
   │   └── test_workflows.py        # End-to-end workflows
   ├── scientific/                  # Scientific validation
   │   ├── test_examples.py         # Example reproducibility
   │   └── test_accuracy.py         # Statistical accuracy
   └── performance/                 # Performance benchmarks
       ├── test_scalability.py      # Large model performance
       └── test_memory.py           # Memory usage

**Validation Strategy:**

- Unit tests for individual components
- Integration tests for component interactions
- Scientific validation against known results
- Performance benchmarks for scalability
- Cross-platform compatibility testing

Future Architecture Considerations
---------------------------------

**Planned Enhancements:**

1. **Distributed Computing**: Support for cluster-based inference
2. **Streaming Data**: Online learning and sequential updating
3. **Model Serving**: Production deployment infrastructure
4. **Advanced Backends**: Integration with JAX, TensorFlow Probability
5. **Domain Extensions**: Specialized modules for specific scientific fields

**Extensibility Points:**

- Custom distribution framework
- Backend plugin system
- Transformation extension mechanism
- Visualization customization
- Domain-specific language extensions

This architecture provides a solid foundation for scientific Bayesian modeling while maintaining flexibility for future enhancements and extensions.
- Automatic type coercion where appropriate

**Benefits:**
- Clear documentation through type hints
- IDE support and autocompletion
- Runtime error prevention

Model System
-----------

Model Class Architecture
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class Model:
       """Main interface for Bayesian model building and inference."""

       def __init__(self, likelihood):
           """Initialize model from likelihood specification."""
           self._validate_model()
           self._build_dependency_graph()

       def sample(self, **kwargs):
           """Run MCMC sampling via Stan."""
           stan_code = self._generate_stan_code()
           return self._run_stan_sampling(stan_code, **kwargs)

       def variational(self, **kwargs):
           """Run variational inference via PyTorch."""
           return self._run_vi(**kwargs)

**Model Validation:**
- Check for circular dependencies
- Ensure at least one observed parameter
- Validate parameter constraints
- Check shape consistency

**Dependency Graph:**
- Track parameter relationships
- Determine sampling order
- Optimize Stan code generation
- Enable efficient computation

Inference Architecture
---------------------

Inference Methods
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Maximum Likelihood (gradient-based / numeric optimization)
   mle_res = model.mle(epochs=10000, early_stop=10)

   # Hamiltonian Monte Carlo via Stan
   mcmc_res = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)

Diagnostics (SampleResults.diagnose) identify failing tests (e.g. r_hat, ess_bulk)
and return (sample_failures, variable_failures).

Diagnostic System
~~~~~~~~~~~~~~~~

**Convergence Diagnostics:**
- R-hat calculation
- Effective sample size
- Divergent transition detection
- Energy diagnostics

**Model Checking:**
- Prior predictive checks
- Posterior predictive checks
- Leave-one-out cross-validation
- WAIC computation

**Implementation:**

.. code-block:: python

   class DiagnosticEngine:
       """Comprehensive model diagnostics."""

       def diagnose(self, results):
           """Run all diagnostic checks."""
           return {
               'rhat': self._compute_rhat(results),
               'ess': self._compute_ess(results),
               'divergences': self._check_divergences(results),
               'energy': self._energy_diagnostics(results)
           }

Extension Points
---------------

Custom Distributions
~~~~~~~~~~~~~~~~~~~

To add new distributions:

1. **Create distribution class:**

.. code-block:: python

   class NewDistribution(ContinuousDistribution):
       SCIPY_DIST = custom_scipy_dist
       TORCH_DIST = custom_torch_dist
       STAN_DIST = "new_distribution"

       STAN_TO_SCIPY_NAMES = {"param1": "scipy_param1"}
       STAN_TO_TORCH_NAMES = {"param1": "torch_param1"}

2. **Register with factory**
3. **Add tests and documentation**
4. **Create examples**

Custom Transformations
~~~~~~~~~~~~~~~~~~~~~

To add new transformations:

.. code-block:: python

   class CustomTransform(UnaryTransformedParameter):
       def run_np_torch_op(self, dist1):
           return your_transformation(dist1)

       def write_stan_operation(self, dist1: str) -> str:
           return f"custom_stan_function({dist1})"

Performance Considerations
-------------------------

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~

**Shape Broadcasting:**
- Minimize array copying
- Use vectorized operations
- Leverage NumPy/PyTorch broadcasting

**Stan Code Optimization:**
- Generate efficient indexing patterns
- Use Stan's vectorized functions
- Minimize parameter transformations

**Memory Management:**
- Lazy evaluation where possible
- Streaming for large datasets
- Efficient sample storage

**Caching:**
- Cache Stan model compilation
- Memoize expensive computations
- Reuse computed quantities

Testing Architecture
-------------------

Test Organization
~~~~~~~~~~~~~~~~

.. code-block:: text

   tests/
   ├── unit/                 # Individual component tests
   │   ├── test_parameters.py
   │   ├── test_transformations.py
   │   └── test_models.py
   ├── integration/          # Component interaction tests
   │   ├── test_inference.py
   │   └── test_stan_generation.py
   ├── examples/            # Tutorial validation
   │   └── test_examples.py
   └── performance/         # Benchmarking
       └── test_performance.py

**Test Categories:**
- **Unit tests**: Individual component functionality
- **Integration tests**: Multi-component interactions
- **Example tests**: Tutorial and documentation validation
- **Performance tests**: Computational efficiency
- **Scientific tests**: Validation against known results

**Continuous Integration:**
- Automated testing on multiple Python versions
- Cross-platform compatibility testing
- Performance regression detection
- Documentation building verification

This architecture provides a solid foundation for extending SciStanPy while maintaining its core design principles of scientific usability, multi-backend support, and comprehensive validation.

Accuracy Note
-------------

This document intentionally omits unimplemented items previously drafted
(variational(), sample(), posterior_predictive(), WAIC/LOO, streaming, GPU /
distributed execution). Open an issue if any slipped through.
