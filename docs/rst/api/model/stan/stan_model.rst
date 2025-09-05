Stan Model API Reference
========================

This reference covers the Stan probabilistic programming language integration and code generation functionality in SciStanPy.

Stan Model Module
-----------------

.. automodule:: scistanpy.model.stan.stan_model

   :members:
   :undoc-members:
   :show-inheritance:

Stan Code Organization Classes
------------------------------

Abstract Base Class
~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.stan.stan_model.StanCodeBase

   :members:
   :undoc-members:
   :show-inheritance:

   **Code Structure Management:**

   The StanCodeBase class provides the foundation for organizing Stan code into hierarchical structures:

   .. code-block:: python

      # Example of code organization hierarchy
      program = StanProgram(model)

      # Access nested structure
      for loop in program.recurse_for_loops():
          print(f"Loop depth: {loop.depth}")
          for component in loop.model_components:
              print(f"  Component: {component.model_varname}")

For-Loop Management
~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.stan.stan_model.StanForLoop

   :members:
   :undoc-members:
   :show-inheritance:

   **Loop Optimization Features:**

   .. code-block:: python

      # Automatic loop range calculation
      # SciStanPy analyzes component dimensions to determine loop bounds

      # Loop combination optimization
      # Compatible adjacent loops are automatically combined for efficiency

      # Singleton elimination
      # Loops with only one iteration are removed for cleaner code

Stan Program Generation
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.stan.stan_model.StanProgram

   :members:
   :undoc-members:
   :show-inheritance:

   **Complete Program Assembly:**

   .. code-block:: python

      # Generate complete Stan program
      program = StanProgram(model)

      # Access generated code blocks
      print("Functions block:")
      print(program.functions_block)

      print("Data block:")
      print(program.data_block)

      print("Parameters block:")
      print(program.parameters_block)

      print("Model block:")
      print(program.model_block)

      # Get complete program
      complete_code = program.code

Enhanced Stan Model Class
-------------------------

.. autoclass:: scistanpy.model.stan.stan_model.StanModel

   :members:
   :undoc-members:
   :show-inheritance:

   **Key Features:**

   - **Automatic Code Generation**: Translates SciStanPy models to Stan
   - **Enhanced Sampling**: Prior initialization and result processing
   - **Data Management**: Automatic gathering of constants and validation
   - **Result Integration**: Comprehensive post-processing and analysis

   **Basic Usage:**

   .. code-block:: python

      import scistanpy as ssp

      # Define SciStanPy model
      class MyModel(ssp.Model):
          def __init__(self):
              super().__init__()
              self.mu = ssp.parameters.Normal(mu=0, sigma=1)
              self.sigma = ssp.parameters.LogNormal(mu=0, sigma=0.5)
              self.y = ssp.parameters.Normal(mu=self.mu, sigma=self.sigma, observable=True)

      # Compile to Stan
      model = MyModel()
      stan_model = model.to_stan(output_dir='./stan_models')

      # Sample using enhanced interface
      results = stan_model.sample(
          data={'y': observed_data},
          chains=4,
          iter_sampling=2000,
          inits='prior'  # Use prior-based initialization
      )

   **Advanced Configuration:**

   .. code-block:: python

      # Custom compilation options
      stan_model = model.to_stan(
          output_dir='./compiled_models',
          force_compile=True,
          stanc_options={'O1': True},  # Optimization level 1
          cpp_options={'STAN_THREADS': True},  # Enable threading
          user_header='#include <custom_functions.hpp>',
          model_name='my_custom_model'
      )

      # Enhanced sampling with memory management
      results = stan_model.sample(
          data={'y': large_dataset},
          chains=8,
          iter_sampling=5000,
          precision='single',  # Memory optimization
          mib_per_chunk=128,   # Chunk size for large arrays
          use_dask=True       # Distributed processing
      )

Automatic Data Gathering
------------------------

**Data Management Features:**

.. code-block:: python

   # Automatic data gathering from model
   gathered_data = stan_model.gather_inputs(
       y=observed_values,  # User provides observables
       # Constants automatically gathered from model
       # Hyperparameters automatically extracted
   )

   # Access auto-gathered data
   constants = stan_model.autogathered_data
   print(f"Auto-gathered: {list(constants.keys())}")

**Data Validation:**

.. code-block:: python

   # Shape validation
   try:
       results = stan_model.sample(data={'y': wrong_shape_data})
   except ValueError as e:
       print(f"Shape validation failed: {e}")

   # Missing observable detection
   try:
       results = stan_model.sample(data={})  # Missing required observables
   except ValueError as e:
       print(f"Missing observables: {e}")

Stan Code Generation Process
----------------------------

**Code Generation Pipeline:**

1. **Dependency Analysis**: Build component dependency graph
2. **Depth Assignment**: Determine nesting levels for components
3. **Loop Organization**: Create optimized for-loop structures
4. **Block Generation**: Generate all Stan program blocks
5. **Code Formatting**: Apply Stan canonical formatting

**Generated Stan Blocks:**

.. code-block:: python

   # Access individual Stan blocks
   program = StanProgram(model)

   # Functions block (if custom functions needed)
   functions = program.functions_block

   # Data block (observables and constants)
   data = program.data_block

   # Transformed data block (data preprocessing)
   transformed_data = program.transformed_data_block

   # Parameters block (model parameters)
   parameters = program.parameters_block

   # Transformed parameters block (derived quantities)
   transformed_parameters = program.transformed_parameters_block

   # Model block (log-probability statements)
   model = program.model_block

   # Generated quantities block (posterior predictions)
   generated_quantities = program.generated_quantities_block

Stan Function Libraries
-----------------------

**Automatic Function Integration:**

SciStanPy automatically includes required Stan function libraries based on model components:

- **Multinomial Functions**: Enhanced multinomial distributions with multiple parameterizations
- **Exp-Distributions**: Log-transformed distributions (Exp-Exponential, Exp-Dirichlet, etc.)
- **Growth Models**: Specialized functions for temporal modeling
- **Sequence Operations**: Convolution and pattern matching functions

**Function Include Process:**

.. code-block:: python

   # Functions are automatically included based on model components

   # If model contains MultinomialLogit:
   # - Includes multinomial.stanfunctions

   # If model contains ExpExponential:
   # - Includes expexponential.stanfunctions

   # If model contains growth operations:
   # - Includes appropriate growth functions

Performance Optimizations
-------------------------

**Loop Optimization:**

.. code-block:: python

   # Automatic loop combination
   # Adjacent compatible loops are merged for efficiency

   # Singleton elimination
   # Single-iteration loops are removed

   # Depth optimization
   # Nested loops are organized for optimal Stan execution

**Memory Management:**

.. code-block:: python

   # Large dataset handling
   results = stan_model.sample(
       data=large_data,
       precision='single',    # Reduce memory usage
       mib_per_chunk=64,     # Control chunk sizes
       use_dask=True         # Enable distributed processing
   )

**Compilation Caching:**

.. code-block:: python

   # Efficient recompilation
   stan_model = model.to_stan(
       output_dir='./cache',
       force_compile=False   # Reuse cached compilation
   )

Integration with CmdStanPy
--------------------------

**Enhanced CmdStanPy Methods:**

All CmdStanPy methods are enhanced with automatic data gathering:

.. code-block:: python

   # Enhanced sampling
   mcmc_results = stan_model.sample(data={'y': observations})

   # Enhanced optimization (experimental)
   mle_results = stan_model.optimize(data={'y': observations})

   # Enhanced variational inference (experimental)
   vi_results = stan_model.variational(data={'y': observations})

   # Enhanced generated quantities (experimental)
   gq_results = stan_model.generate_quantities(
       fitted_params=mcmc_results,
       data={'y': new_observations}
   )

**Result Processing:**

.. code-block:: python

   # Comprehensive result objects
   results = stan_model.sample(data={'y': observations})

   # Access processed results
   summary = results.summary()
   diagnostics = results.diagnose()

   # ArviZ integration
   inference_data = results.to_arviz()

   # Custom analysis
   posterior_mean = results.posterior.mean()

Error Handling and Debugging
----------------------------

**Comprehensive Error Checking:**

.. code-block:: python

   # Model validation during compilation
   try:
       stan_model = model.to_stan()
   except ValueError as e:
       print(f"Model compilation failed: {e}")

   # Data validation during sampling
   try:
       results = stan_model.sample(data={'y': invalid_data})
   except ValueError as e:
       print(f"Data validation failed: {e}")

**Stan Code Inspection:**

.. code-block:: python

   # Inspect generated Stan code
   print("Generated Stan program:")
   print(stan_model.code())

   # Access Stan file location
   print(f"Stan file: {stan_model.stan_program_path}")

   # Inspect variable mappings
   varname_mapping = stan_model.get_varnames_to_dimnames()
   print(f"Variable dimensions: {varname_mapping}")

Best Practices
--------------

1. **Use output directories** for Stan file caching and debugging
2. **Enable prior initialization** for better MCMC convergence
3. **Monitor memory usage** with precision and chunking options
4. **Validate data shapes** before large sampling runs
5. **Inspect generated Stan code** for debugging and optimization
6. **Use appropriate precision** based on computational requirements
7. **Cache compiled models** for repeated use with consistent output directories

The Stan integration provides a complete bridge between SciStanPy's intuitive modeling interface and Stan's high-performance probabilistic programming capabilities.
