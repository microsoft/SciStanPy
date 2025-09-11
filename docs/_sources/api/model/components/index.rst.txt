Model Components API Reference
==============================
.. automodule:: scistanpy.model.components
   :undoc-members:
   :show-inheritance:

Model Components Submodule Overview
-----------------------------------

The model components submodule is itself broken into additional submodules:

.. toctree::
   :maxdepth: 1

   abstract_model_component
   constants
   custom_distributions/index
   parameters
   transformations/index

Most relevant to the typical end user are the :py:mod:`~scistanpy.model.components.parameters`, :py:mod:`~scistanpy.model.components.constants`, and :py:mod:`~scistanpy.model.components.transformations.transformed_parameters` submodules, which provide concrete implementations of various types of model components. Note that the :py:mod:`~scistanpy.model.components.transformations.transformed_parameters` submodule is not typically imported directly, but rather its functionality is accessed through mathematical operations on other component types (e.g., inbuilt Python operators like ``+``, ``-``, ``*``, ``/``, and functions in :py:mod:`scistanpy.operations`).

The remainder of this page highlights the key design principles, architectural features, and usage patterns of the model components framework.

Key Design Principles
---------------------

**Compositional Design:**

All components follow a compositional design that enables complex model construction through combination of simple elements:

.. code-block:: python

   import scistanpy as ssp
   import numpy as np

   class MyModel(ssp.Model):
       def __init__(self, x_data, observations):

         # Record default data
         super().__init__(default_data = {"observed": observations})

         # Basic components
         self.intercept = ssp.parameters.Normal(mu=0.0, sigma=5.0)
         self.slope = ssp.parameters.Normal(mu=0.0, sigma=2.0)
         self.noise = ssp.parameters.LogNormal(mu=0.0, sigma=1.0)

         # Composition through mathematical operations
         linear_predictor = self.intercept + self.slope * x_data

         # Further composition
         self.observed = ssp.parameters.Normal(mu=linear_predictor, sigma=self.noise)

   model_instance = MyModel(x_data=np.linspace(0, 10, 50), observations=np.random.randn(50))

In the above example, simple components (``intercept``, ``slope``, ``noise``) are combined through arithmetic operations to create a more complex component (``linear_predictor``), which is then used as the mean of the observed data distribution.

**Automatic Dependency Tracking:**

Components automatically track their relationships to enable proper Stan code generation and sampling:

.. code-block:: python

   # Dependencies are tracked automatically
   print(f"Linear predictor depends on: {[p.model_varname for p in  model_instance.observed.parents]}")
   print(f"Intercept is used by: {[c.model_varname for c in model_instance.intercept.children]}")

**Shape Broadcasting:**

Components automatically handle shape broadcasting following NumPy conventions:

.. code-block:: python

   # Automatic shape inference and broadcasting
   scalar_param = ssp.parameters.Normal(mu=0.0, sigma=1.0)          # Shape: ()
   vector_param = ssp.parameters.Normal(mu=0.0, sigma=1.0, shape=(5,)) # Shape: (5,)

   # Broadcasting in operations
   broadcasted = scalar_param + vector_param  # Result shape: (5,)

   # Multi-dimensional broadcasting
   matrix_param = ssp.parameters.Normal(mu=0.0, sigma=1.0, shape=(3, 5))
   result = vector_param + matrix_param  # Result shape: (3, 5)

Additional Usage Patterns
-------------------------

**Parent-Child Architecture:**

.. code-block:: python

   # Building hierarchical relationships
   global_mean = ssp.parameters.Normal(mu=0, sigma=5)
   group_means = ssp.parameters.Normal(
       mu=global_mean,          # Parent relationship
       sigma=1.0,
       shape=(10,)              # 10 groups
   )
   observations = ssp.parameters.Normal(
       mu=group_means,          # Another parent relationship
       sigma=0.5,
       observable=True
   )

   # Explore relationships
   print(f"Global mean children: {len(global_mean.children)}")
   print(f"Group means parents: {[p.model_varname for p in group_means.parents]}")

**Dependency Graph Navigation:**

.. code-block:: python

   # Walk up the dependency tree
   def show_dependencies(component, level=0):
       indent = "  " * level
       print(f"{indent}{component.model_varname} ({component.__class__.__name__})")
       for parent in component.parents:
           show_dependencies(parent, level + 1)

   show_dependencies(observations)

**Stan Code Generation Framework:**

.. code-block:: python

   # Automatic Stan variable declarations
   stan_dtype = y.get_stan_dtype()  # "real" for scalar normal
   declaration = y.get_stan_parameter_declaration()

   # Multi-dimensional declarations
   matrix_param = ssp.parameters.Normal(mu=0, sigma=1, shape=(5, 3))
   matrix_decl = matrix_param.get_stan_dtype()  # "array[5] vector[3]"

**Sampling and Drawing Interface:**

   .. code-block:: python

      # Hierarchical sampling
      samples, all_draws = y.draw(n=1000)

      # Access all component draws
      for component, draws in all_draws.items():
          print(f"{component.model_varname}: shape {draws.shape}")

**Multi-dimensional Indexing:**

.. code-block:: python

   # Advanced indexing support
   matrix_param = ssp.parameters.Normal(mu=0, sigma=1, shape=(10, 5))

   # Index into subcomponents
   row_slice = matrix_param[2, :]    # Third row
   column_slice = matrix_param[:, 1] # Second column
   element = matrix_param[3, 4]      # Single element

**Model Structure Analysis:**

.. code-block:: python

   def analyze_model_structure(component):
       """Analyze the structure of a model component tree."""

       # Find all components in the tree
       components = set([component])
       for _, current, relative in component.walk_tree(walk_down=False):
           components.add(current)
           components.add(relative)

       # Categorize components
       parameters = [c for c in components if isinstance(c, ssp.parameters.Parameter)]
       constants = [c for c in components if isinstance(c, ssp.constants.Constant)]
       transforms = [c for c in components if hasattr(c, '_transformation')]

       print(f"Model structure analysis:")
       print(f"  Total components: {len(components)}")
       print(f"  Parameters: {len(parameters)}")
       print(f"  Constants: {len(constants)}")
       print(f"  Transformations: {len(transforms)}")

       return {
           'components': components,
           'parameters': parameters,
           'constants': constants,
           'transformations': transforms
       }


Performance Considerations
--------------------------

**Efficient Construction:**

.. code-block:: python

   # Efficient: Single multi-dimensional parameter
   efficient = ssp.parameters.Normal(mu=0.0, sigma=1.0, shape=(100, 50))

   # Less efficient: Many individual parameters
   # inefficient = [[ssp.parameters.Normal(mu=0.0, sigma=1.0)
   #                 for j in range(50)] for i in range(100)]


Other Notable Features
----------------------

**Automatic Validation:**

Components perform comprehensive validation during construction:

.. code-block:: python

   # Automatic parameter validation
   try:
       invalid_param = ssp.parameters.Beta(
           alpha=-1,  # Must be positive
           beta=2
       )
   except ValueError as e:
       print(f"Parameter validation failed: {e}")

   # Shape compatibility validation
   try:
       incompatible = ssp.parameters.Normal(
           mu=np.zeros((3, 4)),
           sigma=np.ones((5, 2)),  # Incompatible shape
       )
   except ValueError as e:
       print(f"Shape compatibility failed: {e}")

**Constraint Enforcement:**

Components automatically enforce distributional constraints:

.. code-block:: python

   # Constraint checking during sampling
   positive_param = ssp.parameters.Gamma(alpha=2, beta=1)
   samples, _ = positive_param.draw(n=100)
   assert np.all(samples >= 0)  # Automatic constraint enforcement

   # Simplex constraint enforcement
   simplex_param = ssp.parameters.Dirichlet(alpha=[1, 1, 1])
   simplex_samples, _ = simplex_param.draw(n=100)
   assert np.allclose(simplex_samples.sum(axis=-1), 1)

**Automatic Shape Inference:**

.. code-block:: python

   # Broadcasting follows NumPy rules
   a = ssp.parameters.Normal(mu=0, sigma=1, shape=(5, 1))
   b = ssp.parameters.Normal(mu=0, sigma=1, shape=(3,))

   # Combination automatically broadcasts to (5, 3)
   combined = a + b
   print(f"Broadcasted shape: {combined.shape}")

**Shape Validation:**

.. code-block:: python

   try:
       # Incompatible shapes raise clear errors
       incompatible = ssp.parameters.Normal(
           mu=np.zeros((3, 4)),
           sigma=np.ones((2, 5)),  # Incompatible shape
       )
   except ValueError as e:
       print(f"Shape error: {e}")

**Variable Declaration System:**

.. code-block:: python

   # Automatic Stan type inference
   real_param = ssp.parameters.Normal(mu=0, sigma=1)
   int_param = ssp.parameters.Poisson(lambda_=5)
   simplex_param = ssp.parameters.Dirichlet(alpha=[1, 1, 1])

   print(real_param.get_stan_dtype())     # "real"
   print(int_param.get_stan_dtype())      # "int<lower=0>"
   print(simplex_param.get_stan_dtype())  # "simplex[3]"

**Bound Constraint Handling:**

.. code-block:: python

   # Automatic bound detection
   positive_param = ssp.parameters.Gamma(alpha=2, beta=1)
   bounded_param = ssp.parameters.Beta(alpha=2, beta=3)

   print(positive_param.get_stan_dtype())  # "real<lower=0.0>"
   print(bounded_param.get_stan_dtype())   # "real<lower=0.0, upper=1.0>"

**Index Management for Multi-dimensional Arrays:**

.. code-block:: python

   # Automatic indexing for Stan loops
   param_3d = ssp.parameters.Normal(mu=0, sigma=1, shape=(4, 5, 3))

   # Get indexed variable name for Stan code
   index_opts = ('i', 'j', 'k')
   indexed_name = param_3d.get_indexed_varname(index_opts)
   # Result: "param_3d[i,j]" (last dimension vectorized)

**Shape Compatibility Checking:**

.. code-block:: python

   # Shape compatibility validation
   try:
       incompatible = ssp.parameters.Normal(
           mu=np.zeros((3, 4)),
           sigma=np.ones((5, 2)),  # Incompatible
           shape=(2, 2)            # Also incompatible
       )
   except ValueError as e:
       print(f"Shape compatibility: {e}")

**Bound Violation Detection:**

.. code-block:: python

   # Runtime bound checking during sampling
   param = ssp.parameters.Beta(alpha=1, beta=1)
   try:
       # This would violate Beta bounds during internal validation
       samples, _ = param.draw(n=100)
       # Automatic validation ensures samples ∈ (0, 1)
   except Exception as e:
       print(f"Bound violation: {e}")