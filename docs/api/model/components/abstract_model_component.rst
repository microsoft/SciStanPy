Abstract Model Component API Reference
======================================

This reference covers the abstract base classes that form the foundation of SciStanPy's model component architecture.

Abstract Model Component Module
-------------------------------

.. automodule:: scistanpy.model.components.abstract_model_component
   :members:
   :undoc-members:
   :show-inheritance:

Core Abstract Classes
--------------------

AbstractModelComponent
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scistanpy.model.components.abstract_model_component.AbstractModelComponent
   :members:
   :undoc-members:
   :show-inheritance:

   **Foundation of All Model Components:**

   The AbstractModelComponent class provides the essential infrastructure for all SciStanPy model components:

   .. code-block:: python

      import scistanpy as ssp
      import numpy as np

      # All model components inherit from AbstractModelComponent
      mu = ssp.parameters.Normal(mu=0, sigma=1)
      sigma = ssp.parameters.LogNormal(mu=0, sigma=0.5)
      y = ssp.parameters.Normal(mu=mu, sigma=sigma, observable=True)

      # Access component relationships
      print(f"Y has parents: {[p.model_varname for p in y.parents]}")
      print(f"Mu has children: {[c.model_varname for c in mu.children]}")

   **Key Architectural Features:**

   .. code-block:: python

      # Shape inference and broadcasting
      multi_dim_param = ssp.parameters.Normal(
          mu=np.zeros((3, 4)),  # Shape automatically inferred
          sigma=1.0             # Automatically broadcasted
      )
      print(f"Parameter shape: {multi_dim_param.shape}")  # (3, 4)

      # Dependency tree traversal
      for depth, current, relative in y.walk_tree(walk_down=False):
          print(f"Depth {depth}: {current.model_varname} -> {relative.model_varname}")

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

Component Relationships
----------------------

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

Shape and Broadcasting System
----------------------------

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

**Multi-dimensional Indexing:**

.. code-block:: python

   # Advanced indexing support
   matrix_param = ssp.parameters.Normal(mu=0, sigma=1, shape=(10, 5))

   # Index into subcomponents
   row_slice = matrix_param[2, :]    # Third row
   column_slice = matrix_param[:, 1] # Second column
   element = matrix_param[3, 4]      # Single element

Stan Integration Framework
-------------------------

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

Error Handling and Validation
-----------------------------

**Comprehensive Parameter Validation:**

.. code-block:: python

   # Parameter constraint validation
   try:
       invalid_param = ssp.parameters.Beta(
           alpha=-1,  # Must be positive
           beta=2
       )
   except ValueError as e:
       print(f"Parameter validation: {e}")

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

Tree Traversal and Analysis
--------------------------

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

**Dependency Validation:**

.. code-block:: python

   def validate_dependencies(component):
       """Validate that component dependencies form a valid DAG."""

       visited = set()
       rec_stack = set()

       def has_cycle(comp):
           visited.add(comp)
           rec_stack.add(comp)

           for parent in comp.parents:
               if parent not in visited:
                   if has_cycle(parent):
                       return True
               elif parent in rec_stack:
                   return True

           rec_stack.remove(comp)
           return False

       return not has_cycle(component)

Advanced Usage Patterns
----------------------

**Custom Component Development:**

.. code-block:: python

   class CustomTransformation(ssp.model.components.AbstractModelComponent):
       """Example custom component implementation."""

       def __init__(self, input_param, **kwargs):
           super().__init__(input_param=input_param, **kwargs)

       def _draw(self, level_draws, seed):
           # Custom sampling logic
           input_value = level_draws['input_param']
           return np.custom_function(input_value)

       def get_right_side(self, index_opts, **kwargs):
           # Custom Stan code generation
           formattables = super().get_right_side(index_opts, **kwargs)
           return f"custom_function({formattables['input_param']})"

       def __str__(self):
           return f"{self.model_varname} = custom_function({self.input_param.model_varname})"

**Interactive Model Exploration:**

.. code-block:: python

   def interactive_model_explorer(component):
       """Create an interactive model exploration interface."""

       import panel as pn

       # Get all constants for sliders
       all_constants = []
       for _, current, _ in component.walk_tree(walk_down=False):
           all_constants.extend(current.constants.values())

       # Create sliders for togglable constants
       sliders = {}
       for const in all_constants:
           if const.is_togglable:
               sliders[const.model_varname] = pn.widgets.FloatSlider(
                   name=const.model_varname,
                   start=const.slider_start,
                   end=const.slider_end,
                   step=const.slider_step_size,
                   value=const.value.item() if const.value.ndim == 0 else const.value[0]
               )

       return sliders

Performance Considerations
-------------------------

**Efficient Sampling:**

.. code-block:: python

   # Batch sampling for efficiency
   large_param = ssp.parameters.Normal(mu=0, sigma=1, shape=(100, 50))

   # Single large draw vs multiple small draws
   # Efficient: Single large draw
   samples, _ = large_param.draw(n=1000)

   # Less efficient: Multiple small draws in loop
   # for i in range(1000):
   #     sample, _ = large_param.draw(n=1)

**Memory Management:**

.. code-block:: python

   # Be mindful of component relationships for memory
   # Shared parents are stored once
   shared_parent = ssp.parameters.Normal(mu=0, sigma=1)

   children = []
   for i in range(100):
       child = ssp.parameters.Normal(
           mu=shared_parent,  # Shared reference, not copied
           sigma=1.0
       )
       children.append(child)

Best Practices
-------------

1. **Understand the component hierarchy** before building complex models
2. **Use shape broadcasting** to avoid manual shape specification
3. **Validate model structure** with tree traversal methods
4. **Leverage automatic Stan code generation** rather than writing custom Stan
5. **Use descriptive variable names** for better model interpretability
6. **Check component relationships** to ensure model correctness
7. **Monitor memory usage** in models with many shared components

The AbstractModelComponent foundation enables sophisticated probabilistic modeling while maintaining mathematical rigor and computational efficiency across multiple backends.
