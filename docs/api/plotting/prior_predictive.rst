Prior Predictive API Reference
==============================

This reference covers the prior predictive simulation functionality in SciStanPy.

Prior Predictive Module
-----------------------

.. automodule:: scistanpy.plotting.prior_predictive
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Prior Predictive Simulation
---------------------------

Prior predictive simulation is a crucial step in Bayesian model development that allows you to:

- **Validate model specifications**: Check if your priors produce reasonable data
- **Detect modeling errors**: Identify issues before fitting to real data
- **Understand prior implications**: See what your priors actually imply about observable quantities
- **Improve model design**: Iterate on priors based on simulated outcomes

**Integration with SciStanPy Models:**

.. code-block:: python

   import scistanpy as ssp
   import numpy as np

   # Define a simple measurement model
   true_value = ssp.parameters.Normal(mu=10, sigma=2)  # Prior on true value
   measurement_error = ssp.parameters.LogNormal(mu=np.log(0.5), sigma=0.3)

   # Likelihood connects parameters to observations
   likelihood = ssp.parameters.Normal(mu=true_value, sigma=measurement_error)

   # Build model
   model = ssp.Model(likelihood)

   # Prior predictive simulation
   prior_predictions = model.prior_predictive(n_samples=1000)

   # Examine the range of possible data under your priors
   print(f"Predicted data range: [{prior_predictions.min():.2f}, {prior_predictions.max():.2f}]")

**Workflow Integration:**

.. code-block:: python

   # Typical prior predictive workflow

   # 1. Check if predictions are reasonable
   prior_samples = model.prior_predictive(n_samples=500)

   # 2. Compare to domain knowledge
   if prior_samples.min() < 0:
       print("Warning: Model predicts negative values - check priors")

   # 3. Visualize prior predictions
   import matplotlib.pyplot as plt
   plt.hist(prior_samples, bins=50, alpha=0.7)
   plt.title('Prior Predictive Distribution')
   plt.xlabel('Predicted Values')
   plt.ylabel('Frequency')

   # 4. Iterate on priors if needed
   # Adjust priors based on prior predictive results...

**Advanced Usage:**

.. code-block:: python

   # Prior predictive for hierarchical models
   class HierarchicalModel(ssp.Model):
       def __init__(self, group_data):
           # Group-level parameters
           global_mean = ssp.parameters.Normal(mu=0, sigma=5)
           group_variance = ssp.parameters.LogNormal(mu=0, sigma=1)

           # Individual group parameters
           group_means = ssp.parameters.Normal(
               mu=global_mean,
               sigma=group_variance,
               shape=(len(group_data),)
           )

           # Likelihood for each group
           likelihoods = []
           for i, data in enumerate(group_data):
               likelihood = ssp.parameters.Normal(mu=group_means[i], sigma=1.0)
               likelihood.observe(data)
               likelihoods.append(likelihood)

           super().__init__(likelihoods)

   # Prior predictive captures both levels of hierarchy
   hierarchical_model = HierarchicalModel(group_data)
   prior_predictions = hierarchical_model.prior_predictive(n_samples=1000)

**Model Validation:**

.. code-block:: python

   # Use prior predictive to validate model assumptions

   def validate_prior_predictions(model, expected_range, n_samples=1000):
       """Validate that prior predictions fall within expected range."""

       predictions = model.prior_predictive(n_samples=n_samples)

       # Check range
       pred_min, pred_max = predictions.min(), predictions.max()
       exp_min, exp_max = expected_range

       if pred_min < exp_min or pred_max > exp_max:
           print(f"Warning: Predictions [{pred_min:.2f}, {pred_max:.2f}] "
                 f"outside expected range [{exp_min:.2f}, {exp_max:.2f}]")
           return False

       print(f"✓ Prior predictions within expected range")
       return True

   # Validate temperature measurement model
   temp_model = ssp.Model(temperature_likelihood)
   validate_prior_predictions(temp_model, expected_range=(15, 35))  # °C

**Computational Considerations:**

The prior predictive functionality:

- **Backend Agnostic**: Works with NumPy, PyTorch, and Stan backends
- **Memory Efficient**: Streams samples when possible to handle large simulations
- **Parallel Compatible**: Can leverage multiple cores for large models
- **Cached Results**: Optionally cache results for repeated analysis

**Best Practices:**

1. **Always check prior predictions** before fitting real data
2. **Use domain knowledge** to validate prediction ranges
3. **Iterate on priors** based on prior predictive results
4. **Document assumptions** revealed by prior predictive analysis
5. **Compare predictions** with actual data patterns when available
6. **Test extreme scenarios** by examining tail behavior of predictions

Prior Predictive
================

This page documents the interactive prior-predictive visualization helper
implemented in `scistanpy.plotting.prior_predictive`.

.. automodule:: scistanpy.plotting.prior_predictive
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The module implements an interactive dashboard for conducting prior
predictive checks on SciStanPy models. The primary public class is
`PriorPredictiveCheck` which creates a Panel/HoloViews-based interface
that lets users draw from the model priors, inspect simulated outputs,
and interactively adjust toggleable model constants.

Key features (implemented)
- An interactive dashboard with widgets generated from model hyperparameters (toggleable constants are exposed as sliders).
- Multiple plot modes supported: ECDF, KDE, Violin, and Relationship plots.
- Reactive controls to select target variables, grouping dimensions, and independent variables for visualization.
- Methods to run the full pipeline (update model, draw samples, process data, and update the plot) as well as to update only the plot.
- Output is an interactive Panel layout suitable for Jupyter or Panel apps.

Primary class and methods
-------------------------

- PriorPredictiveCheck(model, copy_model: bool = False)
  - Constructs the dashboard for the provided SciStanPy model.

- display() -> panel.Row
  - Assemble and return the Panel layout for display.

- _init_float_sliders()
  - Create sliders for toggleable constants found in the model.

- _update_model()
  - Apply slider values back to the model constants.

- _draw_data()
  - Draw prior predictive samples from the model and store them as an xarray.Dataset.

- _process_data()
  - Convert the drawn xarray data into a pandas DataFrame adapted to the currently selected plot type and grouping choices.

- _update_plot(event=None)
  - Re-render the current plot using hvplot/HoloViews based on processed data.

- _full_pipeline(event=None)
  - Run the full flow: update model constants, draw new data, and refresh plot.

Dependencies and notes
----------------------

- The dashboard uses Panel and HoloViews (hvplot) for widgets and plots.
  Installing these packages is required for the interactive dashboard to work.
- The implementation expects the provided SciStanPy model to expose
  component metadata (e.g., `named_model_components_dict`, `observables`) and
  to support the `draw` method for prior sampling.

Minimal usage example
---------------------

.. code-block:: python

   import scistanpy as ssp
   from scistanpy.plotting.prior_predictive import PriorPredictiveCheck

   model = MySciStanPyModel(...)  # your model subclass
   check = PriorPredictiveCheck(model, copy_model=True)
   dashboard = check.display()
   # In a Jupyter notebook the dashboard can be displayed inline:
   dashboard

See also
--------

- :doc:`index` for the plotting package overview
