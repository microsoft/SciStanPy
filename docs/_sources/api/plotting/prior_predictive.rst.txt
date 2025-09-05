Prior Predictive
================

This page documents the interactive prior-predictive visualization helper
implemented in `scistanpy.plotting.prior_predictive`.

.. automodule:: scistanpy.plotting.prior_predictive
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
