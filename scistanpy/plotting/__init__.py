# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Plotting utilities for SciStanPy visualization and analysis.

This subpackage provides a comprehensive suite of plotting functions designed
specifically for visualizing Bayesian model results, diagnostic plots, and
statistical relationships commonly encountered in scientific analysis workflows.

The plotting utilities are built on top of holoviews, providing interactive visualizations
with sensible defaults while maintaining full customization capabilities.

Key Functionality:
    - Distribution visualization and comparison
    - Model calibration and diagnostic plots
    - Quantile-quantile plots for model validation
    - Empirical cumulative distribution functions
    - Automatic datashading/hexagonal binning for large datasets
    - Relationship plotting with uncertainty quantification

All plotting functions are designed to work with SciStanPy model outputs and standard
scientific Python data structures (NumPy arrays, pandas DataFrames, etc.).

Functions:
    calculate_relative_quantiles: Compute relative quantiles for comparison plots
    hexgrid_with_mean: Create hexagonal binning plots with mean overlays
    plot_calibration: Generate model calibration diagnostic plots
    plot_distribution: Visualize probability distributions
    plot_ecdf_kde: Plot empirical CDFs with kernel density estimates
    plot_ecdf_violin: Combine ECDF plots with violin plot representations
    plot_relationship: Visualize relationships between variables with uncertainty
    quantile_plot: Create quantile-quantile plots for model validation

Example:
    >>> from scistanpy.plotting import plot_distribution, plot_calibration
    >>> # Visualize posterior distributions
    >>> plot_distribution(posterior_samples)
    >>> # Check model calibration
    >>> plot_calibration(reference_data, observed_data)
"""

from .plotting import (
    calculate_relative_quantiles,
    hexgrid_with_mean,
    plot_calibration,
    plot_distribution,
    plot_ecdf_kde,
    plot_ecdf_violin,
    plot_relationship,
    quantile_plot,
)
