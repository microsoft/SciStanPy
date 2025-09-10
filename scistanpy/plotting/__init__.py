# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Plotting utilities for SciStanPy visualization and analysis.

This subpackage provides a comprehensive suite of plotting functions designed
specifically for visualizing Bayesian model results, diagnostic plots, and
statistical relationships commonly encountered in scientific analysis workflows.
Users should not typically directly interact with these functions, which are used
internally to support plotting operations tied to results objects.

The plotting utilities are built on top of holoviews, providing interactive visualizations
with sensible defaults while maintaining full customization capabilities.

Key Functionality:

    - Distribution visualization and comparison
    - Model calibration and diagnostic plots
    - Quantile-quantile plots for model validation
    - Empirical cumulative distribution functions
    - Automatic datashading/hexagonal binning for large datasets
    - Relationship plotting with uncertainty quantification
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
