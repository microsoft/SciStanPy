"""Maximum likelihood estimation utilities for SciStanPy models.

The module centers around two main classes: MLEParam for individual parameter
estimates and MLE for complete model results. These classes provide both the
estimated parameter values and the fitted probability distributions, enabling
downstream analysis including uncertainty quantification and posterior predictive
sampling.

Key Features:
    - Individual parameter MLE estimates with associated distributions
    - Complete model MLE results with loss tracking and diagnostics
    - Efficient sampling from fitted distributions with batch processing
    - Integration with ArviZ for Bayesian analysis workflows
    - Comprehensive visualization of optimization trajectories
    - Support for large-scale sampling with memory management

The module integrates with SciStanPy's broader ecosystem, providing results in
formats compatible with plotting utilities, inference frameworks, and downstream
analysis tools.

Performance Considerations:
    - Batch sampling prevents memory overflow for large sample requests
    - GPU acceleration is preserved through PyTorch distribution objects

The MLE results can be used for various purposes including model comparison,
uncertainty quantification, and as initialization for more sophisticated
inference procedures like MCMC sampling.
"""

from __future__ import annotations

import warnings

from typing import Literal, Optional, overload, TYPE_CHECKING

import arviz as az
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import xarray as xr

from scistanpy.model import results

if TYPE_CHECKING:
    from scistanpy import custom_types
    from scistanpy import model as ssp_model


class MLEParam:
    """Container for maximum likelihood estimate of a single model parameter.

    This class encapsulates the MLE result for an individual parameter,
    including the estimated value and the corresponding fitted probability
    distribution. It provides methods for sampling from the fitted distribution
    and accessing parameter properties.

    :param name: Name of the parameter in the model
    :type name: str
    :param value: Maximum likelihood estimate of the parameter value.
                 Can be None for some distribution types.
    :type value: Optional[npt.NDArray]
    :param distribution: Fitted probability distribution object
    :type distribution: custom_types.SciStanPyDistribution

    :ivar name: Parameter name identifier
    :ivar mle: Stored maximum likelihood estimate
    :ivar distribution: Fitted distribution for sampling and analysis

    The class maintains both point estimates and distributional representations,
    enabling both point-based analysis and uncertainty quantification through
    sampling from the fitted distribution.

    Example:
        >>> param = MLEParam('mu', np.array([2.5]), fitted_normal_dist)
        >>> samples = param.draw(1000, seed=42)
        >>> print(f"MLE estimate: {param.mle}")
    """

    def __init__(
        self,
        name: str,
        value: Optional[npt.NDArray],
        distribution: "custom_types.SciStanPyDistribution",
    ):

        # Store the inputs
        self.name = name
        self.mle = value
        self.distribution = distribution

    def draw(
        self,
        n: int,
        *,
        seed: Optional[custom_types.Integer] = None,
        batch_size: Optional[custom_types.Integer] = None,
    ) -> npt.NDArray:
        """Sample from the fitted parameter distribution.

        This method generates samples from the parameter's fitted probability
        distribution using batch processing to handle large sample requests.

        :param n: Total number of samples to generate
        :type n: int
        :param seed: Random seed for reproducible sampling. Defaults to None.
        :type seed: Optional[custom_types.Integer]
        :param batch_size: Size of batches for memory-efficient sampling.
                          Defaults to None (uses n as batch size).
        :type batch_size: Optional[custom_types.Integer]

        :returns: Array of samples from the fitted distribution
        :rtype: npt.NDArray

        Batch processing prevents memory overflow when requesting large numbers
        of samples from complex distributions, particularly important when
        working with GPU-based computations.

        Example:
            >>> # Generate 10000 samples in batches of 1000
            >>> samples = param.draw(10000, batch_size=1000, seed=42)
            >>> print(f"Sample mean: {samples.mean()}")
        """
        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # If the batch size is not provided, we set it to `n`
        batch_size = batch_size or n

        # Calculate the batch sizes for each sampling iteration
        batch_sizes = [batch_size] * (n // batch_size)
        if (n_remaining := n % batch_size) > 0:
            batch_sizes.append(n_remaining)

        # Sample from the distribution
        return np.concatenate(
            [
                self.distribution.sample((batch_size,)).detach().cpu().numpy()
                for batch_size in batch_sizes
            ]
        )


class MLE:
    """Complete maximum likelihood estimation results for a SciStanPy model.

    This class encapsulates the full results of MLE parameter estimation,
    including parameter estimates, fitted distributions, optimization
    diagnostics, and utilities for further analysis. It provides a
    comprehensive interface for working with MLE results.

    :param model: Original SciStanPy model
    :type model: ssp_model.Model
    :param mle_estimate: Dictionary of parameter names to their MLE values
    :type mle_estimate: dict[str, npt.NDArray]
    :param distributions: Dictionary of parameter names to fitted distributions
    :type distributions: dict[str, torch.distributions.Distribution]
    :param losses: Array of loss values throughout optimization
    :type losses: npt.NDArray
    :param data: Observed data used for parameter estimation
    :type data: dict[str, npt.NDArray]

    :ivar model: Reference to the original model
    :ivar data: Observed data used for fitting
    :ivar model_varname_to_mle: Mapping from parameter names to MLEParam objects
    :ivar losses: DataFrame containing loss trajectory and diagnostics

    :raises ValueError: If MLE estimate keys are not subset of distribution keys
    :raises ValueError: If parameter names conflict with existing attributes

    The class automatically creates attributes for each parameter, allowing
    direct access like `mle_result.mu` for a parameter named 'mu'. It also
    provides comprehensive utilities for visualization, sampling, and
    integration with Bayesian analysis workflows.

    Key Features:
    - Direct attribute access to individual parameter results
    - Comprehensive loss trajectory tracking and visualization
    - Efficient sampling from fitted parameter distributions
    - Integration with ArviZ for Bayesian workflow compatibility
    - Memory-efficient batch processing for large sample requests

    Example:
        >>> mle_result = model.mle(data=observed_data)
        >>> mu_samples = mle_result.mu.draw(1000)  # Direct parameter access
        >>> loss_plot = mle_result.plot_loss_curve()
        >>> inference_data = mle_result.get_inference_obj()
    """

    def __init__(
        self,
        model: "ssp_model.Model",
        mle_estimate: dict[str, npt.NDArray],
        distributions: dict[str, torch.distributions.Distribution],
        losses: npt.NDArray,
        data: dict[str, npt.NDArray],
    ):

        # The keys of the mle estimate should be a subset of the keys of the distributions
        if not set(mle_estimate.keys()).issubset(distributions.keys()):
            raise ValueError(
                "Keys of mle estimate should be a subset of the keys of the distributions"
            )

        # Record the model and data
        self.model = model
        self.data = data

        # Store inputs. Each key in the mle estimate will be mapped to an instance
        # variable
        self.model_varname_to_mle: dict[str, MLEParam] = {
            key: MLEParam(name=key, value=mle_estimate.get(key), distribution=value)
            for key, value in distributions.items()
        }

        # Set an attribute for all MLE parameters
        for k, v in self.model_varname_to_mle.items():
            if hasattr(self, k):
                raise ValueError(
                    f"MLE parameter {k} already exists in the model. Please rename it."
                )
            setattr(self, k, v)

        # Record the loss trajectory as a pandas dataframe
        self.losses = pd.DataFrame(
            {
                "-log pdf/pmf": losses,
                "iteration": np.arange(len(losses)),
                "shifted log(-log pdf/pmf)": losses - losses.min() + 1,
            },
        )

    def plot_loss_curve(self, logy: bool = True):
        """Generate interactive plot of the optimization loss trajectory.

        This method creates a visualization of how the loss function evolved
        during the optimization process, providing insights into convergence
        behavior and optimization effectiveness.

        :param logy: Whether to use logarithmic y-axis scaling. Defaults to True.
        :type logy: bool

        :returns: Interactive HoloViews plot of the loss curve

        The plot automatically handles:
        - Logarithmic scaling with proper handling of negative/zero values
        - Appropriate axis labels and titles based on scaling choice
        - Interactive features for detailed examination of convergence
        - Warning messages for problematic loss trajectories

        For logarithmic scaling with non-positive loss values, the method
        automatically switches to a shifted logarithmic scale to maintain
        visualization quality while issuing appropriate warnings.

        Example:
            >>> # Standard logarithmic loss plot
            >>> loss_plot = mle_result.plot_loss_curve()
            >>> # Linear scale loss plot
            >>> linear_plot = mle_result.plot_loss_curve(logy=False)
        """
        # Get y-label and title
        y = "-log pdf/pmf"
        if logy:
            if self.losses["-log pdf/pmf"].min() <= 0:
                warnings.warn("Negative values in loss curve. Using shifted log10.")
                y = "shifted log(-log pdf/pmf)"
                ylabel = y
            else:
                ylabel = "log(-log pdf/pmf)"
            title = "Log Loss Curve"
        else:
            ylabel = "-log pdf/pmf"
            title = "Loss Curve"

        return self.losses.hvplot.line(
            x="iteration", y=y, title=title, logy=logy, ylabel=ylabel
        )

    @overload
    def draw(
        self,
        n: custom_types.Integer,
        *,
        seed: Optional[custom_types.Integer],
        as_xarray: Literal[True],
        as_inference_data: Literal[False],
        batch_size: Optional[custom_types.Integer] = None,
    ) -> xr.Dataset: ...

    @overload
    def draw(
        self,
        n: custom_types.Integer,
        *,
        seed: Optional[custom_types.Integer],
        as_xarray: Literal[False],
        batch_size: Optional[custom_types.Integer] = None,
    ) -> dict[str, npt.NDArray]: ...

    def draw(self, n, *, seed=None, as_xarray=False, batch_size=None):
        """Generate samples from all fitted parameter distributions.

        This method draws samples from the fitted distributions of all model
        parameters. It supports multiple output formats for integration with
        different analysis workflows.

        :param n: Number of samples to draw from each parameter distribution
        :type n: custom_types.Integer
        :param seed: Random seed for reproducible sampling. Defaults to None.
        :type seed: Optional[custom_types.Integer]
        :param as_xarray: Whether to return results as xarray Dataset. Defaults to False.
        :type as_xarray: bool
        :param batch_size: Batch size for memory-efficient sampling. Defaults to None.
        :type batch_size: Optional[custom_types.Integer]

        :returns: Sampled parameter values in requested format
        :rtype: Union[dict[str, npt.NDArray], xr.Dataset]

        Output Formats:
        - Dictionary (default): Keys are parameter names, values are sample arrays
        - xarray Dataset: Structured dataset with proper dimension labels and coordinates

        This is particularly useful for:
        - Uncertainty propagation through model predictions
        - Bayesian model comparison and validation
        - Posterior predictive checking with MLE-based approximations
        - Sensitivity analysis of parameter estimates

        Example:
            >>> # Draw samples as dictionary
            >>> samples = mle_result.draw(1000, seed=42)
            >>> # Draw as structured xarray Dataset
            >>> dataset = mle_result.draw(1000, as_xarray=True, batch_size=100)
        """
        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Draw samples
        draws = {
            self.model.all_model_components_dict[k]: v.draw(n, batch_size=batch_size)
            for k, v in self.model_varname_to_mle.items()
        }

        # If returning as an xarray or InferenceData object, convert the draws to
        # an xarray format.
        if as_xarray:
            return self.model._dict_to_xarray(draws)  # pylint: disable=protected-access

        # If we make it here, we are not returning as an xarray or InferenceData
        # object, so we need to convert the parameters to their original names
        # and return them as a dictionary
        return {k.model_varname: v for k, v in draws.items()}

    def get_inference_obj(
        self,
        n: custom_types.Integer = 1000,
        *,
        seed: Optional[custom_types.Integer] = None,
        batch_size: Optional[custom_types.Integer] = None,
    ) -> results.MLEInferenceRes:
        """Create ArviZ-compatible inference data object from MLE results.

        This method constructs a comprehensive inference data structure that
        integrates MLE results with the ArviZ ecosystem for Bayesian analysis.
        It organizes parameter samples, observed data, and posterior predictive
        samples into a standardized format.

        :param n: Number of samples to generate for the inference object. Defaults to 1000.
        :type n: custom_types.Integer
        :param seed: Random seed for reproducible sample generation. Defaults to None.
        :type seed: Optional[custom_types.Integer]
        :param batch_size: Batch size for memory-efficient sampling. Defaults to None.
        :type batch_size: Optional[custom_types.Integer]

        :returns: Structured inference data object with all MLE results
        :rtype: results.MLEInferenceRes

        The resulting inference object contains:
        - **Posterior samples**: Draws from fitted parameter distributions
        - **Observed data**: Original data used for parameter estimation
        - **Posterior predictive**: Samples from observable distributions

        Data Organization:
        - Latent parameters are stored in the main posterior group
        - Observable parameters become posterior predictive samples
        - Observed data is stored separately for comparison
        - All data maintains proper dimensional structure and labeling

        This enables:
        - Integration with ArviZ plotting and diagnostic functions
        - Model comparison
        - Posterior predictive checking workflows
        - Standardized reporting and visualization

        Example:
            >>> # Create inference object with default settings
            >>> inference_obj = mle_result.get_inference_obj()
            >>> # Generate larger sample with custom batch size
            >>> inference_obj = mle_result.get_inference_obj(
            ...     n=5000, batch_size=500, seed=42
            ... )
        """
        # Get the samples from the posterior
        draws = self.draw(n, seed=seed, as_xarray=True, batch_size=batch_size)

        # Otherwise, we also are going to want to attach the observed data
        # to the InferenceData object. First, rename the "n" dimension to "sample"
        # and add a dummy "chain" dimension
        draws = draws.rename_dims({"n": "draw"})
        draws = draws.expand_dims("chain", 0)

        # Now separate out the observables from the latent variables. Build
        # the initial inference data object with the latent variables
        inference_data = az.convert_to_inference_data(
            draws[
                [
                    varname
                    for varname, mle_param in self.model_varname_to_mle.items()
                    if not self.model.all_model_components_dict[varname].observable
                ]
            ]
        )

        # Add the observables and the observed data to the inference data object
        # pylint: disable=protected-access
        inference_data.add_groups(
            observed_data=xr.Dataset(
                data_vars={
                    k: self.model._compress_for_xarray(v)[0]
                    for k, v in self.data.items()
                }
            ),
            posterior_predictive=draws[
                [
                    varname
                    for varname, mle_param in self.model_varname_to_mle.items()
                    if self.model.all_model_components_dict[varname].observable
                ]
            ],
        )
        return results.MLEInferenceRes(inference_data)
