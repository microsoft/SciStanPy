# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Maximum likelihood estimation results analysis and visualization for SciStanPy
models.

This module provides analysis tools for maximum likelihood estimation
results from SciStanPy models. It offers diagnostic plots, calibration
checks, and posterior predictive analysis tools designed specifically for MLE-based
inference workflows.

The module centers around three main classes: MLEParam for individual parameter
estimates, MLE for complete model results, and MLEInferenceRes class, which wraps
ArviZ InferenceData objects with specialized methods for MLE result analysis. Together,
these classes provide the estimated parameter values and the fitted probability
distributions resulting from MLE analysis, and allow for downstream analysis including
uncertainty quantification and posterior predictive sampling. It provides both individual
diagnostic tools and analysis workflows that combine multiple checks
into unified reporting interfaces.

Key Features:
    - Individual parameter MLE estimates with associated distributions
    - Complete model MLE results with loss tracking and diagnostics
    - Posterior predictive checking workflows
    - Model calibration analysis with quantitative metrics
    - Interactive visualization with customizable display options
    - Integration with ArviZ for standardized Bayesian workflows
    - Memory-efficient handling of large posterior predictive samples
    - Flexible output formats for different analysis needs

Visualization Capabilities:
    - Posterior predictive sample plotting with confidence intervals
    - Calibration plots with deviation metrics
    - Quantile-quantile plots for model validation
    - Interactive layouts with customizable dimensions


Performance Considerations:
    - Batch sampling prevents memory overflow for large sample requests
    - GPU acceleration is preserved through PyTorch distribution objects

The module is designed to work with SciStanPy's MLE estimation workflow,
providing immediate access to model diagnostics and validation tools
once MLE fitting is complete. The MLE results can be used for various purposes
including model comparison, uncertainty quantification, and as initialization for
more sophisticated inference procedures like MCMC sampling.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Generator, Literal, Optional, Union, overload

import arviz as az
import hvplot.pandas  # pylint: disable=unused-import
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm

import scistanpy

from .base_classes import InferenceRes, SciStanPyToNetCDFConverter

if TYPE_CHECKING:
    from scistanpy import custom_types


class MLEToNetCDFConverter(SciStanPyToNetCDFConverter):
    """
    Used for building inference objects from MLE results where the posterior predictive
    samples are too large to fit into memory at once. In this case, we stream the
    posterior predictive samples into the NetCDF file in batches.
    """

    def __init__(
        self,
        results: "MLE",
        model: "scistanpy.Model",
        data: dict[str, npt.NDArray],
        n: int,
        seed: Optional["custom_types.Integer"] = None,
        batch_size: Optional["custom_types.Integer"] = None,
    ):

        # Initialize the base class
        super().__init__(results=results, model=model, data=data)

        # Record number of draws and chains (always 1 chain)
        self.num_chains = 1
        self.num_draws = n

        # Record seed and batch size
        self.seed = seed
        self.batch_size = batch_size or 1

    def _stream_draws(
        self,
    ) -> Generator[tuple[int, int, dict[str, npt.ArrayLike]], None, None]:

        # Set the random seed if provided
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Loop until we have the requested number of draws
        total_draws = 0
        with tqdm(total=self.num_draws, desc="Bootstrapping PPC samples") as pbar:
            while total_draws < self.num_draws:

                # Get a set of draws. Note that the model.draw method returns a
                # dictionary of numpy arrays. With `batch_size = None`, it returns
                # all requested draws at once. We query this repeatedly to draw
                # in batches.
                batch_size = min(self.batch_size, self.num_draws - total_draws)
                draws = self.results.draw(
                    n=batch_size,
                    seed=None,  # We have already set the seed globally
                    as_xarray=False,
                    batch_size=None,
                )

                # Process all draws. Chain ind is always 0 for MLE results. Add
                # 'ppc' to observables
                new_total = total_draws + batch_size
                for batch_ind, draw_ind in enumerate(range(total_draws, new_total)):
                    yield 0, draw_ind, {
                        k + "_ppc" if self.model[k].observable else k: v[batch_ind]
                        for k, v in draws.items()
                    }

                # Update the total draws and progress bar
                total_draws = new_total
                pbar.update(batch_size)


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
        .. code-block:: python

            # Run MLE fitting
            mle_result = model.mle(data=observed_data)

            # Access a specific parameter (an instance of `MLEParam`) describing
            # the MLE results for that parameter
            mle_param = mle_result.mu
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


class MLEInferenceRes(InferenceRes):
    """Object that holds results from maximum likelihood estimation.

    This class extends the base InferenceRes to handle MLE-specific
    functionality, particularly the construction of ArviZ InferenceData
    objects from MLE results. It supports both in-memory and Dask-based
    processing for large datasets.

    :param model: Original SciStanPy model
    :type model: Union[scistanpy.Model, None]
    :param results: MLE results object
    :type results: Union[MLE, None]
    :param data: Observed data used for parameter estimation
    :type data: dict[str, npt.NDArray]
    :param precision: Numerical precision for stored samples when using Dask.
        Options are "double", "single", or "half". Defaults to "single".
    :type precision: Literal["double", "single", "half"]
    :param inference_obj: Pre-existing ArviZ InferenceData object or filename.
        If None, it will be built from MLE results. Defaults to None.
    :type inference_obj: Optional[az.InferenceData | str]
    :param mib_per_chunk: Memory chunk size in MiB when using Dask. If None,
        defaults to automatic chunk sizing. Defaults to None.
    :type mib_per_chunk: Optional[custom_types.Integer]
    :param use_dask: Whether to use Dask for parallel processing. Defaults to False.
    :type use_dask: bool
    :param output_filename: If provided, saves the inference data to this NetCDF
        file. If None and `use_dask` is True, a temporary file is used. Defaults to `None`.
    :type output_filename: Optional[str]
    :param n: Number of samples to generate for the inference object.
    :type n: custom_types.Integer
    :param seed: Random seed for reproducible sample generation. Defaults to None.
    :type seed: Optional[custom_types.Integer]
    :param batch_size: Batch size for memory-efficient sampling. Defaults to None.
    :type batch_size: Optional[custom_types.Integer]
    """

    RESULTS_TO_NETCDF_CONVERTER = MLEToNetCDFConverter

    def __init__(
        self,
        *,
        model: Union["scistanpy.Model", None] = None,
        results: Union["MLE", None] = None,
        data: Optional[dict[str, npt.NDArray]] = None,
        precision: Literal["double", "single", "half"] = "single",
        inference_obj: Optional[az.InferenceData | str] = None,
        mib_per_chunk: custom_types.Integer | None = None,
        use_dask: bool = False,
        output_filename: str | None = None,
        n: int = 1000,
        seed: Optional["custom_types.Integer"] = None,
        batch_size: Optional["custom_types.Integer"] = None,
    ):
        # Store values for 'n', 'seed', and 'batch_size'
        self.n = n
        self.seed = seed
        self.batch_size = batch_size

        # Call the parent constructor
        super().__init__(
            model=model,
            results=results,
            data=data,
            precision=precision,
            inference_obj=inference_obj,
            mib_per_chunk=mib_per_chunk,
            use_dask=use_dask,
            output_filename=output_filename,
        )

        # Add mle data as a group in the inference object
        self._append_mle(output_filename=output_filename)

    def _build_inference_obj(
        self,
        data: dict[str, npt.NDArray],
        precision: Literal["double", "single", "half"],
        mib_per_chunk: custom_types.Integer | None,
        output_filename: str | None,
        **converter_kwargs: Any,
    ) -> str | az.InferenceData:
        """Build the ArviZ InferenceData object from MLE results.

        This method constructs the InferenceData object by drawing samples
        from the fitted parameter distributions and organizing them into
        the appropriate groups.

        :param data: Observed data used for parameter estimation
        :type data: dict[str, npt.NDArray]
        :param precision: Numerical precision for stored samples when using Dask.
            Options are "double", "single", or "half". Defaults to "single".
        :type precision: Literal["double", "single", "half"]
        :param mib_per_chunk: Memory chunk size in MiB when using Dask. If None,
            defaults to automatic chunk sizing. Defaults to None.
        :type mib_per_chunk: Optional[custom_types.Integer]
        :param output_filename: If provided, saves the inference data to this NetCDF
            file. If None and `use_dask` is True, a temporary file is used. Defaults to `None`.
        :type output_filename: Optional[str]
        :param converter_kwargs: Additional keyword arguments for the converter.
        :type converter_kwargs: Any

        :returns: Structured inference data object with all MLE results if not
            using Dask; otherwise, returns the filename of the saved NetCDF file.
        :rtype: az.InferenceData | str
        """
        # We use the parent method if running with dask
        if self.use_dask:
            return super()._build_inference_obj(
                data=data,
                precision=precision,
                mib_per_chunk=mib_per_chunk,
                output_filename=output_filename,
                n=self.n,
                seed=self.seed,
                batch_size=self.batch_size,
                **converter_kwargs,
            )

        # Otherwise, we need to draw samples and directly build the inference
        # data object
        draws = self.results.draw(
            n=self.n, seed=self.seed, as_xarray=True, batch_size=self.batch_size
        )

        # Rename the "n" dimension to "sample" and add a dummy "chain" dimension
        draws = draws.rename_dims({"n": "draw"})
        draws = draws.expand_dims("chain", 0)

        # Now separate out the observables from the latent variables. Build
        # the initial inference data object with the latent variables
        inference_data = az.convert_to_inference_data(
            draws[
                [
                    varname
                    for varname, mle_param in self.results.model_varname_to_mle.items()
                    if not self.model.all_model_components_dict[varname].observable
                ]
            ]
        )

        # Add the observables and the observed data to the inference data object
        # pylint: disable=protected-access
        inference_data.add_groups(
            observed_data=xr.Dataset(
                data_vars={
                    k: self.model._compress_for_xarray(v)[0] for k, v in data.items()
                }
            ),
            posterior_predictive=draws[
                [
                    varname
                    for varname, mle_param in self.results.model_varname_to_mle.items()
                    if self.model.all_model_components_dict[varname].observable
                ]
            ],
        )

        return inference_data

    def _append_mle(self, output_filename: Optional[str] = None) -> None:
        """Adds MLE point estimates as a group in the inference object."""
        # Null op if we don't have results to append. This happens when we are loading
        # from disk
        if self.results is None:
            return

        # Identify the MLE results and convert them to an xarray dataset. We prepend
        # a dummy "draws" axis to be able to reused the model's existing converter
        # method for converting to xarray format.
        extracted_mle = {
            self.model.all_model_components_dict[varname]: (mle_param.mle[np.newaxis])
            for varname, mle_param in self.results.model_varname_to_mle.items()
            if mle_param.mle is not None
        }
        assert (
            len(extracted_mle) > 0
        ), "No MLE estimates found to append to inference object."

        # Convert the MLE estimates to an xarray dataset and add it as a group in
        # the inference object.
        mle_dataset = self.model._dict_to_xarray(  # pylint: disable=protected-access
            extracted_mle
        )
        mle_dataset = mle_dataset.squeeze("n", drop=True)  # Remove dummy dim
        self.inference_obj.add_groups(mle=mle_dataset)

        # When using dask, the NetCDF file has already been written
        # by the converter. Append the MLE group so it is persisted
        # on disk as well.
        if self.use_dask:
            assert output_filename is not None
            mle_dataset.to_netcdf(
                output_filename, mode="a", group="mle", engine="h5netcdf"
            )


class MLE:
    """Complete maximum likelihood estimation results for a SciStanPy model.

    This class encapsulates the full results of a call to
    :py:meth:`Model.mle() <scistanpy.model.model.Model.mle>` for MLE parameter
    estimation, including parameter estimates, fitted distributions, optimization
    diagnostics, and utilities for further analysis. It provides a
    comprehensive interface for working with MLE results.

    :param model: Original SciStanPy model
    :type model: scistanpy.Model
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

    The class automatically creates attributes for each parameter, allowing, e.g.,
    direct access to a parameter named ``mu`` using the syntax ``mle_result.mu``.
    It also exposes a
    :py:meth:`method for bootstrapping <scistanpy.model.results.mle.MLE.get_inference_obj>`
    samples from the fit model, providing a relatively cheap way to quantify uncertainty
    around MLE estimates.

    Key Features:

    - Direct attribute access to individual parameter results
    - Comprehensive loss trajectory tracking and visualization
    - Efficient sampling from fitted parameter distributions
    - Integration with ArviZ for Bayesian workflow compatibility
    - Memory-efficient batch processing for large sample requests

    Example:
        .. code-block:: python

            # Run MLE fitting
            mle_result = model.mle(data=observed_data)

            # Access optimization diagnostics
            loss_plot = mle_result.plot_loss_curve(logy=True)

            # Sample from all fitted distributions
            parameter_samples = mle_result.draw(n=1000, as_xarray=True)

            # Sample from a specific parameter
            mu_samples = mle_result.mu.draw(1000)

            # Create inference object for detailed analysis
            inference_obj = mle_result.get_inference_obj(n=2000)
    """

    def __init__(
        self,
        model: "scistanpy.Model",
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
        batch_size: Optional[custom_types.Integer],
    ) -> xr.Dataset: ...

    @overload
    def draw(
        self,
        n: custom_types.Integer,
        *,
        seed: Optional[custom_types.Integer],
        as_xarray: Literal[False],
        batch_size: Optional[custom_types.Integer],
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
        use_dask: bool = False,
        netcdf_filename: str | None = None,
        precision: Literal["double", "single", "half"] = "single",
        mib_per_chunk: custom_types.Integer | None = None,
    ) -> MLEInferenceRes:
        """Create ArviZ-compatible inference data object from MLE results.

        This method constructs a comprehensive inference data structure that
        integrates MLE results with the ArviZ ecosystem for Bayesian analysis. Samples
        are bootstrapped from the fitted parameter distributions to approximate
        posterior distributions. It organizes parameter samples, observed data,
        and posterior predictive samples into a standardized format.

        :param n: Number of samples to generate for the inference object. Defaults to 1000.
        :type n: custom_types.Integer
        :param seed: Random seed for reproducible sample generation. Defaults to None.
        :type seed: Optional[custom_types.Integer]
        :param batch_size: Batch size for memory-efficient sampling. Defaults to None.
        :type batch_size: Optional[custom_types.Integer]
        :param use_dask: Whether to use Dask for parallel processing. Defaults to False.
        :type use_dask: bool
        :param netcdf_filename: If provided, saves the inference data to this NetCDF
            file. If None and `use_dask` is True, a temporary file is used. Defaults
            to `None`.
        :type netcdf_filename: Optional[str]
        :param precision: Numerical precision for stored samples when using Dask.
            Options are "double", "single", or "half". Defaults to "single".
        :type precision: Literal["double", "single", "half"]
        :param mib_per_chunk: Memory chunk size in MiB when using Dask. If None,
            defaults to automatic chunk sizing. Defaults to None.
        :type mib_per_chunk: Optional[custom_types.Integer]

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

        .. important::
            Samples are drawn using the optimized value of their parent parameters.
            For example, if a parameter ``y`` is defined in the model as
            ``y ~ Normal(mu, sigma)``, where ``mu`` and ``sigma`` are also parameters
            in the model, then samples of ``y`` will be drawn using the MLE values
            of ``mu`` and ``sigma``. This means that uncertainty in ``mu`` and
            ``sigma`` is not propagated to ``y``. This is a limitation of the
            MLE-based approach and should be considered when interpreting results.

        .. important::
            Related to the above, for root-level parameters with constant values
            for parent parameters, sampling from the fit distribution is identical
            to sampling from the prior distribution. For example, for a parameter,
            ``y`` defined in the model as ``y ~ Normal(mu = 0.0, sigma = 1.0)``,
            the values of ``mu`` and ``sigma`` will not change during fitting, so
            the distribution of ``y`` will remain ``Normal(0.0, 1.0)``.

        Example:
            >>> # Create inference object with default settings
            >>> inference_obj = mle_result.get_inference_obj()
            >>> # Generate larger sample with custom batch size
            >>> inference_obj = mle_result.get_inference_obj(
            ...     n=5000, batch_size=500, seed=42
            ... )
        """
        return MLEInferenceRes(
            model=self.model,
            results=self,
            data=self.data,
            precision=precision,
            inference_obj=None,
            mib_per_chunk=mib_per_chunk,
            use_dask=use_dask,
            output_filename=netcdf_filename,
            n=n,
            seed=seed,
            batch_size=batch_size,
        )
