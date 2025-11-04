# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Base classes for converting SciStanPy results to NetCDF format and analysis.

This module provides the foundational classes for converting SciStanPy modeling results
(from HMC or MLE methods) into structured NetCDF format and conducting posterior predictive
checking analysis. The module includes:

- :class:`SciStanPyToNetCDFConverter`: Abstract base class for NetCDF conversion
- :class:`InferenceRes`: Abstract base class for result analysis and visualization
- Helper functions for statistical computations and transformations

The NetCDF conversion process enables efficient storage and retrieval of large datasets
with proper chunking strategies and data type optimization. The analysis classes
provide comprehensive posterior predictive checking workflows including calibration
assessment, model validation, and interactive visualization capabilities.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Generator, Literal, Optional, Sequence, Union, overload

import arviz as az
import dask
import h5netcdf
import holoviews as hv
import numpy as np
import numpy.typing as npt
import panel as pn
import xarray as xr
from scipy import stats

import scistanpy
from scistanpy import custom_types, plotting, utils
from scistanpy.model.components.transformations import transformed_parameters


class SciStanPyToNetCDFConverter(ABC):
    """Abstract base class for converting SciStanPy inference results to NetCDF format.

    This class provides the foundational structure for converting various types of
    SciStanPy inference results (HMC, MLE, etc.) into standardized NetCDF files
    with proper chunking, data type optimization, and group organization.

    This class serves as the base for method-specific converters:

    - :class:`~scistanpy.model.results.hmc.CmdStanMCMCToNetCDFConverter`
    - :class:`~scistanpy.model.results.mle.MLEToNetCDFConverter`

    :param results: Raw inference results object from SciStanPy methods
    :type results: Any
    :param model: SciStanPy model object containing parameter definitions and metadata
    :type model: scistanpy.Model
    :param data: Observed data dictionary used for model fitting, by default None
    :type data: dict[str, Any] | None

    Attributes
    ----------
    :ivar results: Stored reference to the original inference results
    :ivar model: Reference to the SciStanPy model object
    :ivar data: Observed data used during model fitting
    :ivar num_chains: Number of chains (must be set by subclasses)
    :ivar num_draws: Number of draws per chain (must be set by subclasses)
    :ivar precision: Numerical precision for stored arrays
    :ivar var_dtypes: Mapping from variable names to appropriate NumPy data types
    :ivar varname_to_dset: Mapping from variable names to NetCDF dataset objects

    Notes
    -----

    Subclasses must implement:

    - :meth:`_stream_draws`: Method to yield individual draws from results
    - Set ``num_chains`` and ``num_draws`` attributes during initialization
    """

    # Maps between the precision of the data and the numpy types
    _NP_TYPE_MAP = {
        "double": {"float": np.float64, "int": np.int64},
        "single": {"float": np.float32, "int": np.int32},
        "half": {"float": np.float16, "int": np.int16},
    }

    def __init__(
        self,
        results: Any,
        model: "scistanpy.Model",
        data: dict[str, Any] | None = None,
    ):
        """Initialize the NetCDF converter with results and model information.

        Collects metadata about variables including names, shapes, types, and
        dimensions. This information is used to structure the NetCDF file layout.

        Notes
        -----
        Subclasses must set ``num_chains`` and ``num_draws`` attributes after
        calling this initialization method.
        """

        # The results and model are stored as attributes
        self.results = results
        self.model = model
        self.data = data

        # Users must set the number of chains and draws
        self.num_chains: int = 0
        self.num_draws: int = 0

        # Placeholders for precision and data types
        self.precision: Literal["double", "single", "half"]
        self.var_dtypes: dict[str, Union[type[np.floating], type[np.integer]]]

        # A mapping from variable names to datasets in the NetCDF file
        self.varname_to_dset: dict[str, h5netcdf.Variable] = {}

    def _get_var_dtypes_dimnames(
        self, precision: Literal["double", "single", "half"]
    ) -> tuple[
        dict[str, Union[type[np.floating], type[np.integer]]],
        dict[str, tuple[tuple[str, int], ...]],
    ]:
        """Determine appropriate data types and dimension names for model variables.

        Analyzes the SciStanPy model structure to assign appropriate NumPy data
        types based on parameter characteristics and specified precision, while
        also determining proper dimension naming schemes for multi-dimensional
        parameters.

        :param precision: Numerical precision specification for array storage
        :type precision: Literal["double", "single", "half"]

        :returns: Tuple of (data_types_dict, dimension_names_dict). The first
            element maps variable names to NumPy data types, while the second maps
            variable names to dimension specifications as tuples of (dimension_name,
            dimension_size) pairs.
        :rtype: tuple[dict[str, Union[type[np.floating], type[np.integer]]], dict[str, tuple[tuple[str, int], ...]]]

        Notes
        -----
        The method processes only parameters and transformed parameters from the
        model. Observable parameters are suffixed with '_ppc' to distinguish
        posterior predictive samples from regular parameters.

        Data type assignment follows these rules:

        - Discrete distributions → integer types
        - Continuous distributions → floating-point types
        - Precision determines specific type (float64/int64 for "double", etc.)

        Dimension names are extracted from the model's dimension mapping, with
        singleton dimensions automatically excluded from the final specification.
        """
        # pylint: disable=protected-access

        def get_dimname() -> tuple[tuple[str, int], ...] | tuple[()]:
            """Retrieves the dimension names for the current component."""
            # Get the name of the dimensions
            named_shape = []
            for dimind, dimsize in enumerate(component.shape[::-1]):

                # See if we can get the name of the dimension. If we cannot, this must
                # be a singleton dimension
                if (dimname := dim_map.get((dimind, dimsize))) is None:
                    assert dimsize == 1
                    continue

                # If we have a name, record
                named_shape.append((dimname, dimsize))

            # If we have no dimensions, we return an empty tuple
            if len(named_shape) == 0:
                return ()

            # We have our named shape
            return tuple(named_shape[::-1])

        # We will need the map from dimension depth and size to dimension name
        dim_map = self.model.get_dimname_map()

        # Datatypes for the variables
        var_dtypes = {}
        var_dimnames = {}
        for varname, component in self.model.named_model_components_dict.items():

            # We only take parameters and transformed parameters
            if not isinstance(
                component,
                (
                    scistanpy.parameters.Parameter,
                    transformed_parameters.TransformedParameter,
                ),
            ):
                continue

            # Update the varname if needed
            if (
                isinstance(component, scistanpy.parameters.Parameter)
                and component.observable
            ):
                varname = f"{varname}_ppc"

            # Record the datatype.
            var_dtypes[varname] = self.__class__._NP_TYPE_MAP[precision][
                (
                    "int"
                    if isinstance(component, scistanpy.parameters.DiscreteDistribution)
                    else "float"
                )
            ]

            # Record the dimension names
            var_dimnames[varname] = get_dimname()

        return var_dtypes, var_dimnames

    def _write_attributes(  # pylint: disable=unused-argument
        self, netcdf_file: h5netcdf.File
    ) -> None:
        """Write global attributes to the NetCDF file.

        Base implementation does nothing. Subclasses can override this method
        to add method-specific metadata attributes to the NetCDF file root.

        :param netcdf_file: Opened NetCDF file object
        :type netcdf_file: h5netcdf.File

        This method writes the provided attributes to the root of the NetCDF file.
        """
        # Does nothing in the base class
        return

    def _create_netcdf_groups(
        self, netcdf_file: h5netcdf.File
    ) -> dict[str, h5netcdf.Group]:
        """Create the standard group structure in the NetCDF file.

        Establishes the basic ArviZ-compatible group organization with groups
        for different types of inference data. Subclasses can override this
        method to add method-specific groups.


        :param netcdf_file: Opened NetCDF file object for group creation
        :type netcdf_file: h5netcdf.File

        :returns: Dictionary mapping group identifiers to NetCDF group objects.
            The keys are 'posterior_group', 'ppc_group', and 'observed_group',
            which point to parameter samples, posterior predictive samples, and
            observed data groups, respectively.
        :rtype: dict[str, h5netcdf.Group]
        """
        return {
            "posterior_group": netcdf_file.create_group("posterior"),
            "ppc_group": netcdf_file.create_group("posterior_predictive"),
            "observed_group": netcdf_file.create_group("observed_data"),
        }

    def write_netcdf(
        self,
        filename: str,
        precision: Literal["double", "single", "half"] = "single",
        mib_per_chunk: custom_types.Integer | None = None,
    ) -> str:
        """Write inference results to NetCDF file.

        Orchestrates the complete conversion process from raw inference results
        to a structured, chunked NetCDF file compatible with ArviZ and other
        analysis tools.

        :param filename: Output filename for the NetCDF file
        :type filename: str
        :param precision: Numerical precision for stored arrays
        :type precision: Literal["double", "single", "half"]
        :param mib_per_chunk: Memory limit per chunk in MiB for chunking strategy
        :type mib_per_chunk: custom_types.Integer | None

        :returns: Path to the created NetCDF file
        :rtype: str

        :raises ValueError: If ``num_chains`` or ``num_draws`` have not been set
            by subclass

        Notes
        -----
        The conversion process follows these steps:

        1. **Validation**: Ensures required attributes are set by subclasses
        2. **Type analysis**: Determines data types and dimensions for all variables
        3. **File structure**: Creates NetCDF file with groups and dimensions
        4. **Variable creation**: Sets up chunked variables with optimal storage
        5. **Data population**: Streams and writes inference results to file

        The resulting NetCDF structure includes:

        - **posterior** group: Parameter samples organized by chain/draw
        - **posterior_predictive** group: Observable predictions
        - **observed_data** group: Original observed data

        Chunking Strategy:
            The first two dimensions (chain, draw) are kept intact in each chunk,
            while remaining dimensions are chunked based on memory constraints.
            This enables efficient operations over chains and draws.

        Examples
        --------
        .. code-block:: python

            converter = MyConverter(results, model, data)
            netcdf_path = converter.write_netcdf(
                filename='results.nc',
                precision='single',
                mib_per_chunk=128
            )
        """
        # Set the precision
        self.precision = precision

        # Make sure that the number of chains and draws have been set
        if self.num_chains <= 0 or self.num_draws <= 0:
            raise ValueError(
                "num_chains and num_draws must be set to positive integers before "
                "writing NetCDF file. Did you forget to set them in the subclass?"
            )

        # Get the data types for the variables
        self.var_dtypes, var_dimnames = self._get_var_dtypes_dimnames(precision)

        # Create the HDF5 file
        with h5netcdf.File(filename, "w") as netcdf_file:

            # Write attributes to the file
            self._write_attributes(netcdf_file)

            # Set dimensions
            netcdf_file.dimensions = {
                "chain": self.num_chains,
                "draw": self.num_draws,
                **{
                    dimname: dimsize
                    for varinfo in filter(lambda x: len(x) > 0, var_dimnames.values())
                    for dimname, dimsize in varinfo
                },
            }

            # Create groups
            groups = self._create_netcdf_groups(netcdf_file)

            # Now we can create a dataset for each variable. We update the
            # mapping from the variable name to the dataset object
            for varname, dtype in self.var_dtypes.items():

                # Get the shape of the variable
                if len(shape_info := var_dimnames[varname]) == 0:
                    named_shape, true_shape = (), ()
                else:
                    named_shape, true_shape = zip(*shape_info)

                # Calculate the chunk shape. We always hold the first two dimensions
                # frozen. This is because the first two dimensions are what we
                # are typically performing operations over.
                chunk_shape = utils.get_chunk_shape(
                    array_shape=(self.num_chains, self.num_draws, *true_shape),
                    array_precision=precision,
                    mib_per_chunk=mib_per_chunk,
                    frozen_dims=(0, 1),
                )

                # We record without the '_ppc' suffix
                recorded_varname = varname.removesuffix("_ppc")

                # Build the variable in the appropriate group
                group = (
                    groups["ppc_group"]
                    if varname.endswith("_ppc")
                    else groups["posterior_group"]
                )
                self.varname_to_dset[varname] = group.create_variable(
                    name=recorded_varname,
                    dimensions=("chain", "draw", *named_shape),
                    dtype=dtype,
                    chunks=chunk_shape,
                )

                # If an observable, also create a dataset in the observed group
                # and populate it with the data
                if varname.endswith("_ppc") and self.data is not None:
                    groups["observed_group"].create_variable(
                        name=recorded_varname,
                        data=self.data[recorded_varname].squeeze(),
                        dimensions=named_shape,
                        dtype=dtype,
                        chunks=chunk_shape[2:],
                    )

            # Now we populate the datasets with the data from the csv files
            for chain_ind, draw_ind, draw in self._stream_draws():
                for varname, varvals in draw.items():
                    self.varname_to_dset[varname][
                        chain_ind, draw_ind
                    ] = varvals.squeeze()

        return filename

    @abstractmethod
    def _stream_draws(
        self,
    ) -> Generator[tuple[int, int, dict[str, npt.ArrayLike]], None, None]:
        """Stream individual draws from inference results in memory-efficient manner.

        This abstract method must be implemented by subclasses to provide access
        to inference results one draw at a time, enabling processing of large
        datasets without loading everything into memory simultaneously.

        :returns: Generator yielding individual draws as tuples. Each tuple contains
            the chain index (0-based), draw index within chain (0-based), and a dictionary
            mapping variable names to their sampled values for that draw.
        :rtype: Generator[tuple[int, int, dict[str, npt.ArrayLike]], None, None]
        """


def _log10_shift(*args: npt.NDArray) -> tuple[npt.NDArray, ...]:
    """Apply log10 transformation with automatic shifting for non-positive values.

    Handles logarithmic transformation of arrays that may contain zero or negative
    values by automatically determining an appropriate shift to ensure all values
    are positive before log transformation.

    Parameters
    ----------
    *args : npt.NDArray
        Variable number of NumPy arrays to transform

    Returns
    -------
    tuple[npt.NDArray, ...]
        Tuple of log10-transformed arrays with consistent shifting applied

    Notes
    -----
    The transformation process:

    1. Finds the global minimum value across all input arrays
    2. Shifts all arrays by ``(1 - min_value)`` so minimum becomes 1
    3. Applies ``log10`` transformation to all shifted arrays

    This ensures that logarithmic scaling is always possible, which is
    particularly useful for visualization when data may contain zero or
    negative values (common in certain statistical and scientific contexts).

    All arrays are shifted by the same amount to maintain relative relationships
    between different datasets.

    Examples
    --------
    >>> import numpy as np
    >>> arr1 = np.array([-5, 0, 5])
    >>> arr2 = np.array([-2, 3, 8])
    >>> log_arr1, log_arr2 = _log10_shift(arr1, arr2)
    >>> # Global minimum was -5, so shift by 6
    >>> # arr1 becomes log10([1, 6, 11]) = [0, 0.778, 1.041]
    >>> # arr2 becomes log10([4, 9, 14]) = [0.602, 0.954, 1.146]
    """
    # Get the minimum value across all arrays
    min_val = min(np.min(arg) for arg in args)

    # Shift the arrays and apply log10
    return tuple(np.log10(arg - min_val + 1) for arg in args)


def dask_enabled_summary_stats(inference_obj: az.InferenceData) -> xr.Dataset:
    """Compute summary statistics using Dask for memory-efficient processing.

    Computes essential summary statistics for posterior samples using Dask's
    lazy evaluation and chunked processing, enabling analysis of large datasets
    that exceed available memory.

    Parameters
    ----------
    inference_obj : az.InferenceData
        ArviZ InferenceData object containing posterior samples with 'chain'
        and 'draw' dimensions

    Returns
    -------
    xr.Dataset
        Dataset with 'metric' dimension containing computed statistics:

        - 'mean': Mean across chains and draws
        - 'sd': Standard deviation across chains and draws
        - 'hdi_3%', 'hdi_97%': 94% highest density interval bounds

    Notes
    -----
    This function is used internally by :meth:`InferenceRes.calculate_summaries`
    when Dask processing is enabled. It provides memory-efficient computation
    through:

    - **Lazy evaluation**: Computations are queued and optimized before execution
    - **Chunked processing**: Large arrays processed in manageable chunks
    - **Parallel execution**: Multiple statistics computed simultaneously
    - **Memory optimization**: Intermediate results released automatically

    The computed statistics collapse the 'chain' and 'draw' dimensions, producing
    summary values for each parameter across all samples.

    Examples
    --------
    >>> # Direct usage (typically called internally)
    >>> stats = dask_enabled_summary_stats(inference_data)
    >>> mean_values = stats.sel(metric='mean')
    >>> hdi_lower = stats.sel(metric='hdi_3%')
    >>> hdi_upper = stats.sel(metric='hdi_97%')

    See Also
    --------
    dask_enabled_diagnostics : Compute MCMC diagnostics with Dask
    InferenceRes.calculate_summaries : High-level interface for summary statistics
    """
    # Queue up the delayed computations
    with utils.az_dask():
        delayed_summaries = [
            inference_obj.posterior.mean(dim=("chain", "draw")),
            inference_obj.posterior.std(dim=("chain", "draw")),
            az.hdi(
                inference_obj,
                hdi_prob=0.94,
                dask_gufunc_kwargs={"output_sizes": {"hdi": 2}},
            ),
        ]

        # Compute the results
        mean, std, hdi = dask.compute(*delayed_summaries)

    # Concatenate the results
    return xr.concat(
        [
            mean.assign_coords(metric=["mean"]),
            std.assign_coords(metric=["sd"]),
            hdi.assign_coords(hdi=["hdi_3%", "hdi_97%"]).rename(hdi="metric"),
        ],
        dim="metric",
    )


def dask_enabled_diagnostics(inference_obj: az.InferenceData) -> xr.Dataset:
    """Compute comprehensive MCMC diagnostics using Dask for memory efficiency.

    Computes a full suite of MCMC convergence and efficiency diagnostics using
    Dask's parallel processing capabilities, enabling analysis of large multi-chain
    datasets that exceed available memory.

    Parameters
    ----------
    inference_obj : az.InferenceData
        ArviZ InferenceData object containing posterior samples with 'chain'
        and 'draw' dimensions from multi-chain MCMC sampling

    Returns
    -------
    xr.Dataset
        Dataset with 'metric' dimension containing diagnostic statistics:

        - 'mcse_mean': Monte Carlo standard error for mean estimates
        - 'mcse_sd': Monte Carlo standard error for standard deviation estimates
        - 'ess_bulk': Bulk effective sample size
        - 'ess_tail': Tail effective sample size
        - 'r_hat': R-hat convergence diagnostic

    Notes
    -----
    This function is used internally by :meth:`InferenceRes.calculate_summaries`
    when Dask processing is enabled and multiple chains are available. It provides
    memory-efficient computation of diagnostics through:

    - **Parallel execution**: All diagnostics computed simultaneously
    - **Chunked processing**: Large arrays handled in memory-efficient chunks
    - **Automatic optimization**: Dask optimizes computation graph for efficiency
    - **Load balancing**: Work distributed across available CPU cores

    Diagnostic Interpretation:

    - **MCSE**: Quantifies uncertainty in Monte Carlo estimates
    - **ESS**: Measures effective independent samples (should be >> 100)
    - **R-hat**: Convergence indicator (should be < 1.01 for good convergence)

    Examples
    --------
    >>> # Direct usage (typically called internally)
    >>> diagnostics = dask_enabled_diagnostics(inference_data)
    >>> rhat_values = diagnostics.sel(metric='r_hat')
    >>> ess_bulk = diagnostics.sel(metric='ess_bulk')
    >>> # Check convergence
    >>> converged = (rhat_values < 1.01).all()

    See Also
    --------
    dask_enabled_summary_stats : Compute summary statistics with Dask
    InferenceRes.calculate_summaries : High-level interface for diagnostics
    """
    # Run computations
    with utils.az_dask():
        diagnostics = dask.compute(
            az.mcse(inference_obj.posterior, method="mean"),
            az.mcse(inference_obj.posterior, method="sd"),
            az.ess(inference_obj.posterior, method="bulk"),
            az.ess(inference_obj.posterior, method="tail"),
            az.rhat(inference_obj.posterior),
        )

    # Concatenate the results and return
    return xr.concat(
        [
            dset.assign_coords(metric=[metric])
            for metric, dset in zip(
                ["mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"], diagnostics
            )
        ],
        dim="metric",
    )


class InferenceRes(ABC):
    """Abstract base class for analyzing and visualizing SciStanPy inference results.

    Provides comprehensive tools for posterior predictive checking, model validation,
    calibration assessment, and interactive visualization of results from SciStanPy
    inference methods (HMC, MLE, etc.).

    This class wraps ArviZ InferenceData objects with specialized methods tailored
    for SciStanPy workflows, enabling seamless analysis and visualization of
    complex probabilistic models.

    :param model: SciStanPy model object used for inference, by default None
    :type model: scistanpy.Model | None
    :param results: Raw inference results object, by default None
    :type results: Any | None
    :param data: Observed data dictionary used for model fitting, by default None
    :type data: dict[str, npt.NDArray] | None
    :param precision: Numerical precision for NetCDF storage. Default is 'single'.
    :type precision: Literal['double', 'single', 'half']
    :param inference_obj: Either an existing ArviZ InferenceData object or path
        to NetCDF file, by default None (build from results)
    :type inference_obj: az.InferenceData | str | None
    :param mib_per_chunk: Memory limit per chunk in MiB for Dask processing, by
        default None
    :type mib_per_chunk: int | None
    :param use_dask: Whether to enable Dask for memory-efficient computation
    :type use_dask: bool
    :param output_filename: Filename for NetCDF output when building from results,
        by default None
    :type output_filename: str | None

    :ivar model: Reference to the SciStanPy model object
    :ivar results: Stored raw inference results
    :ivar use_dask: Whether Dask processing is enabled
    :ivar inference_obj: ArviZ InferenceData object containing all results and analysis


    :raise ValueError: If ``RESULTS_TO_NETCDF_CONVERTER`` is not set by subclass
    :raise ValueError: If inference_obj is neither string nor InferenceData object
    :raise ValueError: If required groups (posterior, posterior_predictive) are missing

    Notes
    -----
    The class expects the InferenceData object to contain specific groups:

    - **posterior**: Parameter samples from fitted distributions
    - **posterior_predictive**: Observable samples for model checking
    - **observed_data**: Original observed data used for fitting

    Key capabilities provided:

    - **Posterior predictive checking**: Multiple visualization modes for model validation
    - **Calibration assessment**: Quantitative evaluation of prediction quality
    - **Interactive dashboards**: Widget-based exploration of results
    - **Summary statistics**: Efficient computation with optional Dask acceleration
    - **Persistent storage**: NetCDF serialization for reproducible analysis

    Subclasses must implement:

    - Set ``RESULTS_TO_NETCDF_CONVERTER`` class attribute
    - Override ``_build_inference_obj`` if custom conversion logic needed

    Examples
    --------
    Typical usage through method-specific subclasses:

    .. code-block:: python

        import scistanpy as ssp
        import numpy as np

        # Fit model using MLE
        mle_result = model.mle(data=observed_data)

        # Create analysis object
        analysis = mle_result.get_inference_obj()

        # Run comprehensive posterior predictive checking
        dashboard = analysis.run_ppc()

        # Compute summary statistics
        stats = analysis.calculate_summaries()

        # Save for later analysis
        analysis.save_netcdf('mle_analysis.nc')

        # Load saved analysis
        loaded_analysis = MLEInferenceRes.from_disk('mle_analysis.nc')

    See Also
    --------
    scistanpy.model.results.hmc.SampleResults : HMC-specific implementation
    scistanpy.model.results.mle.MLEInferenceRes : MLE-specific implementation
    """

    # Child classes must set this to the appropriate converter class
    RESULTS_TO_NETCDF_CONVERTER: type[SciStanPyToNetCDFConverter] | None = None
    """Converter class for writing results to NetCDF format."""

    def __init__(
        self,
        *,
        model: Union["scistanpy.Model", None] = None,
        results: Any = None,  # Overwritten in subclasses
        data: dict[str, npt.NDArray] | None = None,
        precision: Literal["double", "single", "half"] = "single",
        inference_obj: Optional[az.InferenceData | str] = None,
        mib_per_chunk: custom_types.Integer | None = None,
        use_dask: bool = False,
        output_filename: str | None = None,
    ):
        """Base class just initializes the ArviZ object."""
        # The NETCDF converter must be set in child classes
        if self.__class__.RESULTS_TO_NETCDF_CONVERTER is None:
            raise ValueError(
                "RESULTS_TO_NETCDF_CONVERTER must be set in child classes."
            )

        # Store the model, the result, and whether to use dask
        self.model = model
        self.results = results
        self.use_dask = use_dask

        # If the inference object is not provided, we build it from the result
        if inference_obj is None:
            inference_obj = self._build_inference_obj(
                data=data,
                precision=precision,
                mib_per_chunk=mib_per_chunk,
                output_filename=output_filename,
            )

        # If the ArviZ object is a string, we assume it is a path to a netcdf file
        # and load it from there
        if isinstance(inference_obj, str):

            # Load the inference object. Ignore warnings about chunking.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="The specified chunks separate the stored chunks along dimension",
                )
                self.inference_obj = az.from_netcdf(
                    filename=inference_obj,
                    engine="h5netcdf",
                    group_kwargs={
                        k: {"chunks": "auto" if use_dask else None}
                        for k in ("posterior", "posterior_predictive", "sample_stats")
                    },
                )

        # If the ArviZ object is an inference data object, we assume it is already
        # built and just assign it to the class
        elif isinstance(inference_obj, az.InferenceData):
            self.inference_obj = inference_obj

        # Otherwise, we raise an error
        else:
            raise ValueError(
                "inference_obj must be either a string or an InferenceData object"
            )

        # The arviz object must have a posterior and a posterior_predictive group
        if missing_groups := (
            {"posterior", "posterior_predictive"} - set(self.inference_obj.groups())
        ):
            raise ValueError(
                f"ArviZ object is missing the following groups: {', '.join(missing_groups)}"
            )

    def _build_inference_obj(
        self,
        data: dict[str, npt.NDArray] | None,
        precision: Literal["double", "single", "half"],
        mib_per_chunk: custom_types.Integer | None,
        output_filename: str,
        **converter_kwargs: Any,
    ) -> str | az.InferenceData:
        """Build ArviZ InferenceData object from SciStanPy inference results.

        Converts raw SciStanPy inference results into a structured ArviZ
        InferenceData object using the appropriate converter class. When Dask
        is enabled, writes results to NetCDF for chunked processing.

        Parameters
        ----------
        :param data: Observed data dictionary used for model fitting
        :type data: dict[str, npt.NDArray], optional
        :param precision: Numerical precision for stored arrays
        :type precision: Literal["double", "single", "half"]
        :param mib_per_chunk: Memory limit per chunk in MiB for chunking strategy
        :type mib_per_chunk: custom_types.Integer | None
        :param output_filename: NetCDF filename for output (required for Dask workflows)
        :type output_filename: str
        :param converter_kwargs: Additional arguments passed to converter constructor
        :type converter_kwargs: Any

        :returns: Either path to NetCDF file (for Dask) or constructed InferenceData object
        :rtype: str | az.InferenceData

        :raises ValueError: If ``output_filename`` is None when using Dask

        Notes
        -----
        The conversion process uses the converter class specified by the
        subclass's ``RESULTS_TO_NETCDF_CONVERTER`` attribute to handle
        method-specific result formats and requirements.
        """
        # There must be a netCDF filename specified
        if output_filename is None:
            raise ValueError(
                "When using dask, `output_filename` must be specified to store "
                "the chunked results."
            )

        # If no data, check for default data in the model. Otherwise, data provided
        # takes priority
        if data is None and self.model.has_default_data:
            data = self.model.default_data

        # Build the converter
        converter = (
            self.__class__.RESULTS_TO_NETCDF_CONVERTER(  # pylint: disable=not-callable
                results=self.results,
                model=self.model,
                data=data,
                **converter_kwargs,
            )
        )

        # Convert
        return converter.write_netcdf(
            filename=output_filename,
            precision=precision,
            mib_per_chunk=mib_per_chunk,
        )

    def save_netcdf(self, filename: str) -> None:
        """Save the ArviZ InferenceData object to NetCDF format.

        :param filename: Path where to save the NetCDF file
        :type filename: str

        This method provides persistent storage of analysis results.

        Example:
            >>> mle_analysis.save_netcdf('my_mle_results.nc')
            >>> # Later: reload with MLEInferenceRes('my_mle_results.nc')
        """
        self.inference_obj.to_netcdf(filename)

    def _update_group(
        self, attrname: str, new_group: xr.Dataset, force_del: bool = False
    ) -> None:
        """Update or add a group to the ArviZ InferenceData object.

        :param attrname: Name of the group to update or create
        :type attrname: str
        :param new_group: New dataset to add or use for updating
        :type new_group: xr.Dataset
        :param force_del: Whether to force deletion before adding. Defaults to False.
        :type force_del: bool

        This internal method manages the ArviZ object structure, enabling
        addition of computed statistics and derived quantities while
        maintaining data integrity.
        """
        # If the group already exists and we are not forcing a delete, we just update
        # the group.
        if hasattr(self.inference_obj, attrname) and not force_del:
            getattr(self.inference_obj, attrname).update(new_group)
            return

        # Otherwise, if we are forcing a delete, we delete the group before adding
        # the new one
        if force_del:
            delattr(self.inference_obj, attrname)
        self.inference_obj.add_groups({attrname: new_group})

    def calculate_summaries(
        self,
        var_names: list[str] | None = None,
        filter_vars: Literal[None, "like", "regex"] = None,
        kind: Literal["all", "stats", "diagnostics"] = "stats",
        round_to: "custom_types.Integer" = 2,
        circ_var_names: list[str] | None = None,
        stat_focus: str = "mean",
        stat_funcs: Optional[Union[dict[str, callable], callable]] = None,
        extend: bool = True,
        hdi_prob: "custom_types.Float" = 0.94,
        skipna: bool = False,
        diagnostic_varnames: Sequence[str] = (
            "mcse_mean",
            "mcse_sd",
            "ess_bulk",
            "ess_tail",
            "r_hat",
        ),
    ) -> xr.Dataset:
        """Compute comprehensive summary statistics and diagnostics for inference results.

        Computes summary statistics and/or MCMC diagnostics with automatic caching
        to the InferenceData object for persistence and reuse. Supports both standard
        ArviZ computations and memory-efficient Dask processing for large datasets.

        :param var_names: Variable names to include in summary, by default None (all variables)
        :type var_names: list[str], optional
        :param filter_vars: Variable filtering method, by default None
        :type filter_vars: {None, "like", "regex"}, optional
        :param kind: Type of statistics to compute
        :type kind: {"all", "stats", "diagnostics"}, default "stats"
        :param round_to: Number of decimal places for rounding results
        :type round_to: int, default 2
        :param circ_var_names: Names of circular variables for specialized handling, by default None
        :type circ_var_names: list[str], optional
        :param stat_focus: Primary statistic for focus in ArviZ computations
        :type stat_focus: str, default "mean"
        :param stat_funcs: Custom statistic functions, by default None
        :type stat_funcs: dict[str, callable] or callable, optional
        :param extend: Whether to extend default functions with custom ones (when stat_funcs provided)
        :type extend: bool, default True
        :param hdi_prob: Probability for highest density interval computation
        :type hdi_prob: float, default 0.94
        :param skipna: Whether to skip NaN values in computations
        :type skipna: bool, default False
        :param diagnostic_varnames: Names of diagnostic metrics for classification
        :type diagnostic_varnames: Sequence[str], default ("mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat")

        :returns: Dataset containing requested statistics with 'metric' dimension
        :rtype: xr.Dataset

        :raises ValueError: If diagnostics requested without 'chain' dimension in posterior
        :raises ValueError: If diagnostics requested with single chain (< 2 chains)
        :raises ValueError: If invalid ``kind`` parameter provided

        Notes
        -----
        **Automatic Caching**: Computed results are automatically added to the
        InferenceData object:

        - Summary statistics → ``variable_summary_stats`` group
        - Diagnostics → ``variable_diagnostic_stats`` group

        **Processing Modes**:

        - **Standard mode** (use_dask=False): Uses ArviZ's ``az.summary`` function
        - **Dask mode** (use_dask=True): Uses optimized Dask implementations
          (:func:`dask_enabled_summary_stats`, :func:`dask_enabled_diagnostics`)

        **Diagnostic Requirements**: MCMC diagnostics require:

        - Multiple chains (≥ 2) for meaningful convergence assessment
        - 'chain' and 'draw' dimensions in posterior samples

        The Dask implementations provide significant performance improvements
        for large datasets by leveraging parallel computation and memory-efficient
        chunked processing.

        Examples
        --------
        >>> # Compute basic summary statistics
        >>> stats = analysis.calculate_summaries()
        >>> mean_values = stats.sel(metric='mean')
        >>>
        >>> # Compute comprehensive diagnostics for multi-chain results
        >>> diagnostics = analysis.calculate_summaries(kind="diagnostics")
        >>> rhat_values = diagnostics.sel(metric='r_hat')
        >>> converged = (rhat_values < 1.01).all()
        >>>
        >>> # Compute everything with custom HDI probability
        >>> all_stats = analysis.calculate_summaries(
        ...     kind="all", hdi_prob=0.89
        ... )

        See Also
        --------
        az.summary : Underlying ArviZ summary function (standard mode)
        """
        # If there is no chain and draw dimension, we cannot run diagnostics
        if "chain" not in self.inference_obj.posterior.dims:
            raise ValueError(
                "Cannot run diagnostics on a dataset without chain and draw dimensions."
            )

        # If there is only one chain, we cannot run diagnostics
        if kind != "stats" and self.inference_obj.posterior.sizes["chain"] <= 1:
            raise ValueError(
                "Cannot run diagnostics on a dataset run using a single chain"
            )

        # Get the summaries and diagnostics, either with Dask or the default method
        if self.use_dask:

            # Special functions for Dask-enabled summary stats and diagnostics
            if kind == "stats":
                summary_stats = dask_enabled_summary_stats(self.inference_obj)
                diagnostics = None
                summaries = summary_stats
            elif kind == "diagnostics":
                summary_stats = None
                diagnostics = dask_enabled_diagnostics(self.inference_obj)
                summaries = diagnostics
            elif kind == "all":
                summary_stats = dask_enabled_summary_stats(self.inference_obj)
                diagnostics = dask_enabled_diagnostics(self.inference_obj)
                summaries = xr.concat([summary_stats, diagnostics], dim="metric")
            else:
                raise ValueError(
                    f"Invalid kind '{kind}'. Must be one of 'all', 'stats', or 'diagnostics'."
                )

        else:
            summaries = az.summary(
                data=self.inference_obj,
                var_names=var_names,
                filter_vars=filter_vars,
                fmt="xarray",
                kind=kind,
                round_to=round_to,
                circ_var_names=circ_var_names,
                stat_focus=stat_focus,
                stat_funcs=stat_funcs,
                extend=extend,
                hdi_prob=hdi_prob,
                skipna=skipna,
            )

            # Identify the diagnostic and summary statistics
            noted_diagnostics = set(diagnostic_varnames)
            calculated_metrics = set(summaries.metric.values.tolist())

            diagnostic_metrics = list(noted_diagnostics & calculated_metrics)
            stat_metrics = list(calculated_metrics - noted_diagnostics)

            summary_stats = summaries.sel(metric=stat_metrics)
            diagnostics = summaries.sel(metric=diagnostic_metrics)

        # Update the groups
        if kind == "all" or kind == "diagnostics":
            self._update_group("variable_diagnostic_stats", diagnostics)
        if kind == "all" or kind == "stats":
            self._update_group("variable_summary_stats", summary_stats)

        return summaries

    def _iter_pp_obs(
        self,
    ) -> Generator[tuple[str, npt.NDArray, npt.NDArray], None, None]:
        """Iterate over posterior predictive samples with corresponding observations.

        Provides standardized access to posterior predictive samples and observed
        data pairs, with automatic dimension reshaping and alignment for consistent
        processing across diagnostic and visualization methods.

        :returns: Generator yielding tuples of variable name, posterior predictive
            samples, and observed data.
        :rtype: Generator[tuple[str, npt.NDArray, npt.NDArray], None, None]


        Notes
        -----

        **Standardized Format**: All yielded arrays follow consistent conventions:

        - **reference_samples**: Shape (n_samples, n_features) where n_samples =
          n_chains x n_draws and n_features is the flattened parameter dimensions
        - **observed_data**: Shape (n_features,) matching the feature dimension
          of reference samples

        This standardization enables uniform processing in downstream methods
        without requiring case-by-case dimension handling.

        **Usage Pattern**: This method is used internally by visualization and
        diagnostic methods to ensure consistent data access patterns.

        Examples
        --------
        >>> # Internal usage pattern in analysis methods
        >>> for var_name, predictions, observations in analysis._iter_pp_obs():
        ...     # predictions.shape = (n_samples, n_features)
        ...     # observations.shape = (n_features,)
        ...     diagnostic_result = compute_diagnostic(predictions, observations)
        """
        # Loop over the posterior predictive samples
        for varname, reference in self.inference_obj.posterior_predictive.items():

            # Get the observed data and convert reference and observed to numpy
            # arrays.
            observed = self.inference_obj.observed_data[  # pylint: disable=no-member
                varname
            ].to_numpy()
            reference = np.moveaxis(
                reference.stack(
                    samples=["chain", "draw"], features=[], create_index=False
                ).to_numpy(),
                -1,
                0,
            )

            # Dims must align
            assert observed.shape == reference.shape[1:]

            yield varname, reference.reshape(reference.shape[0], -1), observed.reshape(
                -1
            )

    @overload
    def check_calibration(
        self,
        *,
        return_deviance: Literal[False],
        display: Literal[True],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
    ) -> hv.Layout: ...

    @overload
    def check_calibration(
        self,
        *,
        return_deviance: Literal[False],
        display: Literal[False],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
    ) -> dict[str, hv.Overlay]: ...

    @overload
    def check_calibration(
        self,
        *,
        return_deviance: Literal[True],
        display: Literal[False],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
    ) -> tuple[dict[str, hv.Overlay], dict[str, float]]: ...

    def check_calibration(
        self, *, return_deviance=False, display=True, width=600, height=600
    ):
        """Assess model calibration through posterior predictive quantile analysis.

        This method evaluates how well the model's posterior predictive distribution
        matches the observed data by analyzing the distribution of quantiles. Well-
        calibrated models should produce observed data that are uniformly distributed
        across the quantiles of the posterior predictive distribution.

        :param return_deviance: Whether to return quantitative deviance metrics.
            Defaults to False.
        :type return_deviance: bool
        :param display: Whether to return formatted layout for display. Defaults to True.
        :type display: bool
        :param width: Width of individual plots in pixels. Defaults to 600.
        :type width: custom_types.Integer
        :param height: Height of individual plots in pixels. Defaults to 600.
        :type height: custom_types.Integer

        :returns: Calibration plots and optionally deviance metrics
        :rtype: Union[hv.Layout, dict[str, hv.Overlay], tuple[dict[str, hv.Overlay],
            dict[str, float]]]

        :raises ValueError: If both display and return_deviance are True

        Internally, this method is just a wrapper around
        :py:func:`ssp.plotting.plot_calibration <scistanpy.plotting.plot_calibration>`.
        See that function for a detailed description of the calibration assessment
        method and returned plots.

        Example:
            >>> # Visual assessment
            >>> cal_layout = mle_analysis.check_calibration()
            >>> # Quantitative assessment
            >>> plots, deviances = mle_analysis.check_calibration(
            ...     return_deviance=True, display=False
            ... )
            >>> print(f"Mean deviance: {np.mean(list(deviances.values())):.3f}")
        """
        # We cannot have both `display` and `return_deviance` set to True
        if display and return_deviance:
            raise ValueError(
                "Cannot have both `display` and `return_deviance` set to True."
            )

        # Loop over the posterior predictive samples
        plots: dict[str, hv.Overlay] = {}
        deviances: dict[str, "custom_types.Float"] = {}
        for varname, reference, observed in self._iter_pp_obs():

            # Build calibration plots and record deviance
            plot, dev = plotting.plot_calibration(reference, observed[None])
            dev = dev.item()
            deviances[varname] = dev

            # Finalize the plot with a text annotation and updates to the axes
            plots[varname] = (
                plot
                * hv.Text(
                    0.95,
                    0.0,
                    f"Absolute Deviance: {dev:.2f}",
                    halign="right",
                    valign="bottom",
                )
            ).opts(
                title=f"ECDF of Quantiles: {varname}",
                xlabel="Quantiles",
                ylabel="Cumulative Probability",
                width=width,
                height=height,
            )

        # If requested, display the plots
        if display:
            return hv.Layout(plots.values()).cols(1)

        # If requested, return the plots and the deviance
        if return_deviance:
            return plots, deviances

        # Otherwise, just return the plots
        return plots

    @overload
    def plot_posterior_predictive_samples(
        self,
        *,
        quantiles: Sequence["custom_types.Float"],
        use_ranks: bool,
        logy: bool,
        display: Literal[True],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
    ) -> hv.Layout: ...

    @overload
    def plot_posterior_predictive_samples(
        self,
        *,
        quantiles: Sequence["custom_types.Float"],
        use_ranks: bool,
        logy: bool,
        display: Literal[False],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
    ) -> dict[str, hv.Overlay]: ...

    def plot_posterior_predictive_samples(
        self,
        *,
        quantiles=(0.025, 0.25, 0.5),
        use_ranks=True,
        logy=False,
        display=True,
        width=600,
        height=400,
    ):
        """Visualize observed data against posterior predictive uncertainty intervals.

        This method creates plots showing how observed data relates to the uncertainty
        quantified by posterior predictive samples. The posterior predictive samples
        are displayed as confidence intervals, with observed data overlaid as points.

        :param quantiles: Quantiles defining confidence intervals. Defaults to
            (0.025, 0.25, 0.5). Note: quantiles are automatically symmetrized and
            median is always included.
        :type quantiles: Sequence[custom_types.Float]
        :param use_ranks: Whether to use ranks instead of raw values for x-axis.
            Defaults to True.
        :type use_ranks: bool
        :param logy: Whether to use logarithmic y-axis scaling. Defaults to False.
        :type logy: bool
        :param display: Whether to return formatted layout for display. Defaults to True.
        :type display: bool
        :param width: Width of individual plots in pixels. Defaults to 600.
        :type width: custom_types.Integer
        :param height: Height of individual plots in pixels. Defaults to 400.
        :type height: custom_types.Integer

        :returns: Posterior predictive plots in requested format
        :rtype: Union[hv.Layout, dict[str, hv.Overlay]]

        Visualization Features:

        - Confidence intervals shown as nested colored regions
        - Observed data displayed as scatter points
        - Optional rank transformation for better visualization of skewed data
        - Logarithmic scaling with automatic shifting for non-positive values
        - Interactive hover labels showing data point identifiers

        The rank transformation is particularly useful when observed values have
        highly skewed distributions, as it emphasizes the ordering rather than
        the absolute magnitudes.

        Example:
            >>> # Standard posterior predictive plot
            >>> pp_layout = mle_analysis.plot_posterior_predictive_samples()
            >>> # Custom quantiles with logarithmic scaling
            >>> pp_plots = mle_analysis.plot_posterior_predictive_samples(
            ...     quantiles=(0.05, 0.5, 0.95), logy=True, display=False
            ... )
        """
        # Process each observed variable
        plots: dict[str, hv.Overlay] = {}
        for varname, reference, observed in self._iter_pp_obs():

            # Get the x-axis data
            x = stats.rankdata(observed, method="ordinal") if use_ranks else observed

            # If using a log-y axis, shift the y-data
            if logy:
                reference, observed = _log10_shift(reference, observed)

            # Get labels
            labels = np.array(
                [
                    ".".join(map(str, indices))
                    for indices in np.ndindex(
                        self.inference_obj.observed_data[  # pylint: disable=no-member
                            varname
                        ].shape
                    )
                ]
            )

            # Sort data for plotting the areas and lines
            sorted_inds = np.argsort(x)
            x, reference, observed, labels = (
                x[sorted_inds],
                reference[:, sorted_inds],
                observed[sorted_inds],
                labels[sorted_inds],
            )

            # Build the plot
            plots[varname] = plotting.quantile_plot(
                x=x,
                reference=reference,
                quantiles=quantiles,
                observed=observed,
                labels={varname: labels},
                include_median=False,
                overwrite_input=True,
                observed_type="scatter",
            ).opts(
                xlabel=f"Observed Value {'Rank' if use_ranks else ''}: {varname}",
                ylabel=f"Value{' log10' if logy else ''}: {varname}",
                title=f"Posterior Predictive Samples: {varname}",
                width=width,
                height=height,
            )

        # If requested, display the plots
        if display:
            return hv.Layout(plots.values()).cols(1).opts(shared_axes=False)

        return plots

    @overload
    def plot_observed_quantiles(
        self,
        *,
        use_ranks: bool,
        display: Literal[True],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
        windowsize: Optional["custom_types.Integer"],
    ) -> hv.Layout: ...

    @overload
    def plot_observed_quantiles(
        self,
        *,
        use_ranks: bool,
        display: Literal[False],
        width: "custom_types.Integer",
        height: "custom_types.Integer",
        windowsize: Optional["custom_types.Integer"],
    ) -> dict[str, hv.Overlay]: ...

    def plot_observed_quantiles(
        self, *, use_ranks=True, display=True, width=600, height=400, windowsize=None
    ):
        """Visualize systematic patterns in observed data quantiles.

        This method creates hexagonal density plots showing the relationship between
        observed data values (or their ranks) and their corresponding quantiles
        within the posterior predictive distribution. A rolling mean overlay
        highlights systematic trends.

        :param use_ranks: Whether to use ranks instead of raw values for x-axis. Defaults to True.
        :type use_ranks: bool
        :param display: Whether to return formatted layout for display. Defaults to True.
        :type display: bool
        :param width: Width of individual plots in pixels. Defaults to 600.
        :type width: custom_types.Integer
        :param height: Height of individual plots in pixels. Defaults to 400.
        :type height: custom_types.Integer
        :param windowsize: Size of rolling window for trend line. Defaults to None (automatic).
        :type windowsize: Optional[custom_types.Integer]

        :returns: Quantile plots in requested format
        :rtype: Union[hv.Layout, dict[str, hv.Overlay]]

        Visualization Components:

        - Hexagonal binning showing density of (value, quantile) pairs
        - Rolling mean trend line highlighting systematic patterns
        - Colormap indicating point density for pattern identification

        Pattern Interpretation:

        - Horizontal trend line around 0.5 with uniformly distributed points indicates
          good calibration
        - Systematic deviations suggest model bias or miscalibration

        The hexagonal binning is particularly effective for visualizing large
        datasets where individual points would create overplotting issues.

        Example:
            >>> # Standard quantile analysis
            >>> quant_layout = mle_analysis.plot_observed_quantiles()
            >>> # Custom window size for trend analysis
            >>> quant_plots = mle_analysis.plot_observed_quantiles(
            ...     windowsize=50, use_ranks=False, display=False
            ... )
        """
        # Loop over quantiles for different observed variables
        plots: dict[str, hv.Overlay] = {}
        for varname, reference, observed in self._iter_pp_obs():

            # Get the quantiles of the observed data relative to the reference
            y = plotting.calculate_relative_quantiles(
                reference, observed[None] if observed.ndim == 1 else observed
            )

            # Flatten the data and update x to use rankings if requested
            x, y = observed.ravel(), y.ravel()
            x = stats.rankdata(x, method="ordinal") if use_ranks else x

            # Build the plot
            plots[varname] = plotting.hexgrid_with_mean(
                x=x, y=y, mean_windowsize=windowsize
            ).opts(
                xlabel=f"Observed Value {'Rank' if use_ranks else ''}: {varname}",
                ylabel=f"Observed Quantile: {varname}",
                title=f"Observed Quantiles: {varname}",
                width=width,
                height=height,
            )

        # If requested, display the plots
        if display:
            return hv.Layout(plots.values()).cols(1).opts(shared_axes=False)

        return plots

    @overload
    def run_ppc(
        self,
        *,
        use_ranks: bool,
        display: Literal[True],
        square_ecdf: bool,
        windowsize: Optional["custom_types.Integer"],
        quantiles: Sequence["custom_types.Float"],
        logy_ppc_samples: bool,
        subplot_width: "custom_types.Integer",
        subplot_height: "custom_types.Integer",
    ) -> pn.Column: ...

    @overload
    def run_ppc(
        self,
        *,
        use_ranks: bool,
        display: Literal[False],
        square_ecdf: bool,
        windowsize: Optional["custom_types.Integer"],
        quantiles: Sequence["custom_types.Float"],
        logy_ppc_samples: bool,
        subplot_width: "custom_types.Integer",
        subplot_height: "custom_types.Integer",
    ) -> list[dict[str, hv.Overlay]]: ...

    def run_ppc(
        self,
        *,
        use_ranks=True,
        display=True,
        square_ecdf=True,
        windowsize=None,
        quantiles=(0.025, 0.25, 0.5),
        logy_ppc_samples=False,
        subplot_width=600,
        subplot_height=400,
    ):
        """Execute comprehensive posterior predictive checking analysis.

        This method provides a complete posterior predictive checking workflow by
        combining multiple diagnostic approaches into a unified analysis. It runs
        the methods
        :py:meth:`~scistanpy.model.results.mle.MLEInferenceRes.plot_posterior_predictive_samples`,
        :py:meth:`~scistanpy.model.results.mle.MLEInferenceRes.plot_observed_quantiles`,
        and :py:meth:`~scistanpy.model.results.mle.MLEInferenceRes.check_calibration`,
        combining their outputs into either an interactive dashboard or a list of
        individual plot dictionaries.

        :param use_ranks: Whether to use ranks instead of raw values for x-axes.
            Defaults to True.
        :type use_ranks: bool
        :param display: Whether to return interactive dashboard layout. Defaults to True.
        :type display: bool
        :param square_ecdf: Whether to make ECDF plots square (width=height). Defaults
            to True.
        :type square_ecdf: bool
        :param windowsize: Size of rolling window for trend analysis. Defaults to
            None (automatic).
        :type windowsize: Optional[custom_types.Integer]
        :param quantiles: Quantiles for confidence intervals. Defaults to (0.025,
            0.25, 0.5).
        :type quantiles: Sequence[custom_types.Float]
        :param logy_ppc_samples: Whether to use log scale for posterior predictive
            plots. Defaults to False.
        :type logy_ppc_samples: bool
        :param subplot_width: Width of individual subplots in pixels. Defaults to 600.
        :type subplot_width: custom_types.Integer
        :param subplot_height: Height of individual subplots in pixels. Defaults to 400.
        :type subplot_height: custom_types.Integer

        :returns: Interactive dashboard or list of plot dictionaries
        :rtype: Union[pn.Column, list[dict[str, hv.Overlay]]]

        Dashboard Features:

        - Interactive variable selection across all diagnostic types
        - Consistent formatting and scaling across related plots
        - Automatic layout optimization for comparison and analysis
        - Widget-based navigation for multi-variable models

        Between the three plots generated, this method provides a holistic view of
        model performance in terms of:

        - **Predictive accuracy**: How well do predictions match observations?
        - **Calibration quality**: Are prediction intervals properly calibrated?
        - **Systematic bias**: Are there patterns indicating model inadequacy?
        """
        # Get ecdf widths and heights
        if square_ecdf:
            ecdf_width = subplot_width
            ecdf_height = ecdf_width
        else:
            ecdf_width = subplot_width
            ecdf_height = subplot_height

        # Get the different plots
        plots = [
            self.plot_posterior_predictive_samples(
                quantiles=quantiles,
                use_ranks=use_ranks,
                logy=logy_ppc_samples,
                display=False,
                width=subplot_width,
                height=subplot_height,
            ),
            self.plot_observed_quantiles(
                use_ranks=use_ranks,
                display=False,
                width=subplot_width,
                height=subplot_height,
                windowsize=windowsize,
            ),
            self.check_calibration(
                return_deviance=False,
                display=False,
                width=ecdf_width,
                height=ecdf_height,
            ),
        ]

        # If not displaying, return the plots
        if not display:
            return plots

        # Otherwise, display the plots
        plots, widget = pn.panel(
            hv.Layout(
                [
                    hv.HoloMap(plots[0], kdims="Variable").opts(
                        hv.opts.Scatter(framewise=True),
                        hv.opts.Area(framewise=True),
                    ),
                    hv.HoloMap(plots[1], kdims="Variable").opts(
                        hv.opts.HexTiles(framewise=True, axiswise=True, min_count=0),
                        hv.opts.Curve(framewise=True, color="darkgray"),
                    ),
                    hv.HoloMap(plots[2], kdims="Variable").opts(
                        hv.opts.Curve(framewise=True),
                    ),
                ]
            )
            .opts(shared_axes=False)
            .cols(1)
        )
        widget.align = ("start", "start")

        return pn.Column(widget, plots)

    @classmethod
    def from_disk(cls, path: str, use_dask: bool = False) -> "InferenceRes":
        """Load InferenceRes object from saved NetCDF file with full analysis capabilities.

        Reconstructs a complete analysis object from a previously saved NetCDF
        file, restoring all inference results, computed statistics, and enabling
        continued analysis from where previous sessions left off.

        :param path: Path to NetCDF file containing saved ArviZ InferenceData
        :type path: str
        :param use_dask: Whether to enable Dask for memory-efficient processing
            of loaded data
        :type use_dask: bool, default False

        :returns: Reconstructed InferenceRes object with all methods available
        :rtype: InferenceRes

        Notes
        -----
        The loaded object provides complete functionality:

        - All original inference results (posterior, posterior_predictive, etc.)
        - Previously computed summary statistics and diagnostics (if any)
        - Full visualization and analysis capabilities
        - Ability to compute additional statistics or save updates

        When ``use_dask=True``, the loaded data is configured for chunked
        processing, enabling analysis of large datasets that exceed memory
        capacity.

        This method is particularly useful for:

        - Resuming interrupted analysis workflows
        - Sharing analysis results between collaborators
        - Creating reproducible analysis pipelines
        - Separating computation from visualization/reporting
        """
        return cls(inference_obj=path, use_dask=use_dask)
