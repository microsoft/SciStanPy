"""Holds the base class for converting SciStanPy results to NetCDF format."""

from abc import ABC, abstractmethod
from typing import Any, Generator, Literal, Union

import h5netcdf
import numpy as np
import numpy.typing as npt

import scistanpy
from scistanpy import custom_types, utils
from scistanpy.model.components.transformations import transformed_parameters


class SciStanPyToNetCDFConverter(ABC):
    """Base class responsible for converting SciStanPy outputs to NetCDF format.
    This class is the base class for
    :py:class:`~scistanpy.model.results.hmc.CmdStanMCMCToNetCDFConverter` and
    :py:class:`~scistanpy.model.results.mle.MLEToNetCDFConverter`. It should not
     be instantiated directly in most use cases.

    :param results: Abstract parameter representing SciStanPy results
    :type results: Any
    :param model: SciStanPy model object for metadata extraction
    :type model: Model
    :param data: Optional observed data dictionary. Defaults to None.
    :type data: Optional[dict[str, Any]]

    :ivar fit: CmdStanMCMC object containing sampling results
    :ivar model: Reference to the original SciStanPy model
    :ivar data: Observed data used for model fitting
    :ivar config: Configuration dictionary from Stan sampling
    :ivar num_draws: Total number of draws including warmup if saved
    :ivar varname_to_column_order: Mapping from variables to csv column indices

    The converter handles:

    - Automatic detection of variable types and dimensions
    - Proper NetCDF group organization
    - Chunking strategies for large datasets
    - Data type optimization based on precision requirements
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
        """
        Initialization involves collecting information about the different variables
        in the results object. This includes the names of the variables, their shapes,
        and their types. This information is used to create the HDF5 file.
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
        """Determine data types and dimension names for variables.

        :param precision: Numerical precision specification
        :type precision: Literal["double", "single", "half"]

        :returns: Tuple of (data_types_dict, dimension_names_dict)
        :rtype: tuple[dict[str, Union[type[np.floating], type[np.integer]]],
            dict[str, tuple[tuple[str, int], ...]]]

        This method analyzes the SciStanPy model to determine appropriate
        NumPy data types and dimension naming schemes for all variables
        that will be stored in the NetCDF file.
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
        """Write attributes to the NetCDF file. By default, this method does nothing
        (no attributes are written). Subclasses can override this method to add
        specific attributes as needed.

        :param netcdf_file: Opened NetCDF file object
        :type netcdf_file: h5netcdf.File

        This method writes the provided attributes to the root of the NetCDF file.
        """
        # Does nothing in the base class
        return

    def _create_netcdf_groups(
        self, netcdf_file: h5netcdf.File
    ) -> dict[str, h5netcdf.Group]:
        """Create necessary groups in the NetCDF file. By default, this method
        creates the groups 'posterior', 'posterior_predictive', and 'observed_data'.
        Subclasses can override this method to add specific groups as needed.

        :param netcdf_file: Opened NetCDF file object
        :type netcdf_file: h5netcdf.File
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
        """Write the converted data to NetCDF format.

        :param filename: Output filename.
        :type filename: str
        :param precision: Numerical precision for arrays. Defaults to "single".
        :type precision: Literal["double", "single", "half"]
        :param mib_per_chunk: Memory limit per chunk in MiB. Defaults to None, meaning
            use Dask default.
        :type mib_per_chunk: Optional[custom_types.Integer]

        :returns: Path to the created NetCDF file
        :rtype: str

        This method orchestrates the complete conversion process:
        1. Creates NetCDF file with appropriate structure
        2. Sets up dimensions based on model and data characteristics
        3. Creates variables with optimal chunking strategies
        4. Populates data

        The resulting NetCDF file contains properly organized groups for
        posterior samples, posterior predictive samples, sample statistics,
        and observed data.
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
        """Stream draws from the results object.

        :yields: Tuples of (chain_index, draw_index, draw_data)
        :rtype: Generator[tuple[int, int, dict[str, npt.ArrayLike]], None, None]

        This abstract method should be implemented by subclasses to yield
        individual draws from the results object in a memory-efficient manner.
        Each yielded item should include the chain index, draw index, and
        a dictionary mapping variable names to their sampled values for that draw.
        """
