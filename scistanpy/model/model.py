"""Core Model class for SciStanPy Bayesian modeling framework.

This module contains the fundamental Model class that serves as the primary
interface for building, compiling, and executing Bayesian models in SciStanPy.
The Model class orchestrates the composition of model components (parameters,
constants, transformations) and provides methods for sampling, compilation,
and analysis.

The Model class uses a metaclass pattern to automatically register model
components defined as instance attributes, enabling intuitive model construction
through simple attribute assignment. It supports multiple backends including
Stan for MCMC sampling and PyTorch for maximum likelihood estimation.

Key Features:
    - Automatic component registration and validation
    - Multiple sampling backends (Stan MCMC, PyTorch MLE)
    - Prior and posterior predictive checking
    - Efficient compilation with caching
    - Comprehensive model introspection and diagnostics
    - xarray integration for structured data handling
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import os.path
import pickle

from typing import Any, Iterable, Literal, Optional, overload, TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
import panel as pn
import torch
import xarray as xr

from scistanpy import utils
from scistanpy.defaults import (
    DEFAULT_CPP_OPTIONS,
    DEFAULT_DIM_NAMES,
    DEFAULT_EARLY_STOP,
    DEFAULT_FORCE_COMPILE,
    DEFAULT_LR,
    DEFAULT_MODEL_NAME,
    DEFAULT_N_EPOCHS,
    DEFAULT_STANC_OPTIONS,
    DEFAULT_USER_HEADER,
)
from scistanpy.model.components import abstract_model_component
from scistanpy.model.components import (
    constants as constants_module,
    parameters as parameters_module,
)
from scistanpy.model.components.transformations import (
    transformed_data,
    transformed_parameters as transformed_parameters_module,
)

if TYPE_CHECKING:
    from scistanpy import custom_types
    from scistanpy.model.results import hmc as hmc_results

mle_module = utils.lazy_import("scistanpy.model.results.mle")
nn_module = utils.lazy_import("scistanpy.model.nn_module")
prior_predictive_module = utils.lazy_import("scistanpy.plotting.prior_predictive")
stan_model = utils.lazy_import("scistanpy.model.stan.stan_model")


def model_comps_to_dict(
    model_comps: Iterable[abstract_model_component.AbstractModelComponent],
) -> dict[str, abstract_model_component.AbstractModelComponent]:
    """Convert an iterable of model components to a dictionary keyed by variable names.

    This utility function creates a dictionary mapping from model variable names
    to their corresponding component objects, facilitating easy lookup and
    access to model components by name.

    :param model_comps: Iterable of model components to convert
    :type model_comps: Iterable[abstract_model_component.AbstractModelComponent]

    :returns: Dictionary mapping variable names to components
    :rtype: dict[str, abstract_model_component.AbstractModelComponent]

    Example:
        >>> components = [param1, param2, observable1]
        >>> comp_dict = model_comps_to_dict(components)
        >>> # Access by name: comp_dict['param1']
    """
    return {comp.model_varname: comp for comp in model_comps}


def run_delayed_mcmc(filepath: str) -> "hmc_results.SampleResults":
    """Execute a delayed MCMC run from a pickled configuration file.

    This function loads and executes MCMC sampling that was previously
    configured with `Model.mcmc(delay_run=True)`. It's useful for
    running computationally intensive sampling jobs in batch systems
    or separate processes.

    :param filepath: Path to the pickled MCMC configuration file
    :type filepath: str

    :returns: MCMC sampling results with posterior draws and diagnostics
    :rtype: hmc_results.SampleResults

    The function automatically enables console output to provide progress
    feedback during the potentially long-running sampling process.

    Example:
        >>> # First, create delayed run
        >>> model.mcmc(output_dir = ".", delay_run=True, chains=4, iter_sampling=2000)
        >>> # Later, execute the run
        >>> results = run_delayed_mcmc(f"{model.stan_executable_path}-delay.pkl")
    """
    # Load the pickled object
    with open(filepath, "rb") as f:
        obj = pickle.load(f)

    # We will be printing to the console
    obj["sample_kwargs"]["show_console"] = True

    # Run sampling and return the results
    return obj["stan_model"].sample(
        inits=obj["inits"],
        data=obj["data"],
        **obj["sample_kwargs"],
    )


class Model:
    """Primary interface for Bayesian model construction and analysis in SciStanPy.

    The Model class provides a declarative interface for building Bayesian models
    by composing parameters, constants, and transformations. It automatically
    handles component registration, validation, and provides methods for sampling,
    compilation, and analysis across multiple backends.

    :param args: Positional arguments (unused, for subclass compatibility)
    :param default_data: Default observed data for model observables. When provided,
        any instance method requiring data will use these if not otherwise provided.
        Defaults to None.
    :type default_data: Optional[dict[str, npt.NDArray]]
    :param kwargs: Additional keyword arguments (unused, for subclass compatibility)

    :ivar _default_data: Stored default data for observables
    :ivar _named_model_components: Tuple of components that have explicit names
    :ivar _model_varname_to_object: Mapping from variable names to components
    :ivar _init_complete: Flag indicating initialization completion

    The class uses a metaclass pattern that automatically registers model
    components defined as instance attributes. Components are validated
    for naming conventions and automatically assigned model variable names.

    Model Construction:
        Models are built by subclassing Model and defining components as
        instance attributes in the __init__ method. The metaclass automatically
        discovers and registers these components.

    Supported Operations:
        - Prior and posterior sampling
        - Maximum likelihood estimation using PyTorch
        - MCMC sampling using Stan
        - Model compilation and caching
        - Prior predictive checking with interactive visualization
        - Model simulation and validation

    Example:
        >>> class MyModel(Model):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.mu = ssp.parameters.Normal(0, 1)
        ...         self.sigma = ssp.parameters.HalfNormal(1)
        ...         self.y = ssp.parameters.Normal(self.mu, self.sigma, observable=True)
        >>>
        >>> model = MyModel()
        >>> prior_samples = model.draw(n=1000)
        >>> mle_result = model.mle()
    """

    def __init__(
        self,
        *args,  # pylint: disable=unused-argument
        default_data: dict[str, npt.NDArray] | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """This should be overridden by the subclass."""
        # Set the default values for the model
        self._default_data: dict[str, npt.NDArray] | None = default_data

        self._named_model_components: tuple[
            abstract_model_component.AbstractModelComponent, ...
        ] = getattr(self, "_named_model_components", ())

        self._model_varname_to_object: dict[
            str, abstract_model_component.AbstractModelComponent
        ] = getattr(self, "_model_varname_to_object", {})

        self._init_complete: bool = getattr(self, "_init_complete", False)

    def __init_subclass__(cls, **kwargs):
        """Configure automatic component registration for Model subclasses.

        This method is called when a class inherits from Model and sets up
        the metaclass behavior that automatically discovers and registers
        model components defined as instance attributes.

        :param kwargs: Keyword arguments passed to the subclass

        :raises ValueError: If forbidden attribute names are used or naming
                           conventions are violated

        The method wraps the subclass __init__ to add component discovery
        and registration logic while preserving the original initialization
        behavior.
        """
        # The old __init__ method of the class is renamed to '_wrapped_init'
        if "_wrapped_init" in cls.__dict__:
            raise ValueError(
                "The attribute `_wrapped_init` cannot be defined in `Model` subclasses"
            )

        # Redefine the __init__ method of the class
        def __init__(
            self: "Model",
            *init_args,
            **init_kwargs,
        ):

            # Initialization is incomplete at this stage
            self._init_complete = False

            # Run the init method that was defined in the class.
            cls._wrapped_init(self, *init_args, **init_kwargs)

            # That's it if we are not the last subclass to be initialized
            if cls is not self.__class__:
                return

            # If we already have model components, defined in the class, update
            # them with the new model components. This situation occurs when a child
            # class is defined that inherits from a parent class that is also a
            # model.
            named_model_components = {}

            # Now we need to find all the model components that are defined in the
            # class.
            for attr in vars(self).keys():
                if not isinstance(
                    retrieved := getattr(self, attr),
                    abstract_model_component.AbstractModelComponent,
                ):
                    continue

                # Double-underscore attributes are forbidden, as this will clash
                # with how we handle unnamed parameters in Stan code.
                if "__" in attr:
                    raise ValueError(
                        "Model component names cannot include double underscores: "
                        f"{attr} is invalid."
                    )

                # Check if the variable name starts with an underscore. This is
                # forbidden in Stan code.
                if attr.startswith("_"):
                    raise ValueError(
                        "Model variable names cannot start with an underscore: "
                        f"{attr} is invalid."
                    )

                # Set the model variable name and record the model component
                retrieved.model_varname = attr
                named_model_components[attr] = retrieved

            # Set the named parameters attribute
            self._named_model_components = tuple(named_model_components.values())

            # Build the mapping between model variable names and parameter objects
            self._model_varname_to_object = self._build_model_varname_to_object()

            # Initialization is complete
            self._init_complete = True

            # Set default data as itself. This will trigger the setter method
            # and will check that the data is valid.
            if self.has_default_data:
                self.default_data = self.default_data

        # Add the new __init__ method
        cls._wrapped_init = cls.__init__
        cls.__init__ = __init__

    def _build_model_varname_to_object(
        self,
    ) -> dict[str, abstract_model_component.AbstractModelComponent]:
        """Build comprehensive mapping from variable names to model components.

        This method constructs a complete dictionary mapping model variable
        names to their corresponding component objects. It walks the component
        dependency tree starting from observables to ensure all referenced
        components are included.

        :returns: Dictionary mapping variable names to components
        :rtype: dict[str, abstract_model_component.AbstractModelComponent]

        The mapping includes:
        - All observable parameters and their dependencies
        - Transformed data components
        - Constants and hyperparameters
        - Transformed parameters used in the model

        The method ensures no duplicate variable names exist and validates
        the integrity of the component dependency graph.
        """

        def build_initial_mapping() -> (
            dict[str, abstract_model_component.AbstractModelComponent]
        ):
            """Builds an initial mapping of model varnames to objects."""

            # Start from each observable and walk up the tree to the root
            model_varname_to_object: dict[
                str, abstract_model_component.AbstractModelComponent
            ] = {}
            for observable in self.observables:

                # Add the observable to the mapping
                assert observable.model_varname not in model_varname_to_object
                model_varname_to_object[observable.model_varname] = observable

                # Add all parents to the mapping and make sure
                # `parameters_module.Parameter` instances are explicitly defined.
                for *_, parent in observable.walk_tree(walk_down=False):

                    # If the parent is already in the mapping, make sure it is the
                    # same
                    if parent.model_varname in model_varname_to_object:
                        assert model_varname_to_object[parent.model_varname] == parent
                    else:
                        model_varname_to_object[parent.model_varname] = parent

            return model_varname_to_object

        def record_transformed_data() -> None:
            """Updates the mapping with all transformed data components."""

            # Add all TransformedData instances to the mapping
            for component in list(model_varname_to_object.values()):
                for child in component._children:  # pylint: disable=protected-access
                    if isinstance(
                        child,
                        transformed_data.TransformedData,
                    ):
                        assert child.model_varname not in model_varname_to_object
                        model_varname_to_object[child.model_varname] = child

        # Run the steps
        model_varname_to_object = build_initial_mapping()
        record_transformed_data()

        # There can be no duplicate values in the mapping
        assert len(model_varname_to_object) == len(
            set(model_varname_to_object.values())
        )

        return model_varname_to_object

    def get_dimname_map(
        self,
    ) -> dict[tuple["custom_types.Integer", "custom_types.Integer"], str]:
        """Generate mapping from dimension specifications to dimension names.

        This method creates a dictionary that maps dimension level and size
        tuples to appropriate dimension names for xarray dataset construction.
        It ensures dimension names don't conflict with model variable names.

        :returns: Dictionary mapping (level, size) tuples to dimension names
        :rtype: dict[tuple[custom_types.Integer, custom_types.Integer], str]

        The mapping is used to create consistent dimension naming across
        all xarray datasets generated from model samples, ensuring proper
        coordinate alignment and data structure.

        Only dimensions with size > 1 are assigned names, as singleton
        dimensions are typically squeezed during xarray construction.
        """
        # Set up variables
        dims: dict[tuple["custom_types.Integer", "custom_types.Integer"], str] = {}

        # The list of dimension options cannot overlap with variable names
        allowed_dim_names = [
            name
            for name in DEFAULT_DIM_NAMES
            if name not in self.named_model_components_dict
        ]

        # Check sizes of all observables and record the dimension names. Dimensions
        # of size '1' are not named as we do not need to distinguish them in an xarray
        for observable in self.named_model_components:
            for dimkey in enumerate(observable.shape[::-1]):
                if dimkey not in dims and dimkey[1] > 1:
                    dims[dimkey] = allowed_dim_names[len(dims)]

        return dims

    def _compress_for_xarray(
        self,
        *arrays: npt.NDArray,
        include_sample_dim: bool = False,
    ) -> list[tuple[tuple[str, ...], npt.NDArray]]:
        """Process arrays for xarray dataset construction with proper dimension naming.

        This method transforms numpy arrays into the format required for xarray
        dataset construction, including appropriate dimension naming and
        singleton dimension handling.

        :param arrays: Arrays to process for xarray construction
        :type arrays: npt.NDArray
        :param include_sample_dim: Whether to include sample dimension in naming.
                                  Defaults to False.
        :type include_sample_dim: bool

        :returns: List of (dimension_names, processed_array) tuples
        :rtype: list[tuple[tuple[str, ...], npt.NDArray]]

        :raises ValueError: If array dimensions don't match expected model structure

        The method:
        - Identifies and removes singleton dimensions
        - Assigns appropriate dimension names based on model structure
        - Handles sample dimensions for drawn data (e.g., from prior predictive checks)
        - Ensures dimensional consistency across all processed arrays
        """
        # Get a mapping from dimension keys to dimension names
        dims = self.get_dimname_map()

        # Set our start and end indices for the shape
        start_ind = int(include_sample_dim)

        # Process each input array
        processed: list[tuple[tuple[str, ...], npt.NDArray]] = []
        for array_ind, array in enumerate(arrays):

            # Identify singleton dimensions and named dimensions
            singleton_axes, dimnames = [], []
            effective_shape = array.shape[start_ind:]
            for dimind, dimsize in enumerate(effective_shape[::-1]):
                forward_ind = len(effective_shape) - 1 - dimind + start_ind
                if dimsize == 1:
                    singleton_axes.append(forward_ind)
                else:
                    try:
                        dimnames.append(dims[(dimind, dimsize)])
                    except KeyError as error:
                        raise ValueError(
                            f"There is no dimension index of {forward_ind - start_ind} "
                            f"with size {dimsize} in this model. Error triggered "
                            f"by array {array_ind}. Options are: {dims}"
                        ) from error

            # Append "n" to the dimension names if we are including the sample dimension
            if include_sample_dim:
                dimnames.append("n")

            # Squeeze the array
            processed.append(
                (tuple(dimnames[::-1]), np.squeeze(array, axis=tuple(singleton_axes)))
            )

        return processed

    def _dict_to_xarray(
        self,
        draws: dict[abstract_model_component.AbstractModelComponent, npt.NDArray],
    ) -> xr.Dataset:
        """Convert model component draws dictionary to structured xarray Dataset.

        This method transforms a dictionary of model component draws into a
        properly structured xarray Dataset with appropriate coordinates
        and dimension names for analysis and visualization.

        :param draws: Dictionary mapping components to their sampled values
        :type draws: dict[abstract_model_component.AbstractModelComponent, npt.NDArray]

        :returns: Structured dataset with draws and coordinates
        :rtype: xr.Dataset

        The resulting dataset includes:
        - Data variables for all non-constant components
        - Coordinates for multi-dimensional constants
        - Proper dimension naming and alignment
        """
        # Split into components and draws and components and values
        model_comps, unpacked_draws = zip(
            *[
                [comp, draw]
                for comp, draw in draws.items()
                if not isinstance(comp, constants_module.Constant)
            ]
        )
        coordinates = list(
            zip(
                *[
                    [parent, parent.value]
                    for component in self.all_model_components
                    for parent in component.parents
                    if isinstance(parent, constants_module.Constant)
                    and np.prod(parent.shape) > 1
                ]
            )
        )
        if len(coordinates) == 0:
            parents, values = [], []
        else:
            parents, values = coordinates

        # Process the draws and values for xarray. Note that because constants
        # have no sample prefix, we do not add the sampling dimension to them
        # when calling `compress_for_xarray` (i.e., we do not use `_n`).
        compressed_draws = self._compress_for_xarray(
            *unpacked_draws, include_sample_dim=True
        )
        compressed_values = self._compress_for_xarray(*values)

        # Build kwargs
        return xr.Dataset(
            data_vars={
                component.model_varname: compressed_draw
                for component, compressed_draw in zip(model_comps, compressed_draws)
            },
            coords={
                parent.model_varname: compressed_value
                for parent, compressed_value in zip(parents, compressed_values)
            },
        )

    @overload
    def draw(
        self,
        n: "custom_types.Integer",
        *,
        named_only: Literal[True],
        as_xarray: Literal[False],
        seed: Optional["custom_types.Integer"],
    ) -> dict[str, npt.NDArray]: ...

    @overload
    def draw(
        self,
        n: "custom_types.Integer",
        *,
        named_only: Literal[False],
        as_xarray: Literal[False],
        seed: Optional["custom_types.Integer"],
    ) -> dict[abstract_model_component.AbstractModelComponent, npt.NDArray]: ...

    @overload
    def draw(
        self,
        n: "custom_types.Integer",
        *,
        named_only: Literal[True],
        as_xarray: Literal[True],
        seed: Optional["custom_types.Integer"],
    ) -> xr.Dataset: ...

    @overload
    def draw(
        self,
        n: "custom_types.Integer",
        *,
        named_only: Literal[False],
        as_xarray: Literal[True],
        seed: Optional["custom_types.Integer"],
    ) -> xr.Dataset: ...

    def draw(self, n, *, named_only=True, as_xarray=False, seed=None):
        """Draw samples from the model's prior distribution.

        This method generates samples from all observable parameters in the
        model by traversing the dependency graph and sampling from each
        component's distribution in topological order.

        :param n: Number of samples to draw from each observable
        :type n: custom_types.Integer
        :param named_only: Whether to return only named components. Defaults to True.
        :type named_only: bool
        :param as_xarray: Whether to return results as xarray Dataset. Defaults to False.
        :type as_xarray: bool
        :param seed: Random seed for reproducible sampling. Defaults to None.
        :type seed: Optional[custom_types.Integer]

        :returns: Sampled values in requested format
        :rtype: Union[dict[str, npt.NDArray], dict[AbstractModelComponent, npt.NDArray], xr.Dataset]

        The method automatically handles:
        - Dependency resolution between model components
        - Consistent random number generation with optional seeding
        - Efficient sampling by reusing intermediate results
        - Format conversion based on return type preferences

        Example:
            >>> # Draw 1000 samples as dictionary
            >>> samples = model.draw(1000)
            >>> # Draw 1000 samples as an xarray Dataset
            >>> dataset = model.draw(1000, as_xarray=True)
        """
        # Draw from all observables
        draws: dict[abstract_model_component.AbstractModelComponent, npt.NDArray] = {}
        for observable in self.observables:
            _, draws = observable.draw(n, _drawn=draws, seed=seed)

        # Filter down to just named parameters if requested
        if named_only:
            draws = {k: v for k, v in draws.items() if k.is_named}

        # Convert to an xarray dataset if requested
        if as_xarray:
            return self._dict_to_xarray(draws)

        # If we are returning only named parameters, then we need to update the
        # dictionary keys to be the model variable names.
        if named_only:
            return {k.model_varname: v for k, v in draws.items()}
        return draws

    def to_pytorch(
        self, seed: Optional["custom_types.Integer"] = None
    ) -> "nn_module.PyTorchModel":
        """Compile the model to a trainable PyTorch module.

        This method converts the SciStanPy model into a PyTorch module that
        can be optimized using standard PyTorch training procedures for
        maximum likelihood estimation or variational inference. The inputs to this
        module (i.e., the inputs to its `forward` method) are all observed data;
        the output is the likelihood of that data given the current model parameters.

        :param seed: Random seed for reproducible compilation. Defaults to None.
        :type seed: Optional[custom_types.Integer]

        :returns: Compiled PyTorch model ready for training
        :rtype: nn_module.PyTorchModel

        The compiled model preserves the probabilistic structure while
        enabling gradient-based optimization of model parameters. It's
        particularly useful for maximum likelihood estimation and can
        leverage GPU acceleration for large models.
        """
        return nn_module.PyTorchModel(self, seed=seed)

    def to_stan(self, **kwargs) -> "stan_model.StanModel":
        """Compile the model to Stan code for MCMC sampling.

        This method automatically generates Stan probabilistic programming
        language code from the SciStanPy model specification and compiles
        it for Hamilitonian Monte-Carlo sampling.

        :param kwargs: Additional compilation options passed to StanModel

        :returns: Compiled Stan model ready for MCMC sampling
        :rtype: stan_model.StanModel
        """
        return stan_model.StanModel(self, **kwargs)

    def mle(
        self,
        epochs: "custom_types.Integer" = DEFAULT_N_EPOCHS,
        early_stop: "custom_types.Integer" = DEFAULT_EARLY_STOP,
        lr: "custom_types.Float" = DEFAULT_LR,
        data: Optional[dict[str, Union[torch.Tensor, npt.NDArray]]] = None,
        device: "custom_types.Integer" | str = "cpu",
        seed: Optional["custom_types.Integer"] = None,
        mixed_precision: bool = False,
    ) -> "mle_module.MLE":
        """Compute maximum likelihood estimates of model parameters.

        This method fits a PyTorch model to observed data by minimizing
        the negative log-likelihood, providing point estimates of all
        model parameters along with optimization diagnostics.

        :param epochs: Maximum number of training epochs. Note that one step is one
            epoch as the model must be evaluated over all observable data to calculate
            loss. Defaults to 100000.
        :type epochs: custom_types.Integer
        :param early_stop: Epochs without improvement before stopping. Defaults to 10.
        :type early_stop: custom_types.Integer
        :param lr: Learning rate for optimization. Defaults to 0.001.
        :type lr: custom_types.Float
        :param data: Observed data for observables. Uses default_data provided at
            initialization if not provided.
        :type data: Optional[dict[str, Union[torch.Tensor, npt.NDArray]]]
        :param device: Computation device ('cpu', 'cuda', or device index). Defaults to 'cpu'.
        :type device: Union[custom_types.Integer, str]
        :param seed: Random seed for reproducible optimization. Defaults to None.
        :type seed: Optional[custom_types.Integer]
        :param mixed_precision: Whether to use mixed precision training. Defaults to False.
        :type mixed_precision: bool

        :returns: MLE results with parameter estimates and diagnostics
        :rtype: mle_module.MLE

        The optimization process:
        - Converts model to PyTorch and moves to specified device
        - Trains for `epochs` number of epochs or until there has been no improvement
          for `early_stop` number epochs.
        - Tracks loss trajectory for convergence assessment
        - Returns parameter estimates and fitted distributions

        Example:
            >>> # Basic MLE with default settings
            >>> mle_result = model.mle(data=observed_data)
            >>> # GPU-accelerated with custom settings
            >>> mle_result = model.mle(data=obs, device='cuda', epochs=50000, lr=0.01)
        """
        # Set the default value for observed data
        data = self.default_data if data is None else data

        # Observed data to tensors and the appropriate device
        data = {
            k: (
                v.to(device=device)
                if isinstance(v, torch.Tensor)
                else torch.tensor(v).to(device=device)
            )
            for k, v in data.items()
        }

        # Check observed data
        nn_module.check_observable_data(self, data)

        # Fit the model
        pytorch_model = self.to_pytorch(seed=seed).to(device=device)
        loss_trajectory = pytorch_model.fit(
            epochs=epochs,
            early_stop=early_stop,
            lr=lr,
            data=data,
            mixed_precision=mixed_precision,
        )

        # Get the MLE estimate for all model parameters
        max_likelihood = {
            k: v.detach().cpu().numpy()
            for k, v in pytorch_model.export_params().items()
        }

        # Get the distributions of the parameters
        distributions = pytorch_model.export_distributions()

        # Return the MLE estimate, the distributions, and the loss trajectory
        return mle_module.MLE(
            model=self,
            mle_estimate=max_likelihood,
            distributions=distributions,
            losses=loss_trajectory.detach().cpu().numpy(),
            data={k: v.detach().cpu().numpy() for k, v in data.items()},
        )

    def _get_simulation_data(
        self, seed: Optional["custom_types.Integer"]
    ) -> dict[str, npt.NDArray]:
        """Generate simulated observable data from model prior.

        This internal method draws a single realization from each observable
        parameter in the model using the current prior specification. It is
        used by simulation methods to generate synthetic datasets.

        :param seed: Random seed for reproducible simulation
        :type seed: Optional[custom_types.Integer]

        :returns: Dictionary mapping observable names to simulated values
        :rtype: dict[str, npt.NDArray]
        """
        data = self.draw(1, named_only=True, as_xarray=False, seed=seed)
        return {
            observable.model_varname: data[observable.model_varname][0]
            for observable in self.observables
        }

    def simulate_mle(self, **kwargs) -> tuple[dict[str, npt.NDArray], "mle_module.MLE"]:
        """Simulate data from model prior and fit via maximum likelihood.

        This method performs a complete simulation study by first generating
        synthetic data from the model's prior distribution, then fitting
        the model to this simulated data using maximum likelihood estimation.

        :param kwargs: Keyword arguments passed to mle() method (except 'data')

        :returns: Tuple of (simulated_data, mle_results)
        :rtype: tuple[dict[str, npt.NDArray], mle_module.MLE]

        This is particularly useful for:
        - Model validation and debugging
        - Assessing parameter identifiability (e.g. by running multiple simulations)
        - Verifying implementation correctness

        The simulated data is automatically passed to the MLE fitting
        procedure, overriding any data specification in kwargs.

        Example:
            >>> # Simulate and fit with custom settings
            >>> sim_data, mle_fit = model.simulate_mle(epochs=10000, lr=0.01)
        """
        # TODO: This and the other simulate method should include non-observables
        # in the returned MLE as well.
        # Get the data
        kwargs["data"] = self._get_simulation_data(seed=kwargs.get("seed"))

        # Fit the model
        return kwargs["data"], self.mle(**kwargs)

    @overload
    def mcmc(
        self,
        *,
        output_dir: Optional[str],
        force_compile: bool,
        stanc_options: Optional[dict[str, Any]],
        cpp_options: Optional[dict[str, Any]],
        user_header: Optional[str],
        model_name: Optional[str],
        inits: Optional[str],
        data: Optional[dict[str, npt.NDArray]],
        delay_run: Literal[False],
        **sample_kwargs,
    ) -> "hmc_results.SampleResults": ...
    @overload
    def mcmc(
        self,
        *,
        output_dir: Optional[str],
        force_compile: bool,
        stanc_options: Optional[dict[str, Any]],
        cpp_options: Optional[dict[str, Any]],
        model_name: Optional[str],
        user_header: Optional[str],
        inits: Optional[str],
        data: Optional[dict[str, npt.NDArray]],
        delay_run: Literal[True] | str,
        **sample_kwargs,
    ) -> None: ...

    def mcmc(
        self,
        *,
        output_dir=None,
        force_compile=DEFAULT_FORCE_COMPILE,
        stanc_options=None,
        cpp_options=None,
        user_header=DEFAULT_USER_HEADER,
        model_name=DEFAULT_MODEL_NAME,
        inits=None,
        data=None,
        delay_run=False,
        **sample_kwargs,
    ):
        """Perform Hamiltonia Monte Carlo sampling using Stan backend.

        This method compiles the model to Stan and executes Hamiltonian
        Monte Carlo sampling to generate posterior samples. It supports
        both immediate execution and delayed runs for batch processing.

        :param output_dir: Directory for compilation and output files. Defaults to None,
            in which case all raw outputs will be saved to a temporary directory and
            be accessible only for the lifetime of this Python process.
        :type output_dir: Optional[str]
        :param force_compile: Whether to force recompilation of Stan model. Defaults
            to False.
        :type force_compile: bool
        :param stanc_options: Options for Stan compiler. Defaults to None (uses
            DEFAULT_STANC_OPTIONS).
        :type stanc_options: Optional[dict[str, Any]]
        :param cpp_options: Options for C++ compilation. Defaults to None (uses
            DEFAULT_CPP_OPTIONS).
        :type cpp_options: Optional[dict[str, Any]]
        :param user_header: Custom C++ header code. Defaults to None.
        :type user_header: Optional[str]
        :param model_name: Name for compiled model. Defaults to 'model'.
        :type model_name: Optional[str]
        :param inits: Initialization strategy. See `stan_model.StanModel`
            for options. Defaults to None.
        :type inits: Optional[str]
        :param data: Observed data for observables. Uses default_data defined during
            initialization if not provided.
        :type data: Optional[dict[str, npt.NDArray]]
        :param delay_run: Whether to delay execution. If `True`, a pickle file that
            can be used for delayed execution will be saved to `output_dir`. A string
            can also be provided to save the pickle file to an alternate location.
            Defaults to False (meaning immediate execution).
        :type delay_run: Union[bool, str]
        :param sample_kwargs: Additional arguments passed to Stan sampling. See
            the `cmdstanpy.CmdStanModel.sample` for options.


        :returns: MCMC results if delay_run=False, None if delayed
        :rtype: Union[hmc_results.SampleResults, None]

        :raises ValueError: If delay_run is True but output_dir is None

        Delayed Execution:
        When delay_run=True, the method saves sampling configuration to a
        pickle file instead of executing immediately. This enables batch
        processing and distributed computing workflows.

        Example:
            >>> # Immediate MCMC sampling
            >>> results = model.mcmc(chains=4, iter_sampling=2000)
            >>> # Delayed execution for batch processing
            >>> model.mcmc(delay_run='batch_job.pkl', chains=8, iter_sampling=5000)
        """
        # Get the default observed data and cpp options
        data = self.default_data if data is None else data
        stanc_options = stanc_options or DEFAULT_STANC_OPTIONS
        cpp_options = cpp_options or DEFAULT_CPP_OPTIONS

        # An output directory must be provided if we are delaying the run
        if delay_run and output_dir is None:
            raise ValueError(
                "An output directory must be provided if `delay_run` is True."
            )

        # Build the Stan model
        model = self.to_stan(
            output_dir=output_dir,
            force_compile=force_compile,
            stanc_options=stanc_options,
            cpp_options=cpp_options,
            user_header=user_header,
            model_name=model_name,
        )

        # Update the output directory in the sample kwargs
        sample_kwargs["output_dir"] = os.path.abspath(model.output_dir)

        # If delaying, then we save the data needed for sampling and return
        if delay_run:
            with open(
                (
                    delay_run
                    if isinstance(delay_run, str)
                    else f"{model.stan_executable_path}-delay.pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(
                    {
                        "stan_model": model,
                        "sample_kwargs": sample_kwargs,
                        "data": data,
                        "inits": inits,
                    },
                    f,
                )
            return

        # Sample from the model
        return model.sample(inits=inits, data=data, **sample_kwargs)

    @overload
    def simulate_mcmc(
        self, delay_run: Literal[False], **kwargs
    ) -> tuple[dict[str, npt.NDArray], "hmc_results.SampleResults"]: ...
    @overload
    def simulate_mcmc(self, delay_run: Literal[True], **kwargs) -> None: ...

    def simulate_mcmc(self, delay_run=False, **kwargs):
        """Simulate data from model prior and perform Hamiltonian Monte Carlo sampling.

        This method generates synthetic data from the model's prior
        distribution and then performs full Bayesian inference via MCMC.
        It's extremely helpful for model validation and posterior recovery testing.

        :param delay_run: Whether to delay MCMC execution. Defaults to False.
        :type delay_run: bool
        :param kwargs: Additional keyword arguments passed to mcmc() method

        :returns: Tuple of (simulated_data, mcmc_results) if delay_run=False,
                 None if delay_run=True
        :rtype: Union[tuple[dict[str, npt.NDArray], hmc_results.SampleResults], None]

        The method automatically updates the model name to indicate
        simulation when using the default name, helping distinguish
        simulated from real data analyses.

        This is crucial for:
        - Validating MCMC implementation correctness
        - Testing posterior recovery in known-truth scenarios
        - Assessing sampler efficiency and convergence
        - Debugging model specification issues

        Example:
            >>> # Simulate and sample with immediate execution
            >>> sim_data, mcmc_results = model.simulate_mcmc(chains=4, iter_sampling=1000)
        """
        # Update the model name
        if kwargs.get("model_name") == DEFAULT_MODEL_NAME:
            kwargs["model_name"] = f"{DEFAULT_MODEL_NAME}-simulated"

        # Get the data
        kwargs["data"] = self._get_simulation_data(seed=kwargs.get("seed"))

        # Run MCMC
        return kwargs["data"], self.mcmc(delay_run=delay_run, **kwargs)

    def prior_predictive(self, *, copy_model: bool = False) -> pn.Row:
        """Create interactive prior predictive check visualization.

        This method generates an interactive dashboard for exploring the
        model's prior predictive distribution. Users can adjust model
        hyperparameters via sliders and immediately see how changes
        affect prior predictions.

        :param copy_model: Whether to copy model to avoid modifying original. Defaults
            to False, meaning the calling model is updated in place by changing
            slider values and clicking "update model".
        :type copy_model: bool

        :returns: Panel dashboard with interactive prior predictive visualization
        :rtype: pn.Row

        The dashboard includes:
        - Sliders for all adjustable model hyperparameters
        - Multiple visualization modes (ECDF, KDE, violin, relationship plots)
        - Real-time updates as parameters are modified
        - Options for different grouping and display configurations

        This is useful for:
        - Prior specification and calibration
        - Understanding model behavior before data fitting
        - Identifying unrealistic prior assumptions

        Example:
            >>> # Create interactive dashboard
            >>> dashboard = model.prior_predictive()
            >>> dashboard.servable()  # For web deployment
            >>> # Or display in Jupyter notebook
            >>> dashboard
        """
        # Create the prior predictive object
        pp = prior_predictive_module.PriorPredictiveCheck(self, copy_model=copy_model)

        # Return the plot
        return pp.display()

    def __str__(self) -> str:
        """Return comprehensive string representation of the model.

        :returns: Formatted string showing all model components organized by type
        :rtype: str

        The representation includes organized sections for:
        - Constants and hyperparameters
        - Transformed parameters
        - Regular parameters
        - Observable parameters

        Each section lists components with their specifications and
        current values, providing a complete overview of model structure.
        """
        # Get all model components
        model_comps = {
            "Constants": [
                el
                for el in self.all_model_components
                if isinstance(el, constants_module.Constant)
            ],
            "Transformed Parameters": self.transformed_parameters,
            "Parameters": self.parameters,
            "Observables": self.observables,
        }

        # Combine representations from all model components
        return "\n\n".join(
            key + "\n" + "=" * len(key) + "\n" + "\n".join(str(el) for el in complist)
            for key, complist in model_comps.items()
            if len(complist) > 0
        )

    def __contains__(self, paramname: str) -> bool:
        """Check if model contains a component with the given name.

        :param paramname: Name of the model component to check
        :type paramname: str

        :returns: True if component exists, False otherwise
        :rtype: bool

        Example:
            >>> 'mu' in model  # Check if parameter 'mu' exists
            True
        """
        return paramname in self._model_varname_to_object

    def __getitem__(
        self, paramname: str
    ) -> abstract_model_component.AbstractModelComponent:
        """Retrieve model component by name.

        :param paramname: Name of the model component to retrieve
        :type paramname: str

        :returns: The requested model component
        :rtype: abstract_model_component.AbstractModelComponent

        :raises KeyError: If component name doesn't exist

        Example:
            >>> mu_param = model['mu']  # Get parameter named 'mu'
            >>> print(mu_param.distribution)
        """
        return self._model_varname_to_object[paramname]

    def __setattr__(self, name: str, value: Any) -> None:
        """Set model attribute with protection for model components.

        :param name: Attribute name to set
        :type name: str
        :param value: Value to assign to the attribute
        :type value: Any

        :raises AttributeError: If attempting to modify existing model component
            or add a new model component after initialization.

        This method prevents modification of model components after
        initialization to maintain model integrity and prevent
        accidental corruption of the dependency graph.
        """
        # We cannot set attributes that are model components
        if (
            hasattr(self, "_model_varname_to_object")
            and name in self._model_varname_to_object
        ):
            raise AttributeError(
                "Model components can only be set during initialization."
            )

        # Otherwise, set the attribute
        super().__setattr__(name, value)

    @property
    def default_data(self) -> dict[str, npt.NDArray] | None:
        """Get the default observed data for model observables.

        :returns: Dictionary mapping observable names to their default data
        :rtype: dict[str, npt.NDArray] | None

        :raises ValueError: If default data has not been set

        Default data is used automatically by methods like mle() and mcmc()
        when no explicit data is provided, streamlining common workflows.
        """
        if getattr(self, "_default_data", None) is None:
            raise ValueError(
                "Default data is not set. Please set the default data using "
                "`model.default_data = data`."
            )
        return self._default_data

    @default_data.setter
    def default_data(self, data: dict[str, npt.NDArray] | None) -> None:
        """Set default observed data for model observables.

        :param data: Dictionary mapping observable names to their data, or None to clear
        :type data: dict[str, npt.NDArray] | None

        :raises ValueError: If data is missing required observable keys or contains extra keys

        The data dictionary must contain entries for all observable parameters
        in the model. Setting to None clears the default data.
        """
        # Reset the default data if `None` is passed
        if data is None:
            self._default_data = None
            return

        # If initialization is complete, the data must be a dictionary and we must
        # have all the appropriate keys. We skip this check if the model is not
        # initialized yet, as we do not know the model variable names yet.
        if self._init_complete:
            expected_keys = {comp.model_varname for comp in self.observables}
            if missing_keys := expected_keys - data.keys():
                raise ValueError(
                    f"The following keys are missing from the default data: {missing_keys}"
                )
            if extra_keys := data.keys() - expected_keys:
                raise ValueError(
                    f"The following keys are not expected in the data: {extra_keys}"
                )

        # Set the default data
        self._default_data = data

    @property
    def has_default_data(self) -> bool:
        """Check whether the model has default data configured.

        :returns: True if default data is set, False otherwise
        :rtype: bool

        This property is useful for conditional logic that depends on
        whether default data is available for automatic use in methods.
        """
        return getattr(self, "_default_data", None) is not None

    @property
    def named_model_components(
        self,
    ) -> tuple[abstract_model_component.AbstractModelComponent, ...]:
        """Get all named model components.

        :returns: Tuple of named components
        :rtype: tuple[abstract_model_component.AbstractModelComponent, ...]

        Named components are those explicitly assigned as instance attributes
        during model construction, as opposed to intermediate components
        created automatically during dependency resolution.
        """
        return self._named_model_components

    @property
    def named_model_components_dict(
        self,
    ) -> dict[str, abstract_model_component.AbstractModelComponent]:
        """Get named model components as a dictionary.

        :returns: Dictionary mapping variable names to named components
        :rtype: dict[str, abstract_model_component.AbstractModelComponent]

        This provides convenient access to named components by their
        string names for programmatic model inspection and manipulation.
        """
        return model_comps_to_dict(self.named_model_components)

    @property
    def all_model_components(
        self,
    ) -> tuple[abstract_model_component.AbstractModelComponent, ...]:
        """Get all model components including unnamed intermediate components.

        :returns: Tuple of all components sorted by variable name
        :rtype: tuple[abstract_model_component.AbstractModelComponent, ...]

        This includes both explicitly named components and any intermediate
        components created during dependency resolution, providing complete
        visibility into the model's computational graph.
        """
        return tuple(
            sorted(
                self._model_varname_to_object.values(), key=lambda x: x.model_varname
            )
        )

    @property
    def all_model_components_dict(
        self,
    ) -> dict[str, abstract_model_component.AbstractModelComponent]:
        """Get all model components as a dictionary.

        :returns: Dictionary mapping variable names to all components
        :rtype: dict[str, abstract_model_component.AbstractModelComponent]

        This comprehensive mapping includes both named and intermediate
        components, enabling full programmatic access to the model structure.
        """
        return self._model_varname_to_object

    @property
    def parameters(self) -> tuple[parameters_module.Parameter, ...]:
        """Get all non-observable parameters in the model.

        :returns: Tuple of parameter components that are not observables
        :rtype: tuple[parameters_module.Parameter, ...]

        These are the latent variables and hyperparameters that will be
        inferred during MCMC sampling or optimized during MLE fitting.
        """
        return tuple(
            filter(
                lambda x: isinstance(x, parameters_module.Parameter)
                and not x.observable,
                self.all_model_components,
            )
        )

    @property
    def parameter_dict(self) -> dict[str, parameters_module.Parameter]:
        """Get non-observable parameters as a dictionary.

        :returns: Dictionary mapping names to non-observable parameters
        :rtype: dict[str, parameters_module.Parameter]

        Provides convenient named access to the model's latent parameters
        for inspection and programmatic manipulation.
        """
        return model_comps_to_dict(self.parameters)

    @property
    def hyperparameters(self) -> tuple[parameters_module.Parameter, ...]:
        """Get hyperparameters (parameters with only constant parents).

        :returns: Tuple of parameters that depend only on constants
        :rtype: tuple[parameters_module.Parameter, ...]

        Hyperparameters are the highest-level parameters in the model
        hierarchy, typically representing prior distribution parameters
        that are not derived from other random variables.
        """
        return tuple(filter(lambda x: x.is_hyperparameter, self.parameters))

    @property
    def hyperparameter_dict(self) -> dict[str, parameters_module.Parameter]:
        """Get hyperparameters as a dictionary.

        :returns: Dictionary mapping names to hyperparameters
        :rtype: dict[str, parameters_module.Parameter]

        Provides convenient access to the model's hyperparameters by name
        for prior specification and sensitivity analysis.
        """
        return model_comps_to_dict(self.hyperparameters)

    @property
    def transformed_parameters(
        self,
    ) -> tuple[transformed_parameters_module.TransformedParameter, ...]:
        """Get all named transformed parameters in the model.

        :returns: Tuple of transformed parameter components
        :rtype: tuple[transformed_parameters_module.TransformedParameter, ...]

        Transformed parameters are deterministic functions of other model
        components, representing computed quantities like sums, products,
        or other mathematical transformations.
        """
        return tuple(
            filter(
                lambda x: isinstance(
                    x, transformed_parameters_module.TransformedParameter
                ),
                self.named_model_components,
            )
        )

    @property
    def transformed_parameter_dict(
        self,
    ) -> dict[str, transformed_parameters_module.TransformedParameter]:
        """Get named transformed parameters as a dictionary.

        :returns: Dictionary mapping names to transformed parameters
        :rtype: dict[str, transformed_parameters_module.TransformedParameter]

        Enables convenient access to transformed parameters for model
        inspection and derived quantity analysis.
        """
        return model_comps_to_dict(self.transformed_parameters)

    @property
    def constants(self) -> tuple[constants_module.Constant, ...]:
        """Get all named constants in the model.

        :returns: Tuple of constant components
        :rtype: tuple[constants_module.Constant, ...]

        Constants represent fixed values and hyperparameter specifications
        that do not change during inference or optimization procedures.
        """
        return tuple(
            filter(
                lambda x: isinstance(x, constants_module.Constant),
                self.named_model_components,
            )
        )

    @property
    def constant_dict(self) -> dict[str, constants_module.Constant]:
        """Get named constants as a dictionary.

        :returns: Dictionary mapping names to constant components
        :rtype: dict[str, constants_module.Constant]

        Provides convenient access to model constants for hyperparameter
        inspection and sensitivity analysis workflows.
        """
        return model_comps_to_dict(self.constants)

    @property
    def observables(self) -> tuple[parameters_module.Parameter, ...]:
        """Get all observable parameters in the model (observables are always named).

        :returns: Tuple of parameters marked as observable
        :rtype: tuple[parameters_module.Parameter, ...]

        Observable parameters represent the data-generating components
        of the model - the variables for which observed data will be
        provided during inference procedures.
        """
        return tuple(
            filter(
                lambda x: isinstance(x, parameters_module.Parameter) and x.observable,
                self.named_model_components,
            )
        )

    @property
    def observable_dict(self) -> dict[str, parameters_module.Parameter]:
        """Get observable parameters as a dictionary.

        :returns: Dictionary mapping names to observable parameters
        :rtype: dict[str, parameters_module.Parameter]

        Enables convenient access to observable parameters for data
        specification and model validation workflows.
        """
        return model_comps_to_dict(self.observables)
