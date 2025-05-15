"""Contains the Model base class, which is used to define all DMS Stan models."""

import os.path
import pickle

from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal, Optional, overload, Union

import numpy as np
import numpy.typing as npt
import panel as pn
import torch
import xarray as xr

import dms_stan as dms

from dms_stan.custom_types import CombinableParameterType
from dms_stan.defaults import (
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

from .components import (
    Constant,
    LogExponentialGrowth,
    LogSigmoidGrowth,
    Multinomial,
    Normal,
    Parameter,
    TransformedData,
    TransformedParameter,
)
from .components.abstract_model_component import AbstractModelComponent
from .pytorch.map import MAP
from .pytorch import check_observable_data, PyTorchModel
from .stan import SampleResults, StanModel


def components_to_dict(
    components: Iterable[AbstractModelComponent],
) -> dict[str, AbstractModelComponent]:
    """
    Converts a list of components to a dictionary where the keys are the model variable
    names of the components.
    """
    return {comp.model_varname: comp for comp in components}


def run_delayed_mcmc(filepath: str) -> SampleResults:
    """
    Runs a delayed MCMC run created by calling `Model.mcmc()` with `delay_run=True`.
    The filepath should be the path to the pickled object that resulted from this
    function call.
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


class Model(ABC):
    """
    A metaclass that modifies the __init__ method of a class to register all instance
    variables that are instances of the `Parameter` class and observables in the
    `_observables` attribute and all that are not observables in the `_parameters`
    attribute.
    """

    @abstractmethod
    def __init__(
        self,
        *args,
        default_parallelize: Optional[bool] = None,
        default_data: dict[str, npt.NDArray] | None = None,
        **kwargs,
    ):
        """This should be overridden by the subclass."""
        # Set the default values for the model
        self._default_parallelize: Optional[bool] = default_parallelize
        self._default_data: dict[str, npt.NDArray] | None = default_data
        self._named_model_components: tuple[AbstractModelComponent, ...] = getattr(
            self, "_named_model_components", ()
        )
        self._model_varname_to_object: dict[str, AbstractModelComponent] = getattr(
            self, "_model_varname_to_object", {}
        )
        self._init_complete: bool = getattr(self, "_init_complete", False)

    def __init_subclass__(cls, **kwargs):
        """"""
        # The old __init__ method of the class is renamed to '_wrapped_init'
        if "_wrapped_init" in cls.__dict__:
            raise ValueError(
                "The attribute `_wrapped_init` cannot be defined in `Model` subclasses"
            )

        # Redefine the __init__ method of the class
        def __init__(
            self: "Model",
            *init_args,
            parallelize: Optional[bool] = None,
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
                    retrieved := getattr(self, attr), AbstractModelComponent
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
            self._model_varname_to_object = self._build_model_varname_to_object(
                parallelize=parallelize
            )

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
        self, parallelize: bool | None
    ) -> dict[str, AbstractModelComponent]:
        """Builds a mapping between model varnames and objects for easy access."""

        def build_initial_mapping() -> dict[str, AbstractModelComponent]:
            """Builds an initial mapping of model varnames to objects."""

            # Start from each observable and walk up the tree to the root
            model_varname_to_object: dict[str, AbstractModelComponent] = {}
            for observable in self.observables:

                # Add the observable to the mapping
                assert observable.model_varname not in model_varname_to_object
                model_varname_to_object[observable.model_varname] = observable

                # Add all parents to the mapping and make sure `Parameter` instances
                # are explicitly defined.
                for *_, parent in observable.walk_tree(walk_down=False):

                    # If the parent is already in the mapping, make sure it is the
                    # same
                    if parent.model_varname in model_varname_to_object:
                        assert model_varname_to_object[parent.model_varname] == parent
                    else:
                        model_varname_to_object[parent.model_varname] = parent

            return model_varname_to_object

        def update_parallelization(parallelize_local: bool | None) -> None:
            """
            For all components identified in `build_initial_mapping`, set the
            parallelization attribute. This is done in `_build_model_varname_to_object`
            because the act of setting parallelization may add new child components
            to some of the components. These child components handle transformed
            data.
            """
            # Handle parallelization. This might add some child components that handle
            # transformed data
            if (
                parallelize_local is None
                and getattr(self, "_default_parallelize", None) is not None
            ):
                parallelize_local = self._default_parallelize
            if parallelize_local is not None:
                for component in model_varname_to_object.values():
                    if isinstance(component, TransformedData):
                        continue
                    component._parallelized = (  # pylint: disable=protected-access
                        parallelize_local
                    )

        def record_transformed_data() -> None:
            """Updates the mapping with all transformed data components."""

            # Add all TransformedData instances to the mapping
            for component in list(model_varname_to_object.values()):
                for child in component._children:  # pylint: disable=protected-access
                    if isinstance(child, TransformedData):
                        assert child.model_varname not in model_varname_to_object
                        model_varname_to_object[child.model_varname] = child

        # Run the steps
        model_varname_to_object = build_initial_mapping()
        update_parallelization(parallelize)
        record_transformed_data()

        # There can be no duplicate values in the mapping
        assert len(model_varname_to_object) == len(
            set(model_varname_to_object.values())
        )

        return model_varname_to_object

    def get_dimname_map(self) -> dict[tuple[int, int], str]:
        """
        Retrieves a dictionary that maps from the level and size of a dimension
        to the name of that dimension. This is used to build xarray datasets.
        """

        # Set up variables
        dims: dict[tuple[int, int], str] = {}

        # The list of dimension options cannot overlap with variable names
        allowed_dim_names = [
            name
            for name in DEFAULT_DIM_NAMES
            if name not in self.named_model_components_dict
        ]

        # Check sizes of all observables and record the dimension names
        for observable in self.observables:
            for dimkey in enumerate(observable.shape[::-1]):
                if dimkey not in dims:
                    dims[dimkey] = allowed_dim_names[len(dims)]

        return dims

    def _compress_for_xarray(
        self,
        *arrays: npt.NDArray,
        include_sample_dim: bool = False,
    ) -> list[tuple[tuple[str, ...], npt.NDArray]]:
        """
        Retrieves a tuple of dimension names for the given shapes. This is used to
        build xarray datasets.
        """
        # Get a mapping from dimension keys to dimension names
        dims = self.get_dimname_map()

        # Set our start and end indices for the shape
        start_ind = int(include_sample_dim)

        # Process each input array
        results: list[tuple[tuple[str, ...], npt.NDArray]] = []
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
            results.append(
                (tuple(dimnames[::-1]), np.squeeze(array, axis=tuple(singleton_axes)))
            )

        return results

    def _dict_to_xarray(
        self, draws: dict[AbstractModelComponent, npt.NDArray]
    ) -> xr.Dataset:
        """
        Builds the kwargs for the xarray dataset. The data values are the
        draws and the coordinates are inputs to transformed parameters that
        are constants and that have a shape.
        """
        # Split into components and draws and components and values
        components, unpacked_draws = zip(
            *[
                [comp, draw]
                for comp, draw in draws.items()
                if not isinstance(comp, Constant)
            ]
        )
        coordinates = list(
            zip(
                *[
                    [parent, parent.value]
                    for component in self.all_model_components
                    for parent in component.parents
                    if isinstance(parent, Constant) and np.prod(parent.shape) > 1
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
                for component, compressed_draw in zip(components, compressed_draws)
            },
            coords={
                parent.model_varname: compressed_value
                for parent, compressed_value in zip(parents, compressed_values)
            },
        )

    @overload
    def draw(
        self,
        n: int,
        *,
        named_only: Literal[True],
        as_xarray: Literal[False],
        seed: Optional[int],
    ) -> dict[str, npt.NDArray]: ...

    @overload
    def draw(
        self,
        n: int,
        *,
        named_only: Literal[False],
        as_xarray: Literal[False],
        seed: Optional[int],
    ) -> dict[AbstractModelComponent, npt.NDArray]: ...

    @overload
    def draw(
        self,
        n: int,
        *,
        named_only: Literal[True],
        as_xarray: Literal[True],
        seed: Optional[int],
    ) -> xr.Dataset: ...

    @overload
    def draw(
        self,
        n: int,
        *,
        named_only: Literal[False],
        as_xarray: Literal[True],
        seed: Optional[int],
    ) -> xr.Dataset: ...

    def draw(self, n, *, named_only=True, as_xarray=False, seed=None):
        """Draws from the model. By default, this will draw from the observable
        values of the model. If a parameter is specified, then it will draw from
        the distribution of that parameter.

        Args:
            n (int): The number of samples to draw.

        Returns:
            dict[str, npt.NDArray]: A dictionary where the keys are the names of
                the model components and the values are the samples drawn.
        """
        # Draw from all observables
        draws: dict[AbstractModelComponent, npt.NDArray] = {}
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

    def to_pytorch(self, seed: Optional[int] = None) -> PyTorchModel:
        """
        Compiles the model to a trainable PyTorch model.
        """
        return PyTorchModel(self, seed=seed)

    def to_stan(self, **kwargs) -> StanModel:
        """
        Compiles the model to a Stan model.
        """
        return StanModel(self, **kwargs)

    def approximate_map(
        self,
        epochs: int = DEFAULT_N_EPOCHS,
        early_stop: int = DEFAULT_EARLY_STOP,
        lr: float = DEFAULT_LR,
        data: Optional[dict[str, Union[torch.Tensor, npt.NDArray]]] = None,
        device: int | str = "cpu",
        seed: Optional[int] = None,
    ) -> MAP:
        """
        Approximate the maximum a posteriori (MAP) estimate of the model parameters.
        Under the hood, this fits a PyTorch model to the data that minimizes the
        sum of `log_pdf` and `log_pmf` for all distributions. The parameter values
        that minimize this loss are then returned.
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
        check_observable_data(self, data)

        # Fit the model
        pytorch_model = self.to_pytorch(seed=seed).to(device=device)
        loss_trajectory = pytorch_model.fit(
            epochs=epochs,
            early_stop=early_stop,
            lr=lr,
            data=data,
        )

        # Get the MAP estimate for all model parameters
        map_ = {
            k: v.detach().cpu().numpy()
            for k, v in pytorch_model.export_params().items()
        }

        # Get the distributions of the parameters
        distributions = pytorch_model.export_distributions()

        # Return the MAP estimate, the distributions, and the loss trajectory
        return MAP(
            model=self,
            map_estimate=map_,
            distributions=distributions,
            losses=loss_trajectory.detach().cpu().numpy(),
            data={k: v.detach().cpu().numpy() for k, v in data.items()},
        )

    def _get_simulation_data(self, seed: Optional[int]) -> dict[str, npt.NDArray]:
        """Draws observable data from the model prior."""
        data = self.draw(1, named_only=True, as_xarray=False, seed=seed)
        return {
            observable.model_varname: data[observable.model_varname][0]
            for observable in self.observables
        }

    def simulate_map_approximation(
        self, **kwargs
    ) -> tuple[dict[str, npt.NDArray], MAP]:
        """
        Samples data from the model prior, then fits a PyTorch model to this sampled
        data. This is useful for debugging, mainly for checking that any peculiarities
        observed during MAP approximation are not due to the data.
        Args:
            **kwargs: Keyword arguments to pass to the `approximate_map` method,
                excepting `data`.

        Returns:
            dict[str, npt.NDArray]: A dictionary where the keys are the names of
                the observable parameters and the values are the samples drawn.
            MAP: The MAP object resulting from the fit to the simulated data.
        """
        # TODO: This and the other simulate method should include non-observables
        # in the returned MAP as well.
        # Get the data
        kwargs["data"] = self._get_simulation_data(seed=kwargs.get("seed"))

        # Fit the model
        return kwargs["data"], self.approximate_map(**kwargs)

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
    ) -> SampleResults: ...
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
        inits="prior",
        data=None,
        delay_run=False,
        **sample_kwargs,
    ):
        """Samples from the model using MCMC. This is a wrapper around the `sample`
        method of the `StanModel` class.
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
        stan_model = self.to_stan(
            output_dir=output_dir,
            force_compile=force_compile,
            stanc_options=stanc_options,
            cpp_options=cpp_options,
            user_header=user_header,
            model_name=model_name,
        )

        # Update the output directory in the sample kwargs
        sample_kwargs["output_dir"] = os.path.abspath(stan_model.output_dir)

        # If delaying, then we save the data needed for sampling and return
        if delay_run:
            with open(
                (
                    delay_run
                    if isinstance(delay_run, str)
                    else f"{stan_model.stan_executable_path}-delay.pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(
                    {
                        "stan_model": stan_model,
                        "sample_kwargs": sample_kwargs,
                        "data": data,
                        "inits": inits,
                    },
                    f,
                )
            return

        # Sample from the model
        return stan_model.sample(inits=inits, data=data, **sample_kwargs)

    @overload
    def simulate_mcmc(
        self, delay_run: Literal[False], **kwargs
    ) -> tuple[dict[str, npt.NDArray], SampleResults]: ...
    @overload
    def simulate_mcmc(self, delay_run: Literal[True], **kwargs) -> None: ...

    def simulate_mcmc(self, delay_run=False, **kwargs):
        """
        Simulates data from the model and then runs MCMC on that data. This is useful
        for debugging, mainly for checking that any peculiarities observed during
        MCMC are not due to the data.
        """
        # Update the model name
        if kwargs.get("model_name") == DEFAULT_MODEL_NAME:
            kwargs["model_name"] = f"{DEFAULT_MODEL_NAME}-simulated"

        # Get the data
        kwargs["data"] = self._get_simulation_data(seed=kwargs.get("seed"))

        # Run MCMC
        return kwargs["data"], self.mcmc(delay_run=delay_run, **kwargs)

    def prior_predictive(self, *, copy_model: bool = False) -> pn.Row:
        """
        Creates an interactive plot of the prior predictive distribution of the
        model. The plot can be used to update the model's parameters dynamically.
        See `dms_stan.prior_predictive.PriorPredictiveCheck` for more details.
        """
        # Create the prior predictive object
        pp = dms.model.PriorPredictiveCheck(self, copy_model=copy_model)

        # Return the plot
        return pp.display()

    def __str__(self) -> str:
        """Returns a string representation of the model."""
        # Get all model components
        components = {
            "Constants": [
                el for el in self.all_model_components if isinstance(el, Constant)
            ],
            "Transformed Parameters": self.transformed_parameters,
            "Parameters": self.parameters,
            "Observables": self.observables,
        }

        # Combine representations from all model components
        return "\n\n".join(
            key + "\n" + "=" * len(key) + "\n" + "\n".join(str(el) for el in complist)
            for key, complist in components.items()
            if len(complist) > 0
        )

    def __contains__(self, paramname: str) -> bool:
        """Checks if the model contains a parameter or observable with the given name."""
        return paramname in self._model_varname_to_object

    def __getitem__(self, paramname: str) -> AbstractModelComponent:
        """Returns the parameter or observable with the given name."""
        return self._model_varname_to_object[paramname]

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets the attribute of the model. We cannot set attributes that are model
        components, as this will break the model
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
        """Returns the default data for the model. Errors if it is not set."""
        if getattr(self, "_default_data", None) is None:
            raise ValueError(
                "Default data is not set. Please set the default data using "
                "`model.default_data = data`."
            )
        return self._default_data

    @default_data.setter
    def default_data(self, data: dict[str, npt.NDArray] | None) -> None:
        """
        Sets the default data for the model. The dictionary must contain data for
        all observables in the model.
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
        """Returns True if the model has default data."""
        return getattr(self, "_default_data", None) is not None

    @property
    def named_model_components(self) -> tuple[AbstractModelComponent, ...]:
        """Returns the named model components sorted by the model variable name."""
        return self._named_model_components

    @property
    def named_model_components_dict(self) -> dict[str, AbstractModelComponent]:
        """Returns the named model components as a dictionary."""
        return components_to_dict(self.named_model_components)

    @property
    def all_model_components(self) -> tuple[AbstractModelComponent, ...]:
        """Returns all model components sorted by the model variable name."""
        return tuple(
            sorted(
                self._model_varname_to_object.values(), key=lambda x: x.model_varname
            )
        )

    @property
    def all_model_components_dict(self) -> dict[str, AbstractModelComponent]:
        """Returns all model components as a dictionary."""
        return self._model_varname_to_object

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Returns the parameters of the model."""
        return tuple(
            filter(
                lambda x: isinstance(x, Parameter) and not x.observable,
                self.all_model_components,
            )
        )

    @property
    def parameter_dict(self) -> dict[str, Parameter]:
        """Returns the parameters of the model as a dictionary."""
        return components_to_dict(self.parameters)

    @property
    def hyperparameters(self) -> tuple[Parameter, ...]:
        """
        Returns the hyperparameters of the model. These are `Parameter` instances
        whose parents are constants.
        """
        return tuple(filter(lambda x: x.is_hyperparameter, self.parameters))

    @property
    def hyperparameter_dict(self) -> dict[str, Parameter]:
        """
        Returns the hyperparameters of the model as a dictionary. These are `Parameter`
        instances whose parents are constants.
        """
        return components_to_dict(self.hyperparameters)

    @property
    def transformed_parameters(self) -> tuple[TransformedParameter, ...]:
        """Returns the transformed parameters of the model."""
        return tuple(
            filter(
                lambda x: isinstance(x, TransformedParameter),
                self.named_model_components,
            )
        )

    @property
    def transformed_parameter_dict(self) -> dict[str, TransformedParameter]:
        """Returns the transformed parameters of the model as a dictionary."""
        return components_to_dict(self.transformed_parameters)

    @property
    def constants(self) -> tuple[Constant, ...]:
        """Returns named constants of the model"""
        return tuple(
            filter(lambda x: isinstance(x, Constant), self.named_model_components)
        )

    @property
    def constant_dict(self) -> dict[str, Constant]:
        """
        Returns the hyperparameters of the model. These are explicitly defined
        constants and constants implicit to the model based on parameter definitions.
        """
        return components_to_dict(self.constants)

    @property
    def observables(self) -> tuple[Parameter, ...]:
        """Returns the observables of the model."""
        return tuple(
            filter(
                lambda x: isinstance(x, Parameter) and x.observable,
                self.named_model_components,
            )
        )

    @property
    def observable_dict(self) -> dict[str, Parameter]:
        """Returns the observables of the model as a dictionary."""
        return components_to_dict(self.observables)


class BaseGrowthModel(Model):
    """Defines a model of count data over time."""

    def __init__(  # pylint: disable=super-init-not-called, unused-argument
        self,
        *,
        t: npt.NDArray[np.floating],
        counts: npt.NDArray[np.integer],
        sigma: CombinableParameterType,
        **kwargs,
    ):
        # Assign the noise parameter
        self.sigma = sigma

        # Define the growth curve
        self.log_theta_unorm_mean = self._define_growth_curve(t=t, counts=counts)

        # Define the regression distribution
        self.log_theta_unorm = Normal(
            mu=self.log_theta_unorm_mean,
            sigma=self.sigma,
            shape=self.log_theta_unorm_mean.shape,
        )

        # We normalize the thetas to add to 1
        self.log_theta = dms.operations.normalize_log(self.log_theta_unorm)

        # Transform the log thetas to thetas
        self.theta = dms.operations.exp(self.log_theta)

        # Counts are "Multinomial" distributed as the base
        self.counts = Multinomial(
            theta=self.theta,
            N=counts.sum(axis=-1, keepdims=True),
            shape=counts.shape,
        ).as_observable()

    @abstractmethod
    def _define_growth_curve(
        self, t: npt.NDArray[np.floating], counts: npt.NDArray[np.integer]
    ) -> AbstractModelComponent:
        """Define the growth curve of the model."""


class ExponentialGrowthBinomialModel(BaseGrowthModel):
    """Mix in class for exponential growth."""

    def __init__(
        self,
        *,
        t: npt.NDArray[np.floating],
        counts: npt.NDArray[np.integer],
        log_A: CombinableParameterType,
        r: CombinableParameterType,
        sigma: CombinableParameterType,
        **kwargs,
    ):

        # Assign the growth parameters
        self.log_A = log_A  # pylint: disable=invalid-name
        self.r = r

        # Call the parent class constructor. This will set up the remaining parameters
        super().__init__(t=t, counts=counts, sigma=sigma, **kwargs)

    def _define_growth_curve(
        self, t: npt.NDArray[np.floating], counts: npt.NDArray[np.integer]
    ) -> AbstractModelComponent:
        return LogExponentialGrowth(t=t, log_A=self.log_A, r=self.r, shape=counts.shape)


class SigmoidGrowthBinomialModel(BaseGrowthModel):
    """Mix in class for sigmoid growth."""

    def __init__(
        self,
        *,
        t: npt.NDArray[np.floating],
        counts: npt.NDArray[np.integer],
        log_A: CombinableParameterType,
        r: CombinableParameterType,
        c: CombinableParameterType,
        sigma: CombinableParameterType,
        **kwargs,
    ):

        # Assign the growth parameters
        self.log_A = log_A  # pylint: disable=invalid-name
        self.r = r
        self.c = c

        # Call the parent class constructor. This will set up the timepoints as a
        # constant but do nothing with the counts except check their shape.
        super().__init__(t=t, counts=counts, sigma=sigma, **kwargs)

    def _define_growth_curve(
        self, t: npt.NDArray[np.floating], counts: npt.NDArray[np.integer]
    ) -> AbstractModelComponent:
        return LogSigmoidGrowth(
            t=t, log_A=self.log_A, r=self.r, c=self.c, shape=counts.shape
        )
