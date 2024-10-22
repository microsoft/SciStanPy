"""Contains the Model base class, which is used to define all DMS Stan models."""

import collections

from typing import Generator, NamedTuple, Optional, TypedDict, Union

import hvplot.interactive
import numpy as np
import numpy.typing as npt
import torch

import dms_stan as dms
from .components.abstract_classes import AbstractModelComponent, AbstractParameter
from .components.constants import Constant, Hyperparameter
from .components.custom_types import CombinableParameterType
from .components.parameters import Binomial, Normal
from .components.transformed_parameters import LogExponentialGrowth, LogSigmoidGrowth


# Special type for the MAP estimate
class MAPDict(TypedDict):
    """Type for `Model.approximate_map` method return value."""

    MAP: dict[str, npt.NDArray]
    distributions: dict[str, torch.distributions.Distribution]
    losses: npt.NDArray


class Model:
    """
    A metaclass that modifies the __init__ method of a class to register all instance
    variables that are instances of the `Parameter` class and observables in the
    `_observables` attribute and all that are not observables in the `_parameters`
    attribute.
    """

    def __init_subclass__(cls, **kwargs):
        """"""
        # The old __init__ method of the class is renamed to '_wrapped_init'
        if "_wrapped_init" in cls.__dict__:
            raise ValueError(
                "The attribute `_wrapped_init` cannot be defined in `Model` subclasses"
            )

        # Redefine the __init__ method of the class
        def __init__(self, *init_args, **init_kwargs):

            # Run the init method that was defined in the class.
            cls._wrapped_init(self, *init_args, **init_kwargs)

            # If we already have parameters, observables, and constants defined in
            # the class, update them with the new parameters, observables, and constants.
            # This situation occurs when a child class is defined that inherits
            # from a parent class that is also a model.
            if hasattr(self, "_parameters"):
                assert hasattr(self, "_observables")
                assert hasattr(self, "_constants")
                assert hasattr(self, "_hyperparameters")
                parameters = self.parameter_dict
                observables = self.observable_dict
                constants = self.constant_dict
                hyperparameters = self.hyperparameter_dict
            else:
                assert not hasattr(self, "_observables")
                assert not hasattr(self, "_constants")
                assert not hasattr(self, "_hyperparameters")
                parameters, observables, constants, hyperparameters = {}, {}, {}, {}

            # Now we need to find all the parameters, observables, hyperparameters,
            # and constants that are defined in the class. Skip any attributes defined
            # in this meta class.
            for attr in set(dir(self)) - set(dir(Model)):

                # Get the attribute's value
                retrieved = getattr(self, attr)

                # Continue if not a model component
                if not isinstance(retrieved, AbstractModelComponent):
                    continue

                # Bin the model component appropriately
                retrieved.model_varname = attr
                if isinstance(retrieved, AbstractParameter):

                    # Parameters are either observables or not
                    if retrieved.observable:
                        observables[attr] = retrieved
                    else:
                        parameters[attr] = retrieved

                    # Check to see if there are any hyperparameters
                    hyperparameters.update(
                        {
                            f"{attr}.{name}": param
                            for name, param in retrieved.hyperparameters.items()
                        }
                    )

                elif isinstance(retrieved, Constant):
                    constants[attr] = retrieved

            # Convert the parameters, observables, hyperparameters, and constants
            # to named tuples
            self._parameters = collections.namedtuple("Parameters", parameters.keys())(
                **parameters
            )
            self._hyperparameters = hyperparameters
            self._observables = collections.namedtuple(
                "Observables", observables.keys()
            )(**observables)
            self._constants = collections.namedtuple("Constants", constants.keys())(
                **constants
            )

        # Add the new __init__ method
        cls._wrapped_init = cls.__init__
        cls.__init__ = __init__

    def get_parameter_depths(self) -> dict[str, int]:
        """
        Get the depths of all parameters in the model. This will output a dictionary
        that maps from a parameter name to the depth of that parameter in the model.
        """
        # Get an inverted dictionary of parameters
        param_to_name = {param: name for name, param in self.parameter_dict.items()}

        # Starting from observables, walk up the tree of parameters
        parameter_depths = {}
        for observable in self.observables:

            # Get the parents for this observable and use them to update the parameter
            # depths
            for depth, parent, _ in observable.recurse_parents():

                # If the parent is in the model, record its depth. If the parameter is
                # already in the dictionary, we want to take the maximum depth.
                if name := param_to_name.get(parent):
                    parameter_depths[name] = max(parameter_depths.get(name, 0), depth)

        return parameter_depths

    def draw(
        self, size: Union[int, tuple[int, ...]], param: Optional[str] = None
    ) -> dict[str, npt.NDArray]:
        """Draws from the model. By default, this will draw from the observable
        values of the model. If a parameter is specified, then it will draw from
        the distribution of that parameter.

        Args:
            size (Union[int, tuple[int, ...]]): The size of the sample to draw.
                This is treated as the size parameter for the underlying numpy
                random number generator.
            param (Optional[str], optional): The parameter whose values should be
                returned. Defaults to None, meaning that observable values will
                be returned.

        Returns:
            dict[str, npt.NDArray]: A dictionary where the keys are the names of
                the observables or parameters and the values are the samples drawn.
        """
        # If a parameter is specified, draw from that parameter
        if param is not None:
            return {param: self.parameter_dict[param].draw(size)}

        # Otherwise, return draws from the observables
        return {name: obs.draw(size) for name, obs in self.observable_dict.items()}

    def draw_from(self, paramname: str, size: int) -> npt.NDArray:
        """Draw from a parameter."""
        return getattr(self, paramname).draw(size)

    def to_pytorch(self):
        """
        Compiles the model to a trainable PyTorch model.
        """
        return dms.pytorch.PyTorchModel(self)

    def to_stan(self):
        """
        Compiles the model to a Stan model.
        """
        return dms.stan.StanModel(self)

    def approximate_map(
        self,
        epochs: int = dms.defaults.DEFAULT_N_EPOCHS,
        early_stop: int = dms.defaults.DEFAULT_EARLY_STOP,
        lr: float = dms.defaults.DEFAULT_LR,
        **observed_data: Union[torch.Tensor, npt.NDArray],
    ) -> MAPDict:
        """
        Approximate the maximum a posteriori (MAP) estimate of the model parameters.
        Under the hood, this fits a PyTorch model to the data that minimizes the
        sum of `log_pdf` and `log_pmf` for all distributions. The parameter values
        that minimize this loss are then returned.
        """
        # Check observed data
        dms.pytorch.check_observable_data(self, observed_data)

        # Fit the model
        pytorch_model = self.to_pytorch()
        loss_trajectory = pytorch_model.fit(
            epochs=epochs,
            early_stop=early_stop,
            lr=lr,
            **observed_data,
        )

        # Get the MAP estimate for all model parameters
        map_ = {
            k: v.detach().cpu().numpy()
            for k, v in pytorch_model.export_params().items()
        }

        # Get the distributions of the parameters
        distributions = pytorch_model.export_distributions()

        # Return the MAP estimate, the distributions, and the loss trajectory
        return {
            "MAP": map_,
            "distributions": distributions,
            "losses": loss_trajectory.detach().cpu().numpy(),
        }

    def prior_predictive(
        self,
        *,
        copy_model: bool = False,
        initial_view: Optional[str] = None,
        independent_dim: Optional[int] = None,
        independent_labels: Optional[npt.NDArray] = None,
    ) -> hvplot.interactive.Interactive:
        """
        Creates an interactive plot of the prior predictive distribution of the
        model. The plot can be used to update the model's parameters dynamically.
        See `dms_stan.prior_predictive.PriorPredictiveCheck` for more details.
        """
        # Create the prior predictive object
        pp = dms.prior_predictive.PriorPredictiveCheck(self, copy_model=copy_model)

        # Return the plot
        return pp.display(
            initial_view=initial_view,
            independent_dim=independent_dim,
            independent_labels=independent_labels,
        )

    def __iter__(
        self,
    ) -> Generator[tuple[str, AbstractParameter], None, None]:
        """
        Loops over the parameters and observables in the model. Parameters are
        emitted first in order of depth from deepest to shallowest. Ties in depth
        are broken by alphabetical order of the parameter name. Observables are
        emitted last in alphabetical order.
        """
        # Get dictionaries of the parameters and observables
        parameters, observables = self.parameter_dict, self.observable_dict

        # Get the depths of all parameters
        parameter_depths = self.get_parameter_depths()

        # Sort all the parameters by depth and then by name
        sorted_param_names = sorted(parameters, key=lambda x: (-parameter_depths[x], x))

        # Sort the observables by name
        sorted_obs_names = sorted(observables)

        # Yield the parameters in order
        for param_name in sorted_param_names:
            yield param_name, parameters[param_name]

        # Yield the observables in order
        for obs_name in sorted_obs_names:
            yield obs_name, observables[obs_name]

    def __contains__(self, paramname: str) -> bool:
        """Checks if the model contains a parameter or observable with the given name."""
        return paramname in self.parameter_dict or paramname in self.observable_dict

    def __getitem__(self, paramname: str) -> AbstractParameter:
        """Returns the parameter or observable with the given name."""
        return getattr(self, paramname)

    @property
    def parameters(self) -> NamedTuple:
        """Returns the parameters of the model."""
        return self._parameters  # pylint: disable=no-member

    @property
    def parameter_dict(self) -> dict[str, AbstractParameter]:
        """Returns the parameters of the model as a dictionary."""
        return self._parameters._asdict()  # pylint: disable=no-member

    @property
    def hyperparameter_dict(self) -> dict[str, Hyperparameter]:
        """Returns the hyperparameters of the model as a dictionary."""
        return self._hyperparameters

    @property
    def observables(self) -> NamedTuple:
        """Returns the observables of the model."""
        return self._observables  # pylint: disable=no-member

    @property
    def observable_dict(self) -> dict[str, AbstractParameter]:
        """Returns the observables of the model as a dictionary."""
        return self._observables._asdict()  # pylint: disable=no-member

    @property
    def constants(self) -> NamedTuple:
        """Returns the constants of the model."""
        return self._constants  # pylint: disable=no-member

    @property
    def constant_dict(self) -> dict[str, Constant]:
        """Returns the constants of the model as a dictionary."""
        return self._constants._asdict()  # pylint: disable=no-member

    @property
    def root_nodes(self) -> dict[str, AbstractParameter]:
        """Returns the parameters that are root nodes in the model."""
        return {
            name: param
            for name, param in self.parameter_dict.items()
            if param.is_root_node
        }


class BaseGrowthModel(Model):
    """Defines a model of count data over time."""

    def __init__(  # pylint: disable=super-init-not-called, unused-argument
        self, *, t: npt.NDArray[np.floating], counts: npt.NDArray[np.integer], **kwargs
    ):

        # Time should be 1D
        if t.ndim != 1:
            raise ValueError("`t` should be a 1D array")

        # Counts should be 3D. The first dimension is replicates, the second is
        # timepoints, and the third is the counts.
        if counts.ndim != 3:
            raise ValueError("`counts` should be a 3D array")

        # The second counts dimension should be the same as the length of the time
        # array.
        if counts.shape[1] != len(t):
            raise ValueError(
                "Mismatch between the length of `t` and the second dimension of `counts`"
            )

        # Record the timepoints as a constant. Expand the timepoints to the same
        # dimensionality as the counts.
        self.t = Constant(t[None, :, None])

    def _finalize_regressor(self, sigma: CombinableParameterType):

        # pylint: disable = no-member, attribute-defined-outside-init
        # Assign the noise parameter
        self.sigma = sigma

        # Define the regression distribution
        self.log_theta_unorm = Normal(
            mu=self.log_theta_unorm_mean,
            sigma=self.sigma,
            shape=self.log_theta_unorm_mean.shape,
        )

        # We normalize the thetas to add to 1
        self.log_theta = dms.operations.normalize_log(
            self.log_theta_unorm, shape=self.log_theta_unorm_mean.shape, axis=-1
        )

        # Transform the log thetas to thetas
        self.theta = dms.operations.exp(self.log_theta)


class ExponentialGrowthMixIn(BaseGrowthModel):
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
        # Call the parent class constructor. This will set up the timepoints as a
        # constant but do nothing with the counts except check their shape.
        super().__init__(t=t, counts=counts, **kwargs)

        # Assign the growth parameters
        self.log_A = log_A  # pylint: disable=invalid-name
        self.r = r

        # Get the log theta values
        self.log_theta_unorm_mean = LogExponentialGrowth(
            t=self.t, log_A=self.log_A, r=self.r, shape=counts.shape
        )

        # Build regressor params
        self._finalize_regressor(sigma=sigma)


class SigmoidGrowthMixIn(BaseGrowthModel):
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
        # Call the parent class constructor. This will set up the timepoints as a
        # constant but do nothing with the counts except check their shape.
        super().__init__(t=t, counts=counts, **kwargs)

        # Assign the growth parameters
        self.log_A = log_A  # pylint: disable=invalid-name
        self.r = r
        self.c = c

        # Get the log theta values
        self.log_theta_unorm_mean = LogSigmoidGrowth(
            t=self.t, log_A=self.log_A, r=self.r, c=self.c, shape=counts.shape
        )

        # Build regressor params
        self._finalize_regressor(sigma=sigma)


class BaseBinomialGrowthModel(BaseGrowthModel):
    """
    Defines a growth model of count data over time where the counts are modeled
    as binomially distributed.
    """

    def __init__(
        self, *, t: npt.NDArray[np.floating], counts: npt.NDArray[np.integer], **kwargs
    ):

        # Call the parent class constructor
        super().__init__(t=t, counts=counts, **kwargs)

        # Set up the binomial distribution for the counts. "N" is inferred as the
        # sum of the counts at each timepoint.
        self.counts = Binomial(
            theta=self.theta,  # pylint: disable=no-member
            N=counts.sum(axis=2, keepdims=True),
            shape=counts.shape,
        ).as_observable()


class ExponentialGrowthBinomialModel(BaseBinomialGrowthModel, ExponentialGrowthMixIn):
    """Defines a model of count data over time with exponential growth."""
