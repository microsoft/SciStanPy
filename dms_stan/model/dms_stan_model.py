"""Contains the Model base class, which is used to define all DMS Stan models."""

from abc import ABC, abstractmethod
from typing import Literal, Optional, overload, TypedDict, Union

import hvplot.interactive
import numpy as np
import numpy.typing as npt
import torch

import dms_stan as dms

from .components import (
    Binomial,
    Constant,
    LogExponentialGrowth,
    LogSigmoidGrowth,
    Normal,
)
from .components.abstract_classes import AbstractModelComponent
from .components.custom_types import CombinableParameterType
from .components.parameters import Parameter
from .components.pytorch import check_observable_data, PyTorchModel
from .components.stan import StanModel
from dms_stan.defaults import DEFAULT_EARLY_STOP, DEFAULT_LR, DEFAULT_N_EPOCHS


# Special type for the MAP estimate
class MAPDict(TypedDict):
    """Type for `Model.approximate_map` method return value."""

    MAP: dict[str, npt.NDArray]
    distributions: dict[str, torch.distributions.Distribution]
    losses: npt.NDArray


class Model(ABC):
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

            # If we already have parameters, defined in the class, update them with
            # the new parameters. This situation occurs when a child class is defined
            # that inherits from a parent class that is also a model.
            parameters = (
                self.parameter_dict.copy() if hasattr(self, "_parameters") else {}
            )

            # Now we need to find all the parameters that are defined in the class.
            for attr in set(dir(self)) - set(dir(Model)):
                if isinstance(retrieved := getattr(self, attr), AbstractModelComponent):
                    retrieved.model_varname = attr  # Set the model variable name
                    assert attr not in parameters
                    parameters[attr] = retrieved

            # Set the parameters attribute
            self._parameters = parameters

            # Make sure there is at least one observable
            if len(self.observables) == 0:
                raise ValueError("At least one observable must be defined.")

            # Build the mapping between model variable names and parameter objects
            self._model_varname_to_object = self._build_model_varname_to_object()

        # Add the new __init__ method
        cls._wrapped_init = cls.__init__
        cls.__init__ = __init__

    def _build_model_varname_to_object(self) -> dict[str, AbstractModelComponent]:
        """
        Builds a mapping between model varnames and objects for easy access. Also
        makes sure that all Parameter instances are explicitly defined.
        """
        # Start from each observable and walk up the tree to the root
        unnamed_params: list[str] = []
        model_varname_to_object: dict[str, AbstractModelComponent] = {}
        for observable in self.observables:

            # Add the observable to the mapping
            assert observable.model_varname not in model_varname_to_object
            model_varname_to_object[observable.model_varname] = observable

            # Add all parents to the mapping and make sure `Parameter` instances
            # are explicitly defined.
            for _, parent in observable.walk_tree(walk_down=False):

                # If the parent is already in the mapping, make sure it is the
                # same
                if parent.model_varname in model_varname_to_object:
                    assert model_varname_to_object[parent.model_varname] == parent
                else:
                    model_varname_to_object[parent.model_varname] = parent

                # If the parent is a `Parameter` instance and it is not named, add
                # it to the list of unnamed parameters.
                if isinstance(parent, Parameter) and not parent.is_named:
                    unnamed_params.append(parent.model_varname)

        # Make sure all parameters are explicitly defined
        if unnamed_params:
            raise ValueError(
                "All `Parameter` instances must be explicitly defined. The following "
                f"are not: {', '.join(unnamed_params)}"
            )

        # There can be no duplicate values in the mapping
        assert len(model_varname_to_object) == len(
            set(model_varname_to_object.values())
        )

        return model_varname_to_object

    @overload
    def draw(self, n: int, named_only: Literal[True]) -> dict[str, npt.NDArray]: ...

    @overload
    def draw(
        self, n: int, named_only: Literal[False]
    ) -> dict[AbstractModelComponent, npt.NDArray]: ...

    def draw(self, n, named_only=True):
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

            # Draw from the observable
            _, full_draw = observable.draw(n)

            # Update the draws. We only take draws from named parameters.
            for model_component, sample in full_draw.items():
                if model_component in draws:
                    assert np.array_equal(draws[model_component], sample)
                else:
                    draws[model_component] = sample

        # Filter down to just named parameters if requested
        if named_only:
            return {k: v for k, v in draws.items() if k.is_named}
        else:
            return draws

    def to_pytorch(self):
        """
        Compiles the model to a trainable PyTorch model.
        """
        return PyTorchModel(self)

    def to_stan(self):
        """
        Compiles the model to a Stan model.
        """
        return StanModel(self)

    def approximate_map(
        self,
        epochs: int = DEFAULT_N_EPOCHS,
        early_stop: int = DEFAULT_EARLY_STOP,
        lr: float = DEFAULT_LR,
        **observed_data: Union[torch.Tensor, npt.NDArray],
    ) -> MAPDict:
        """
        Approximate the maximum a posteriori (MAP) estimate of the model parameters.
        Under the hood, this fits a PyTorch model to the data that minimizes the
        sum of `log_pdf` and `log_pmf` for all distributions. The parameter values
        that minimize this loss are then returned.
        """
        # Check observed data
        check_observable_data(self, observed_data)

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
        pp = dms.model.PriorPredictiveCheck(self, copy_model=copy_model)

        # Return the plot
        return pp.display(
            initial_view=initial_view,
            independent_dim=independent_dim,
            independent_labels=independent_labels,
        )

    def __contains__(self, paramname: str) -> bool:
        """Checks if the model contains a parameter or observable with the given name."""
        return paramname in self._model_varname_to_object

    def __getitem__(self, paramname: str) -> AbstractModelComponent:
        """Returns the parameter or observable with the given name."""
        return self._model_varname_to_object[paramname]

    @property
    def parameters(self) -> tuple[AbstractModelComponent, ...]:
        """Returns the parameters of the model."""
        return tuple(
            sorted(self._parameters.values(), key=lambda x: x.model_varname)
        )  # pylint: disable=no-member

    @property
    def parameter_dict(self) -> dict[str, AbstractModelComponent]:
        """Returns the parameters of the model as a dictionary."""
        return self._parameters.copy()  # pylint: disable=no-member

    @property
    def constants(self) -> tuple[Constant, ...]:
        """
        Returns the hyperparameters of the model. These are explicitly defined
        constants and constants implicit to the model based on parameter definitions.
        """
        return tuple(sorted(self.constants_dict, key=lambda x: x.model_varname))

    @property
    def constants_dict(self) -> dict[str, Constant]:
        """
        Returns the hyperparameters of the model. These are explicitly defined
        constants and constants implicit to the model based on parameter definitions.
        """
        constants_dict: dict[str, Constant] = {}
        for param in self.parameters:

            # If the parameter is a constant, add it to the dictionary
            if isinstance(param, Constant):
                constants_dict[param.model_varname] = param

            # Loop over all parents to the parameter. If any of the parents are
            # constants, add them to the dictionary as well.
            for parent in param.parents:
                if isinstance(parent, Constant):
                    if parent.model_varname in constants_dict:
                        assert constants_dict[parent.model_varname] == parent
                    else:
                        constants_dict[parent.model_varname] = parent

        return constants_dict

    @property
    def observables(self) -> Parameter:
        """Returns the observables of the model."""
        return tuple(param for param in self.parameters if param.observable)

    @property
    def observable_dict(self) -> dict[str, Parameter]:
        """Returns the observables of the model as a dictionary."""
        return {k: v for k, v in self.parameter_dict.items() if v.observable}


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
        self.log_theta = dms.operations.normalize_log(
            self.log_theta_unorm, shape=self.log_theta_unorm_mean.shape, axis=-1
        )

        # Transform the log thetas to thetas
        self.theta = dms.operations.exp(self.log_theta)

    @abstractmethod
    def _define_growth_curve(
        self, t: npt.NDArray[np.floating], counts: npt.NDArray[np.integer]
    ) -> AbstractModelComponent:
        """Define the growth curve of the model."""


class ExponentialGrowthModel(BaseGrowthModel):
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


class SigmoidGrowthModel(BaseGrowthModel):
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


class BinomialGrowthModelMixin:
    """
    Helper for defining a growth model of count data over time where the counts
    are modeled as binomially distributed.
    """

    def __init__(
        self,
        *,
        t: npt.NDArray[np.floating],
        counts: npt.NDArray[np.integer],
        sigma: CombinableParameterType,
        **kwargs,
    ):

        # Call the parent class constructor
        super().__init__(t=t, counts=counts, sigma=sigma, **kwargs)

        # Set up the binomial distribution for the counts. "N" is inferred as the
        # sum of the counts at each timepoint.
        self.counts = Binomial(
            theta=self.theta,
            N=counts.sum(axis=-1, keepdims=True),
            shape=counts.shape,
        ).as_observable()


class ExponentialGrowthBinomialModel(ExponentialGrowthModel, BinomialGrowthModelMixin):
    """Defines a model of count data over time with exponential growth."""
