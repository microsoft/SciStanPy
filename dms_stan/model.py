"""Contains the Model base class, which is used to define all DMS Stan models."""

import collections

from typing import Generator, NamedTuple, Optional, Union

import numpy.typing as npt

import dms_stan.constant as dmsc
import dms_stan.param as dmsp


class Model:
    """
    A metaclass that modifies the __init__ method of a class to register all instance
    variables that are instances of the `Parameter` class and observables in the
    `_observables` attribute and all that are not observables in the `_parameters`
    attribute.
    """

    def __init_subclass__(cls, **kwargs):
        """"""
        # Make sure an `__init__` method is defined in the class
        if "__init__" not in cls.__dict__:
            raise ValueError("`Model` subclasses must define an `__init__` method")

        # The old __init__ method of the class is renamed to '_wrapped_init'
        if "_wrapped_init" in cls.__dict__:
            raise ValueError(
                "The attribute `_wrapped_init` cannot be defined in `Model` subclasses"
            )

        # Redefine the __init__ method of the class
        def __init__(self, *init_args, **init_kwargs):

            # Run the init method that was defined in the class.
            cls._wrapped_init(self, *init_args, **init_kwargs)

            # If this is not the first class in the MRO, then we do not want to
            # review parameters and observables further, as this is duplicate effort
            if self.__class__.mro()[0] is not cls:
                return

            # Our parameters and observables are what are already in the class
            if hasattr(self, "_parameters"):
                # if not all()
                raise ValueError(
                    "The attribute `_parameters` cannot be defined `Model` subclasses"
                )
            if hasattr(self, "_observables"):
                raise ValueError(
                    "The attribute `_observables` cannot be defined in `Model` subclasses"
                )

            # Now we need to find all the parameters, observables, and constants
            # that are defined in the class. Skip any attributes defined in this
            # meta class.
            parameters, observables, constants = {}, {}, {}
            for attr in set(dir(self)) - set(dir(Model)):

                # If the attribute is a parameter, add it to the parameters dictionary.
                # If it is an observable, add it to the observables dictionary. If
                # it is a constant, add it to the constants dictionary.
                retrieved = getattr(self, attr)
                if isinstance(retrieved, dmsp.Parameter):
                    if retrieved.observable:
                        observables[attr] = retrieved
                    else:
                        parameters[attr] = retrieved
                elif isinstance(retrieved, dmsc.Constant):
                    constants[attr] = retrieved

            # At least one of these parameters must be an observable and at least
            # one of them must be a parameter.
            if len(parameters) == 0:
                raise ValueError("At least one parameter must be defined in the model")
            if len(observables) == 0:
                raise ValueError("At least one observable must be defined in the model")

            # Convert the parameters and observables to named tuples
            self._parameters = collections.namedtuple("Parameters", parameters.keys())(
                **parameters
            )
            self._observables = collections.namedtuple(
                "Observables", observables.keys()
            )(**observables)

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

    def __iter__(self) -> Generator[tuple[str, dmsp.Parameter], None, None]:
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
        sorted_param_names = sorted(
            parameters, key=lambda x: (-parameter_depths[x[0]], x[0])
        )

        # Sort the observables by name
        sorted_obs_names = sorted(observables)

        # Yield the parameters in order
        for param_name in sorted_param_names:
            yield param_name, parameters[param_name]

        # Yield the observables in order
        for obs_name in sorted_obs_names:
            yield obs_name, observables[obs_name]

    @property
    def parameters(self) -> NamedTuple:
        """Returns the parameters of the model."""
        return self._parameters  # pylint: disable=no-member

    @parameters.setter
    def parameters(self, value) -> None:
        raise ValueError("The parameters attribute cannot be set")

    @parameters.deleter
    def parameters(self) -> None:
        raise ValueError("The parameters attribute cannot be deleted")

    @property
    def parameter_dict(self) -> dict[str, dmsp.Parameter]:
        """Returns the parameters of the model as a dictionary."""
        return self._parameters._asdict()  # pylint: disable=no-member

    @property
    def observables(self) -> NamedTuple:
        """Returns the observables of the model."""
        return self._observables  # pylint: disable=no-member

    @observables.setter
    def observables(self, value) -> None:
        raise ValueError("The observables attribute cannot be set")

    @observables.deleter
    def observables(self) -> None:
        raise ValueError("The observables attribute cannot be deleted")

    @property
    def observable_dict(self) -> dict:
        """Returns the observables of the model as a dictionary."""
        return self._observables._asdict()  # pylint: disable=no-member
