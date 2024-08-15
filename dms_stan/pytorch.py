"""Holds utilities for integrating DMS Stan with Pytorch"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn

import dms_stan as dms


class TorchContainer(ABC):
    """
    Holds all the necessary information to use a `dms_stan.param.AbstractParameter`
    child class with Pytorch.
    """

    def __init__(self, bound_param: dms.param.AbstractParameter):
        """
        Args:
            bound_param: The dms_stan parameter to which this container is bound.
        """
        # Record the bound parameter
        self.bound_param = bound_param

        # Get the PyTorch parameters from the parent parameters
        self._torch_parameters: dict[str, torch.Tensor] = {}
        for param_name, param in bound_param.parameters.items():

            # Add to the dictionary. Parent parameters drawn from a distribution
            # will be defined as torch parameters. Non-parameters will be defined
            # as torch tensors.
            if isinstance(param, dms.param.AbstractParameter):
                init_vals = param.draw(1).squeeze(0)
                assert init_vals.shape == param.shape
                self._torch_parameters[param_name] = nn.Parameter(
                    torch.tensor(init_vals)
                )
            else:
                self._torch_parameters[param_name] = torch.tensor(param)

    def get_child_paramname(self, child_param: dms.param.AbstractParameter) -> str:
        """
        Gets the name of the parameter that the bound parameter defines in the child.

        Args:
            child_param (dms.param.AbstractParameter): A child of the bound parameter.

        Returns:
            str: The name of the parameter in the child that the bound parameter
                defines.
        """
        # Get the name of the parameter in the child that the bound parameter defines
        name = [
            paramname
            for paramname, param in child_param.parameters.items()
            if param is self.bound_param
        ]
        assert len(name) == 1
        return name[0]

    def get_observables(
        self, observed: Optional[torch.Tensor] = None
    ) -> list[torch.Tensor]:
        """
        Gets the values of the observables from the perspective of the bound parameter.
        These are values used to parametrize any child distributions that are generated
        from the bound parameter.

        Args:
            observed (Optional[torch.Tensor], optional): The observed value. This
                only needs to be provided for observed parameters. Latent parameters
                will automatically identify the child parameters and in turn use
                that parameter's parameters as the observed value. Defaults to None.

        Returns:
            list[torch.Tensor]: The values of the observables from the perspective of
                the bound parameter.
        """
        # Get the child parameters.
        child_params = self.bound_param.get_children()

        # If there are no child parameters, the bound parameter must be an observed
        # parameter. In this case, the observed value must be provided.
        if not child_params:
            assert (
                self.bound_param.observable
            ), "Only observed parameters should have no children."
            if observed is None:
                raise ValueError("Observed parameters must have an observed value.")

        # Gather the appropriate value from the child parameters
        vals = [observed] if observed is not None else []
        for child_param in child_params:

            # Get the name of the parameter in the child that the bound parameter
            # defines
            child_paramname = self.get_child_paramname(child_param)

            # If the child is a transformed parameter, the log-probability is calculated
            # from the inverse operation's output.
            vals.append(child_param.torch_container.parameters[child_paramname])

        return vals

    @property
    def parameters(self) -> dict[str, torch.Tensor]:
        """Gets the parameters of the bound parameter as PyTorch parameters."""
        return self._torch_parameters


class ParameterContainer(TorchContainer):
    """
    Holds all the necessary information to use a `dms_stan.param.Parameter`
    child class with Pytorch.
    """

    def __init__(self, bound_param: dms.param.Parameter):
        """
        Args:
            bound_param: The dms_stan parameter to which this container is bound.
        """
        # Assign the bound parameter to the container
        super().__init__(bound_param)

        # Set up the distribution
        self.distribution = bound_param.torch_dist(
            {
                bound_param.stan_to_torch_names[param_name]: param
                for param_name, param in self._torch_parameters.items()
            }
        )

    def calculate_log_prob(
        self, observed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculates the log probability of the parameters given the observed data.

        Args:
            observed (Optional[torch.Tensor], optional): The observed value. This
                only needs to be provided for observed parameters. Latent parameters
                will automatically identify the child parameters and in turn use
                that parameter's parameters as the observed value. Defaults to None.

        Returns:
            torch.Tensor: Log probability of the parameters given the observed data.
        """
        # Get the observables
        observables = self.get_observables(observed)
        assert len(observables) > 0

        # Calculate the log probability
        log_prob = 0
        for val in observables:
            log_prob += self.distribution.log_prob(val).sum()

        return log_prob


class TransformedContainer(TorchContainer):
    """
    Holds all the necessary information to use a `dms_stan.param.TransformedParameter`
    child class with Pytorch.
    """

    # We have a class attribute that is the list of parameters that might be missing.
    # The parameter whose value is calculated in the `calculate_missing_param` method
    # is chosen in order of the parameters in this list. The parameter chosen is
    # the first one that is not a constant
    missing_param_order: tuple[str, ...]

    def __init__(self, bound_param: dms.param.TransformedParameter):
        """
        Args:
            bound_param: The dms_stan parameter to which this container is bound.
        """
        # Initialize parameters
        super().__init__(bound_param)

        # The parameters must overlap with the missing parameters
        if set(self._torch_parameters.keys()) != set(self.missing_param_order):
            raise ValueError(
                "The torch parameters must match the potential missing parameters."
            )

        # Now choose a missing parameter. This is the degree of freedom that we
        # solve for in order to calculate the inverse-transformed parameters.
        for param in self.missing_param_order:
            if isinstance(self._torch_parameters[param], nn.Parameter):
                self.missing_param = param
                break
        else:
            raise ValueError("All parameters are constants. Cannot backpropagate.")

        # Record the shape of the missing parameter. The default hidden shape is
        # also this shape
        self.missing_shape = bound_param.parameters[self.missing_param].shape

        # Remove the missing parameter from the dictionary
        del self._torch_parameters[self.missing_param]

    @abstractmethod
    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        """
        Calculates the missing parameter from the observable. This is necessary
        for calculating the inverse-transformed parameters.

        Args:
            observable (torch.Tensor): The observable value.

        Returns:
            torch.Tensor: The missing parameter.
        """

    # The bound parameter must have inverse operations defined and mapped to the
    # parameter names in the child.
    @property
    def parameters(self) -> dict[str, torch.Tensor]:

        # Get the observables from the perspective of the bound parameter. There
        # should be exactly one observable.
        observables = self.get_observables()
        assert len(observables) == 1

        # Get a shallow copy of the parameters
        all_params = self._torch_parameters.copy()

        # Add the missing parameter back to the dictionary
        all_params[self.missing_param] = self.calculate_missing_param(observables[0])

        return all_params


class BinaryTransformedContainer(TransformedContainer):
    """
    Holds all the necessary information to use a `dms_stan.param.BinaryTransformedParameter`
    child class with PyTorch.
    """

    # Prioritize the first distribution as the missing parameter
    missing_param_order = ("dist1", "dist2")

    def __init__(self, bound_param: dms.param.BinaryTransformedParameter):
        """
        Args:
            bound_param: The dms_stan parameter to which this container is bound.
        """
        super().__init__(bound_param)

        # There should be exactly one parameter
        assert len(self._torch_parameters) == 1

    @abstractmethod
    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        """
        Calculates the first distribution from the observable. This is necessary
        for calculating the inverse-transformed parameters.

        Args:
            observable (torch.Tensor): The observable value.

        Returns:
            torch.Tensor: The first distribution.
        """

    @abstractmethod
    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        """
        Calculates the second distribution from the observable. This is necessary
        for calculating the inverse-transformed parameters.

        Args:
            observable (torch.Tensor): The observable value.

        Returns:
            torch.Tensor: The second distribution.
        """

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        if self.missing_param == "dist1":
            return self.calculate_dist1(observable)
        elif self.missing_param == "dist2":
            return self.calculate_dist2(observable)
        else:
            raise ValueError("Missing parameter not recognized.")


class UnaryTransformedContainer(TransformedContainer):
    """
    Holds all the necessary information to use a `dms_stan.param.UnaryTransformedParameter`
    child class with PyTorch.
    """

    missing_param_order = ("dist1",)  # The one and only parameter

    def __init__(self, bound_param: dms.param.UnaryTransformedParameter):
        """
        Args:
            bound_param: The dms_stan parameter to which this container is bound.
        """
        super().__init__(bound_param)

        # There should be no parameters
        assert len(self._torch_parameters) == 0


class AddTransformedContainer(BinaryTransformedContainer):
    """Pairs with the `dms_stan.param.AddParameter` class."""

    def __init__(self, bound_param: dms.param.AddParameter):
        super().__init__(bound_param)

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable - self._torch_parameters["dist2"]

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return observable - self._torch_parameters["dist1"]


class SubtractTransformedContainer(BinaryTransformedContainer):
    """Pairs with the `dms_stan.param.SubtractParameter` class."""

    def __init__(self, bound_param: dms.param.SubtractParameter):
        super().__init__(bound_param)

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable + self._torch_parameters["dist2"]

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return self._torch_parameters["dist1"] - observable


class MultiplyTransformedContainer(BinaryTransformedContainer):
    """Pairs with the `dms_stan.param.MultiplyParameter` class."""

    def __init__(self, bound_param: dms.param.MultiplyParameter):
        super().__init__(bound_param)

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable / self._torch_parameters["dist2"]

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return observable / self._torch_parameters["dist1"]


class DivideTransformedContainer(BinaryTransformedContainer):
    """Pairs with the `dms_stan.param.DivideParameter` class."""

    def __init__(self, bound_param: dms.param.DivideParameter):
        super().__init__(bound_param)

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable * self._torch_parameters["dist2"]

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return self._torch_parameters["dist1"] / observable


class PowerTransformedContainer(BinaryTransformedContainer):
    """Pairs with the `dms_stan.param.PowerParameter` class."""

    def __init__(self, bound_param: dms.param.PowerParameter):
        super().__init__(bound_param)

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable ** (1 / self._torch_parameters["dist2"])

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return torch.log(observable) / torch.log(self._torch_parameters["dist1"])


class NegateTransformedContainer(UnaryTransformedContainer):
    """Pairs with the `dms_stan.param.NegateParameter` class."""

    def __init__(self, bound_param: dms.param.NegateParameter):
        super().__init__(bound_param)

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return -observable


class AbsTransformedContainer(UnaryTransformedContainer):
    """Pairs with the `dms_stan.param.AbsParameter` class."""

    def __init__(self, bound_param: dms.param.AbsParameter):
        super().__init__(bound_param)

        # Add another parameter to the dictionary. This is a learnable parameter
        # that gives the sign of the observable.
        self._torch_parameters["latent_sign"] = nn.Parameter(
            torch.rand(self.missing_shape)
        )

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:

        # Get the sign of the observable
        sign = (self._torch_parameters["latent_sign"] > 0.5).float() * 2 - 1

        # Add the sign to the observable
        return observable * sign


class LogTransformedContainer(UnaryTransformedContainer):
    """Pairs with the `dms_stan.param.LogParameter` class."""

    def __init__(self, bound_param: dms.param.LogParameter):
        super().__init__(bound_param)

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return torch.exp(observable)


class ExpTransformedContainer(UnaryTransformedContainer):
    """Pairs with the `dms_stan.param.ExpParameter` class."""

    def __init__(self, bound_param: dms.param.ExpParameter):
        super().__init__(bound_param)

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return torch.log(observable)


class ScaledTransformedContainer(UnaryTransformedContainer):
    """Pairs with any class that scales the observable."""

    def __init__(
        self,
        bound_param: dms.param.UnaryTransformedParameter,
        init_scale: Union[str, float],
    ):
        super().__init__(bound_param)

        # If "axis" is an additional keyword argument provided to the bound parameter,
        # this is the axis whose scales we need to learn. So, this is a singleton
        # parameter in the shape.
        scale_shape = list(self.missing_shape)
        if axinds := bound_param.operation_kwargs.get("axis"):
            for axind in axinds:
                scale_shape[axind] = 1

        # Add a scale parameter to the dictionary. This is a learnable parameter
        # that scales the observable.
        if isinstance(init_scale, str):
            if init_scale == "normal":
                init_params = torch.randn(scale_shape)
            elif init_scale == "uniform":
                init_params = torch.rand(scale_shape)
            else:
                raise ValueError("Invalid initialization method.")
        else:
            init_params = torch.full(scale_shape, init_scale)
        self._torch_parameters["latent_scale"] = nn.Parameter(init_params)


class NormalizeTransformedContainer(ScaledTransformedContainer):
    """Pairs with the `dms_stan.param.NormalizeParameter` class."""

    def __init__(self, bound_param: dms.param.NormalizeParameter):
        super().__init__(bound_param, init_scale=1.0)

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return observable * self._torch_parameters["latent_scale"]


class NormalizeLogTransformedContainer(ScaledTransformedContainer):
    """Pairs with the `dms_stan.param.NormalizeLogParameter` class."""

    def __init__(self, bound_param: dms.param.NormalizeLogParameter):
        super().__init__(bound_param, init_scale=0.0)

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return observable + self._torch_parameters["latent_scale"]


class LogExponentialGrowthContainer(TransformedContainer):
    """Pairs with the `dms_stan.param.LogExponentialGrowth` class."""

    missing_param_order = ("log_A", "r", "t")

    def __init__(self, bound_param: dms.param.LogExponentialGrowth):
        super().__init__(bound_param)

    def calculate_logA(  # pylint: disable=invalid-name
        self, observable: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the log of the amplitude from the observable."""
        return observable - self._torch_parameters["r"] * self._torch_parameters["t"]

    def calculate_r(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the growth rate from the observable."""
        return (observable - self._torch_parameters["log_A"]) / self._torch_parameters[
            "t"
        ]

    def calculate_t(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the time from the observable."""
        return (observable - self._torch_parameters["log_A"]) / self._torch_parameters[
            "r"
        ]

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        if self.missing_param == "log_A":
            return self.calculate_logA(observable)
        elif self.missing_param == "r":
            return self.calculate_r(observable)
        elif self.missing_param == "t":
            return self.calculate_t(observable)
        else:
            raise ValueError("Missing parameter not recognized.")


class LogSigmoidGrowthContainer(TransformedContainer):
    """Pairs with the `dms_stan.param.LogSigmoidGrowth` class."""

    missing_param_order = ("log_A", "r", "c", "t")

    def __init__(self, bound_param: dms.param.LogSigmoidGrowth):
        super().__init__(bound_param)

    def calculate_logA(  # pylint: disable=invalid-name
        self, observable: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the log of the amplitude from the observable."""
        return observable + torch.log(
            1
            + torch.exp(
                self._torch_parameters["r"]
                * (self._torch_parameters["c"] - self._torch_parameters["t"])
            )
        )

    def calculate_r(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the growth rate from the observable."""
        return torch.log(
            torch.exp(self._torch_parameters["log_A"] - observable) - 1
        ) / (self._torch_parameters["c"] - self._torch_parameters["t"])

    def calculate_c(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the inflection point from the observable."""
        return (
            torch.log(torch.exp(self._torch_parameters["log_A"] - observable) - 1)
            / self._torch_parameters["r"]
        ) + self._torch_parameters["t"]

    def calculate_t(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the time from the observable."""
        return self._torch_parameters["c"] - (
            torch.log(torch.exp(self._torch_parameters["log_A"] - observable) - 1)
            / self._torch_parameters["r"]
        )


# TODO: Figure out which parameters need to be kept in log space during optimization
# to force positivity.
