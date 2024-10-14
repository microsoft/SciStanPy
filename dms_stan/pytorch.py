"""Holds utilities for integrating DMS Stan with Pytorch"""

import warnings

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy.typing as npt
import torch
import torch.nn as nn

from tqdm import tqdm

import dms_stan as dms


def check_observable_data(
    model: "dms.model.Model", observed_data: dict[str, Union[torch.Tensor, npt.NDArray]]
):
    """Makes sure that the correct observables are provided for a givne model."""
    # There must be perfect overlap between the keys of the provided data and the
    # expected observations
    expected_set = set(model.observable_dict.keys())
    provided_set = set(observed_data.keys())
    missing = expected_set - provided_set
    extra = provided_set - expected_set

    # If there are missing or extra, raise an error
    if missing:
        raise ValueError(
            "The provided data must match the observable distribution names."
            f"The following observables are missing: {', '.join(missing)}"
        )
    if extra:
        raise ValueError(
            "The provided data must match the observable distribution names. The "
            "following observables were provided in addition to the expected: "
            f"{', '.join(extra)}"
        )

    # Shapes must match
    for name, param in model.observable_dict.items():
        if observed_data[name].shape != param.shape:
            raise ValueError(
                f"The shape of the provided data for observable {name} does not match "
                f"the expected shape. Expected: {param.shape}, provided: "
                f"{observed_data[name].shape}"
            )


class TorchContainer(ABC):
    """
    Holds all the necessary information to use a `dms_stan.param.AbstractParameter`
    child class with Pytorch.
    """

    def __init__(
        self, bound_param: "dms_components.abstract_classes.AbstractParameter"
    ):
        """
        Args:
            bound_param: The dms_stan parameter to which this container is bound.
        """
        # Record the bound parameter
        self.bound_param = bound_param

        # Set for parameters that are shared
        self._shared_params: set[str] = set()

        # Get the PyTorch parameters from the parent parameters
        self._torch_parameters: dict[str, torch.Tensor] = {}
        self._learnable_parameters: set[str] = set()
        self._parent_to_paramname: dict[
            dms_components.abstract_classes.AbstractParameter, str
        ] = {}
        for param_name, param in bound_param.parameters.items():

            # Add to the dictionary. Parent parameters drawn from a distribution
            # will be defined as torch parameters. Non-parameters will be defined
            # as torch tensors.
            if isinstance(param, dms_components.abstract_classes.AbstractParameter):
                self._learnable_parameters.add(param_name)
                init_vals = self._inverse_transform_parameter(
                    param_name, torch.tensor(param.draw(1).squeeze(0))
                )
                assert init_vals.shape == param.shape
                self._torch_parameters[param_name] = nn.Parameter(init_vals)
                self._parent_to_paramname[param] = param_name

            else:
                self._torch_parameters[param_name] = torch.tensor(param)

    def get_child_paramname(
        self, child_param: "dms_components.abstract_classes.AbstractParameter"
    ) -> str:
        """
        Gets the name of the parameter that the bound parameter defines in the child.

        Args:
            child_param (dms_components.abstract_classes.AbstractParameter): A child of the bound parameter.

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

        # In the case of multiple observables, they must be the same object
        if len(vals) > 1:
            assert all(val is vals[0] for val in vals[1:])

        return vals

    def _transform_parameter(self, param_name: str) -> torch.Tensor:
        """Converts the raw parameters to the appropriate space."""

        # Pull the parameter
        param = self._torch_parameters[param_name]

        # If the parameter is to be transformed, apply the appropriate transformation
        # We only need to transform the parameters that are learnable
        if (
            param_name in self._learnable_parameters
            and param_name in self.transformed_parameters
        ):
            transformed_param = torch.exp(param)
            if param_name in self.bound_param.NEGATIVE_PARAMS:
                transformed_param *= -1
            if param_name in self.bound_param.SIMPLEX_PARAMS:
                transformed_param = transformed_param / transformed_param.sum()
            return transformed_param

        # Otherwise, just return the parameter
        return param

    def _inverse_transform_parameter(
        self, param_name: str, param: torch.Tensor
    ) -> torch.Tensor:
        """Performs the inverse transformation on the parameter."""
        # If the parameter is to be transformed, apply the appropriate transformation
        # We only need to transform the parameters that are learnable
        if (
            param_name in self._learnable_parameters
            and param_name in self.transformed_parameters
        ):

            # Negative parameters will be defined as such
            if param_name in self.bound_param.NEGATIVE_PARAMS:
                if not torch.all(param < 0):
                    raise ValueError("Negative parameter must be negative.")
                param *= -1

            # A simplex parameter will be normalized
            if param_name in self.bound_param.SIMPLEX_PARAMS:
                param = param / param.sum()

            # Then to log space
            return torch.log(param)

        # Otherwise, just return the parameter
        return param

    def _transform_parameters(self) -> dict[str, torch.Tensor]:
        """
        Takes the self._torch_parameters dictionary and transforms it to the appropriate
        space. Essentially, this means that the parameters that are bounded to be
        positive, negative, or simplexes, are defined in log space to ensure that
        they remain in the appropriate range. This function will exponentiate the
        raw parameters, apply the appropriate sign, and, if necessary, normalize
        them to sum to 1.
        """
        return {
            param_name: self._transform_parameter(param_name)
            for param_name in self._torch_parameters
        }

    def _link_torch_parameters(
        self,
        parent: "dms_components.abstract_classes.AbstractParameter",
        sibling: "TorchContainer",
    ):
        """
        Replaces the torch parameters of this instance with the shared torch parameters
        of the sibling instance. A sibling instance is another parameter whose
        prior is the same parent parameter as this instance.
        """
        # pylint: disable=protected-access
        # Get the name of the parameter in this instance and the sibling instance
        param_name = self._parent_to_paramname[parent]
        sibling_param_name = sibling._parent_to_paramname[parent]
        assert param_name in self._torch_parameters
        assert sibling_param_name in sibling._torch_parameters

        # Replace the parameter
        self._torch_parameters[param_name] = sibling._torch_parameters[
            sibling_param_name
        ]

        # Note that this parameter is shared
        self._shared_params.add(param_name)

    @property
    def parameters(self) -> dict[str, torch.Tensor]:
        """Gets the parameters of the bound parameter as PyTorch parameters."""
        return self._transform_parameters()

    @property
    def transformed_parameters(self) -> set[str]:
        """
        Gets the identities of the parameters that are stored in a different space
        in PyTorch than in the bound parameter.
        """
        return (
            self.bound_param.POSITIVE_PARAMS
            | self.bound_param.NEGATIVE_PARAMS
            | self.bound_param.SIMPLEX_PARAMS
        )


class ParameterContainer(TorchContainer):
    """
    Holds all the necessary information to use a `dms_stan.param.Parameter`
    child class with Pytorch.
    """

    def __init__(self, bound_param: "dms_components.parameters.Parameter"):
        """
        Args:
            bound_param: The dms_stan parameter to which this container is bound.
        """
        # Assign the bound parameter to the container
        super().__init__(bound_param)

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

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """
        Gets the distribution of the bound parameter. The transformed parameters
        are used to define the distribution, not the raw torch parameters.
        """
        return self.bound_param.torch_dist(
            **{
                self.bound_param.stan_to_torch_names[param_name]: param
                for param_name, param in self.parameters.items()
            }
        )


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

    def __init__(self, bound_param: "dms_components.parameters.TransformedParameter"):
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

        # Store a shallow copy of the torch parameters. We need this for setting
        # changing the missing parameter later if necessary.
        self._copied_torch_parameters = self._torch_parameters.copy()

        # Choose a missing parameter.
        self._set_missing_param()

    def _set_missing_param(self):
        """
        Selects the missing parameter from the list of potential missing parameters.
        This is the degree of freedom that we solve for in order to calculate the
        inverse-transformed parameters.
        """
        # Set the missing parameter
        for param in self.missing_param_order:

            # The parameter cannot be shared
            if param in self._shared_params:
                continue

            # Take the first allowable parameter
            if isinstance(self._torch_parameters[param], nn.Parameter):
                self.missing_param = param
                break
        else:
            raise ValueError(
                "All parameters are constants and/or shared already. Cannot backpropagate."
            )

        # Record the shape of the missing parameter. The default hidden shape is
        # also this shape
        self.missing_shape = self.bound_param.parameters[self.missing_param].shape

        # Remove the missing parameter from the dictionary
        self._torch_parameters = self._copied_torch_parameters.copy()
        del self._torch_parameters[self.missing_param]

    def _link_torch_parameters(
        self,
        parent: "dms_components.abstract_classes.AbstractParameter",
        sibling: TorchContainer,
    ):

        # Run the parent method
        super()._link_torch_parameters(parent, sibling)

        # If the missing parameter is shared, we need to reset it
        if self.missing_param in self._shared_params:
            self._set_missing_param()

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

    def _transform_parameters(self) -> dict[str, torch.Tensor]:

        # Transform the parameters
        transformed_params = super()._transform_parameters()

        # Get the observables from the perspective of the bound parameter. There
        # should be exactly one observable.
        observables = self.get_observables()
        assert len(observables) == 1

        # Add the missing parameter back to the dictionary
        transformed_params[self.missing_param] = self.calculate_missing_param(
            observables[0]
        )

        return transformed_params


class BinaryTransformedContainer(TransformedContainer):
    """
    Holds all the necessary information to use a `dms_stan.param.BinaryTransformedParameter`
    child class with PyTorch.
    """

    # Prioritize the first distribution as the missing parameter
    missing_param_order = ("dist1", "dist2")

    def __init__(
        self,
        bound_param: "dms_components.transformed_parameters.BinaryTransformedParameter",
    ):
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

    def __init__(
        self,
        bound_param: "dms_components.transformed_parameters.UnaryTransformedParameter",
    ):
        """
        Args:
            bound_param: The dms_stan parameter to which this container is bound.
        """
        super().__init__(bound_param)

        # There should be no parameters
        assert len(self._torch_parameters) == 0


class AddTransformedContainer(BinaryTransformedContainer):
    """Pairs with the `dms_stan.param.AddParameter` class."""

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.AddParameter"
    ):
        super().__init__(bound_param)

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable - self._transform_parameter("dist2")

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return observable - self._transform_parameter("dist1")


class SubtractTransformedContainer(BinaryTransformedContainer):
    """Pairs with the `dms_stan.param.SubtractParameter` class."""

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.SubtractParameter"
    ):
        super().__init__(bound_param)

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable + self._transform_parameter("dist2")

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return self._transform_parameter("dist1") - observable


class MultiplyTransformedContainer(BinaryTransformedContainer):
    """Pairs with the `dms_stan.param.MultiplyParameter` class."""

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.MultiplyParameter"
    ):
        super().__init__(bound_param)

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable / self._transform_parameter("dist2")

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return observable / self._transform_parameter("dist1")


class DivideTransformedContainer(BinaryTransformedContainer):
    """Pairs with the `dms_stan.param.DivideParameter` class."""

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.DivideParameter"
    ):
        super().__init__(bound_param)

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable * self._transform_parameter("dist2")

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return self._transform_parameter("dist1") / observable


class PowerTransformedContainer(BinaryTransformedContainer):
    """Pairs with the `dms_stan.param.PowerParameter` class."""

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.PowerParameter"
    ):
        super().__init__(bound_param)

    def calculate_dist1(self, observable: torch.Tensor) -> torch.Tensor:
        return observable ** (1 / self._transform_parameter("dist2"))

    def calculate_dist2(self, observable: torch.Tensor) -> torch.Tensor:
        return torch.log(observable) / torch.log(self._transform_parameter("dist1"))


class NegateTransformedContainer(UnaryTransformedContainer):
    """Pairs with the `dms_stan.param.NegateParameter` class."""

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.NegateParameter"
    ):
        super().__init__(bound_param)

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return -observable


class AbsTransformedContainer(UnaryTransformedContainer):
    """Pairs with the `dms_stan.param.AbsParameter` class."""

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.AbsParameter"
    ):
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

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.LogParameter"
    ):
        super().__init__(bound_param)

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return torch.exp(observable)


class ExpTransformedContainer(UnaryTransformedContainer):
    """Pairs with the `dms_stan.param.ExpParameter` class."""

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.ExpParameter"
    ):
        super().__init__(bound_param)

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return torch.log(observable)


class ScaledTransformedContainer(UnaryTransformedContainer):
    """Pairs with any class that scales the observable."""

    def __init__(
        self,
        bound_param: "dms_components.transformed_parameters.UnaryTransformedParameter",
        init_scale: Union[str, float],
    ):
        super().__init__(bound_param)

        # If "axis" is an additional keyword argument provided to the bound parameter,
        # this is the axis whose scales we need to learn. So, this is a singleton
        # parameter in the shape.
        scale_shape = list(self.missing_shape)
        if axinds := bound_param.operation_kwargs.get("axis"):
            axinds = [axinds] if isinstance(axinds, int) else axinds
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

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.NormalizeParameter"
    ):
        super().__init__(bound_param, init_scale=1.0)

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return observable * self._torch_parameters["latent_scale"]


class NormalizeLogTransformedContainer(ScaledTransformedContainer):
    """Pairs with the `dms_stan.param.NormalizeLogParameter` class."""

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.NormalizeLogParameter"
    ):
        super().__init__(bound_param, init_scale=0.0)

    def calculate_missing_param(self, observable: torch.Tensor) -> torch.Tensor:
        return observable + self._torch_parameters["latent_scale"]


class LogExponentialGrowthContainer(TransformedContainer):
    """Pairs with the `dms_stan.param.LogExponentialGrowth` class."""

    missing_param_order = ("log_A", "r", "t")

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.LogExponentialGrowth"
    ):
        super().__init__(bound_param)

    def calculate_logA(  # pylint: disable=invalid-name
        self, observable: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the log of the amplitude from the observable."""
        return observable - self._transform_parameter("r") * self._transform_parameter(
            "t"
        )

    def calculate_r(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the growth rate from the observable."""
        # Calling `self.parameters` multiple times is inefficient. We call the
        # function wrapped by that property once and store the result.
        return (
            observable - self._transform_parameter("log_A")
        ) / self._transform_parameter("t")

    def calculate_t(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the time from the observable."""
        # Calling `self.parameters` multiple times is inefficient. We call the
        # function wrapped by that property once and store the result.
        return (
            observable - self._transform_parameter(["log_A"])
        ) / self._transform_parameter("r")

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

    def __init__(
        self, bound_param: "dms_components.transformed_parameters.LogSigmoidGrowth"
    ):
        super().__init__(bound_param)

    def calculate_logA(  # pylint: disable=invalid-name
        self, observable: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the log of the amplitude from the observable."""
        # Calling `self.parameters` multiple times is inefficient. We call the
        # function wrapped by that property once and store the result.
        return observable + torch.log(
            1
            + torch.exp(
                self._transform_parameter("r")
                * (self._transform_parameter("c") - self._transform_parameter("t"))
            )
        )

    def calculate_r(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the growth rate from the observable."""
        # Calling `self.parameters` multiple times is inefficient. We call the
        # function wrapped by that property once and store the result.
        return torch.log(
            torch.exp(self._transform_parameter("log_A") - observable) - 1
        ) / (self._transform_parameter("c") - self._transform_parameter("t"))

    def calculate_c(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the inflection point from the observable."""
        # Calling `self.parameters` multiple times is inefficient. We call the
        # function wrapped by that property once and store the result.
        return (
            torch.log(torch.exp(self._transform_parameter("log_A") - observable) - 1)
            / self._transform_parameter("r")
        ) + self._transform_parameter("t")

    def calculate_t(self, observable: torch.Tensor) -> torch.Tensor:
        """Calculates the time from the observable."""
        # Calling `self.parameters` multiple times is inefficient. We call the
        # function wrapped by that property once and store the result.
        return self._transform_parameter("c") - (
            torch.log(torch.exp(self._transform_parameter("log_A") - observable) - 1)
            / self._transform_parameter("r")
        )


class PyTorchModel(nn.Module):
    """
    A PyTorch-trainable version of a `dms_stan.model.Model`. This should not be
    used directly, but instead accessed by calling the `to_pytorch` method on a
    `dms_stan.model.Model` instance.
    """

    def __init__(self, model: "dms.model.Model"):
        """
        Args:
            model: The `dms_stan.model.Model` instance to convert to PyTorch.
        """
        super().__init__()

        # Define types for variables where it is unclear
        observable: dms_components.parameters.Parameter
        encountered_params: dict[
            dms_components.parameters.Parameter, dms_components.parameters.Parameter
        ] = {}
        self._observable_loss_calculators: dict[str, ParameterContainer] = {}
        self._loss_calculators: list[ParameterContainer] = []
        torch_params: list[nn.Parameter] = []

        # Record the model
        self.model = model

        # Starting from each observable, walk up the tree and initialize the PyTorch
        # containers, recording Parameter instances that are used to calculate the
        # loss.
        for obsname, observable in model.observable_dict.items():

            # Initialize and record the observable
            observable.init_pytorch()
            self._observable_loss_calculators[obsname] = observable.torch_container

            # Record the PyTorch parameters
            torch_params.extend(observable.torch_container._torch_parameters.values())

            # Process all parents
            for _, parent, child in observable.recurse_parents():

                # If we have already encountered this parent, we link the appropriate
                # parameters between the shared children and continue
                if shared_child := encountered_params.get(parent, None):
                    child.torch_container._link_torch_parameters(
                        parent, shared_child.torch_container
                    )
                    continue

                # Note that the parent has been encountered
                encountered_params[parent] = child

                # Initialize the PyTorch container
                parent.init_pytorch()

                # If the parent is a Parameter, record it as a loss calculator
                if isinstance(parent, dms_components.parameters.Parameter):
                    self._loss_calculators.append(parent.torch_container)

                # Record the torch parameters
                torch_params.extend(parent.torch_container._torch_parameters.values())

        # Record the learnable parameters as a parameter list. This is necessary
        # for PyTorch to recognize the parameters.
        self.torch_params = nn.ParameterList(
            [param for param in torch_params if isinstance(param, nn.Parameter)]
        )

    def forward(
        self, **observed_data: Union[npt.NDArray, torch.Tensor, float, int]
    ) -> torch.Tensor:
        """
        Each observation is passed in as a keyword argument whose name matches the
        name of the corresponding observable distribution in the `dms_stan.model.Model`
        instance. This will calculate the log probability of the observed data given
        the parameters of the model. Stress: this is the log-probability, not the
        log loss, which is the negative log-probability.
        """
        # Check the observed data
        check_observable_data(self.model, observed_data)

        # Any observed data that is not a tensor is converted to a tensor
        observed_data = {
            k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            for k, v in observed_data.items()
        }

        # Sum the log-probs of the observables
        log_prob = 0.0
        for name, container in self._observable_loss_calculators.items():
            log_prob += container.calculate_log_prob(observed_data[name])

        # Sum the log-probs of the parameters
        for container in self._loss_calculators:
            log_prob += container.calculate_log_prob()

        return log_prob

    def fit(
        self,
        epochs: int = dms.defaults.DEFAULT_N_EPOCHS,
        early_stop: int = dms.defaults.DEFAULT_EARLY_STOP,
        lr: float = dms.defaults.DEFAULT_LR,
        **observed_data: Union[torch.Tensor, npt.NDArray, float, int],
    ) -> torch.Tensor:
        """Optimizes the parameters of the model."""
        # Train mode. This should be a null-op, but it is set to future-proof in
        # case this ever changes
        self.train()

        # Build the optimizer
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        # Set up for optimization
        best_loss = float("inf")  # Records the best loss
        loss_trajectory = [None] * (epochs + 1)  # Records all losses
        n_without_improvement = 0  # Epochs without improvement

        # Run optimization
        with tqdm(total=epochs, desc="Epochs", postfix={"loss": "N/A"}) as pbar:
            for epoch in range(epochs):

                # Get the loss
                log_loss = -1 * self(**observed_data)

                # Step the optimizer
                optim.zero_grad()
                log_loss.backward()
                optim.step()

                # Record loss
                log_loss = log_loss.item()
                loss_trajectory[epoch] = log_loss

                # Update best loss
                if log_loss < best_loss:
                    n_without_improvement = 0
                    best_loss = log_loss
                else:
                    n_without_improvement += 1

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"loss": f"{log_loss:.2f}"})

                # Check for early stopping
                if early_stop > 0 and n_without_improvement >= early_stop:
                    break

            # Note that early stopping was not triggered if the loop completes
            else:
                if early_stop > 0:
                    warnings.warn("Early stopping not triggered.")

        # Back to eval mode
        self.eval()

        # Get a final loss
        with torch.no_grad():
            loss_trajectory[epoch + 1] = -1 * self(**observed_data).item()

        # Trim off the None values of the loss trajectory and convert to a tensor
        return torch.tensor(loss_trajectory[: epoch + 2], dtype=torch.float32)

    def export_params(self) -> dict[str, torch.Tensor]:
        """
        Exports all parameters from the DMS Stan model used to construct the PyTorch
        model. This is primarily used once the model has been fit.
        """
        return {
            name: param.torch_container.get_observables()[0]
            for name, param in self.model.parameter_dict.items()
        }

    def export_distributions(self) -> dict[str, torch.distributions.Distribution]:
        """
        Exports all distributions from the DMS Stan model used to construct the PyTorch
        model. This is primarily used once the model has been fit.
        """
        return {
            name: param.torch_container.distribution
            for name, param in self.model.parameter_dict.items()
            if isinstance(param, dms_components.parameters.Parameter)
        } | {
            name: param.torch_container.distribution
            for name, param in self.model.observable_dict.items()
            if isinstance(param, dms_components.parameters.Parameter)
        }
