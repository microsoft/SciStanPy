"""Holds the abstract classes for the core components of a DMS Stan model."""

from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

import dms_stan as dms
import dms_stan.model.components as dms_components


class AbstractModelComponent(ABC):
    """Base class for all core components of a DMS Stan model."""

    # Define allowed ranges for the parameters
    POSITIVE_PARAMS: set[str] = set()
    NEGATIVE_PARAMS: set[str] = set()
    SIMPLEX_PARAMS: set[str] = set()

    # Define the stan data type
    BASE_STAN_DTYPE: Literal["real", "int", "simplex"] = "real"
    STAN_LOWER_BOUND: Optional[float | int] = None
    STAN_UPPER_BOUND: Optional[float | int] = None

    def __init__(  # pylint: disable=unused-argument
        self,
        *,
        shape: tuple[int, ...] = (),
        **parameters: "dms.custom_types.CombinableParameterType",
    ):
        """Builds a parameter instance with the given shape."""

        # Define placeholder variables
        self._model_varname: str = ""  # DMS Stan Model variable name
        self.observable: bool = False  # Whether the parameter is observable
        self._parents: dict[str, AbstractModelComponent]
        self._component_to_paramname: dict[AbstractModelComponent, str]
        self._shape: tuple[int, ...] = shape  # Shape of the parameter
        self._draw_shape: tuple[int, ...]  # Shape of the draws
        self._children: list[AbstractModelComponent] = []  # Children of the component
        self._torch_parameters: dict[str, torch.Tensor]  # Pytorch parameters
        self._shared_parameters: set[str] = set()  # Parents shared with siblings

        # Validate incoming parameters
        self._validate_parameters(parameters)

        # Set parents
        self._set_parents(parameters)

        # Link parent and child parameters
        for val in self._parents.values():
            val._record_child(self)

        # Set the draw shape
        self._set_draw_shape()

    def _validate_parameters(
        self,
        parameters: dict[str, "dms.custom_types.CombinableParameterType"],
    ) -> None:
        """Checks inputs to the __init__ method for validity."""
        # No incoming parameters can be observables
        if any(
            isinstance(param, AbstractModelComponent) and param.observable
            for param in parameters.values()
        ):
            raise ValueError("Parent parameters cannot be observables")

        # All bounded parameters must be named in the parameter dictionary
        if missing_names := (
            self.POSITIVE_PARAMS | self.NEGATIVE_PARAMS | self.SIMPLEX_PARAMS
        ) - set(parameters.keys()):
            raise ValueError(
                f"{', '.join(missing_names)} are bounded parameters but are missing "
                "from those defined"
            )

    def _set_parents(
        self,
        parameters: dict[str, "dms.custom_types.CombinableParameterType"],
    ) -> None:
        """Sets the parent parameters of the current parameter."""

        # Convert any non-model components to model components, making sure to
        # propagate any restrictions on
        self._parents = {}
        for name, val in parameters.items():

            # Just the value if an AbstractModelComponent
            if isinstance(val, AbstractModelComponent):
                self._parents[name] = val
                continue

            # Otherwise, convert to a constant model component with the appropriate
            # bounds
            if name in self.POSITIVE_PARAMS:
                stan_lower_bound = 0
                stan_upper_bound = None
            elif name in self.NEGATIVE_PARAMS:
                stan_lower_bound = None
                stan_upper_bound = 0
            elif name in self.SIMPLEX_PARAMS:
                stan_lower_bound = 0
                stan_upper_bound = 1
            else:
                stan_lower_bound = None
                stan_upper_bound = None
            self._parents[name] = dms_components.Constant(
                value=val,
                stan_lower_bound=stan_lower_bound,
                stan_upper_bound=stan_upper_bound,
            )

        # Map components to param names
        self._component_to_paramname = {v: k for k, v in self._parents.items()}
        assert len(self._component_to_paramname) == len(self._parents)

    def _record_child(self, child: "AbstractModelComponent") -> None:
        """
        Records a child parameter of the current parameter. This is used to keep
        track of the lineage of the parameter.

        Args:
            child (AbstractModelComponent): The child parameter to record.
        """

        # If the child is already in the list of children, then we don't need to
        # add it again
        assert child not in self._children, "Child already recorded"

        # Record the child
        self._children.append(child)

    def _set_draw_shape(self) -> None:
        """Sets the shape of the draws for the parameter."""

        # The shape must be broadcastable to the shapes of the parameters.
        try:
            self._draw_shape = np.broadcast_shapes(
                self._shape, *[param.shape for param in self.parents]
            )
        except ValueError as error:
            raise ValueError("Shape is not broadcastable to parent shapes") from error

    def get_child_paramnames(self) -> dict["AbstractModelComponent", str]:
        """
        Gets the names of the parameters that this component defines in its children

        Returns:
            dict[AbstractModelComponent, str]: The child objects mapped to the names
            of their parameters defiend by this object.
        """
        # Set up the dictionary to return
        child_paramnames = {}

        # Process all children
        for child in self._children:

            # Make sure the child is not already recorded
            assert child not in child_paramnames

            # Get the name of the parameter in the child that the bound parameter
            # defines. There should only be one parameter in the child that the
            # bound parameter defines.
            names = [
                paramname
                for paramname, param in child._parents.items()  # pylint: disable=protected-access
                if param is self
            ]
            assert len(names) == 1

            # Record the name
            child_paramnames[child] = names[0]

        return child_paramnames

    def get_indexed_varname(
        self, index_opts: tuple[str, ...], _name_override: str = ""
    ) -> str:
        """
        Returns the variable name used by stan with the appropriate indices. Note
        that DMS Stan always assumes that computations on the last dimension can
        be vectorized.

        Args:
            index_opts (tuple[str, ...]): The index options for the variable name.

        Returns:
            str: The variable name with the appropriate indices (if any).
            bool: Whether the variable is a vector.
        """
        # If the name override is provided, then we use that name
        base_name = _name_override or self.model_varname

        # If there are no indices, then we just return the variable name
        if self.ndim == 0:
            return base_name

        # Singleton dimensions get a "1" index. All others get the index options.
        indices = [
            "1" if dimsize == 1 else index_opts[i]
            for i, dimsize in enumerate(self.shape)
        ]

        # If the last dimension is vectorized, then we don't need to index it. We
        # assume that the last dimension is always vectorized.
        if indices[-1] != "1":
            indices = indices[:-1]

        # If there are no indices, then we just return the variable name
        if len(indices) == 0:
            return base_name

        # Build the indexed variable name
        indexed_varname = f"{base_name}[{','.join(indices)}]"

        return indexed_varname

    def init_pytorch(
        self,
        draws: Optional[dict["AbstractModelComponent", npt.NDArray]] = None,
    ) -> None:
        """
        Sets up the parameters needed for training a Pytorch model and defines the
        Pytorch operation that will be performed on the parameter. Operations can
        be either calculation of loss or transformation of the parameter, depending
        on the subclass.
        """
        # Reset torch parameters
        self._torch_parameters = {}

        # If the draws are not provided, then we draw from the parameter
        if draws is None:
            _, draws = self.draw(1)

        # Set up the Pytorch parameters. We parametrize in terms of the parents
        for paramname, param in self._parents.items():

            # Get the current draw. Squash the first dimension if it exists (this
            # is the sample dimension).
            draw = torch.from_numpy(draws[param].squeeze(axis=0))

            # Record the PyTorch parameter. Everything is learnable except for
            # constants.
            if isinstance(
                param, (dms_components.Parameter, dms_components.TransformedParameter)
            ):
                # Transform the parameter if it is bounded
                if paramname in self.bounded_parameters:

                    # Negatives should be transformed to be positive
                    if paramname in self.NEGATIVE_PARAMS:
                        draw = torch.abs(draw)

                    # To log space
                    draw = torch.log(draw)

                # Record the parameter as a learnable parameter
                self._torch_parameters[paramname] = nn.Parameter(draw)
            else:
                assert isinstance(param, dms_components.Constant)
                self._torch_parameters[paramname] = draw

    def get_torch_observables(
        self, observed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Gets the values of the observables from the perspective of the model component.
        An "observable" is what would be drawn from this component if it were
        generating. In practice, this means the values of the PyTorch tensors of
        the child components of this component.

        Args:
            observed (Optional[torch.Tensor], optional): The observed value. This
                only needs to be provided for observed components. Latent components
                will automatically identify the child components and in turn use
                that parameter's components as the observed value. Defaults to None.

        Returns:
            torch.Tensor: The values of the observables from the perspective of
                the bound component.
        """
        # If an observable, then we must have an observed value
        if self.observable and observed is None:
            raise ValueError(
                "Observed value must be provided for observable parameters"
            )

        # Get the observables from the children
        observations = [] if observed is None else [observed]
        for child in self.children:

            # Get the parameter name in the child
            # pylint: disable=protected-access
            child_paramname = child._component_to_paramname[self]
            # pylint: enable=protected-access

            # Get the observed value
            observations.append(child.torch_parameters[child_paramname])

        # If multiple observables, they must be the same object
        assert all(obs is observations[0] for obs in observations)

        # Return the observations
        return observations[0]

    def _link_to_sibling(
        self, parent: "AbstractModelComponent", sibling: "AbstractModelComponent"
    ) -> None:
        """
        If two components have the same parent, then that parent must always have
        the same observable value. This function ensures that that is the case by
        replacing the PyTorch parameter of this component with the equivalent PyTorch
        parameter of the sibling component.

        Args:
            parent (AbstractModelComponent): The parent component.
            sibling (AbstractModelComponent): The sibling component. It shares a
                parent with this component.
        """
        # Get the name of the parent component in both this component and its sibling
        this_paramname = self._component_to_paramname[parent]
        # pylint: disable=protected-access
        sibling_paramname = sibling._component_to_paramname[parent]
        # pylint: enable=protected-access

        # Replace the PyTorch parameter of this component with the PyTorch parameter
        # of the sibling component
        # pylint: disable=protected-access
        self._torch_parameters[this_paramname] = sibling._torch_parameters[
            sibling_paramname
        ]
        # pylint: enable=protected-access

        # Record the shared parameter
        self._shared_parameters.add(this_paramname)

    @abstractmethod
    def _draw(self, n: int, level_draws: dict[str, npt.NDArray]) -> npt.NDArray:
        """Sample from the distribution that represents the parameter"""

    def draw(
        self,
        n: int,
        _drawn: Optional[dict["AbstractModelComponent", npt.NDArray]] = None,
    ) -> tuple[npt.NDArray, dict["AbstractModelComponent", npt.NDArray]]:
        """
        Recursively draws from the parameter and its parents. The draws for this
        component are the first object returned. The draws from the recursion are
        the second object returned.
        """
        # Build the _drawn dictionary if it is not already built
        if _drawn is None:
            _drawn = {}

        # Loop over the parents and draw from them if we haven't already
        level_draws: dict[str, npt.NDArray] = {}
        for paramname, parent in self._parents.items():

            # If the parent has been drawn, use the already drawn value. Otherwise,
            # draw from the parent.
            if parent in _drawn:
                parent_draw = _drawn[parent]
            else:
                parent_draw, _ = parent.draw(n, _drawn)
                _drawn[parent] = parent_draw

            # Add the parent draw to the level draws. Expand the number of dimensions
            # if necessary to account for the addition of "n" draws.
            print(paramname, parent_draw)
            dims_to_add = (
                self.ndim + 1
            ) - parent_draw.ndim  # Implicit prepended dimensions
            assert (
                dims_to_add >= 0
            ), f"{self.ndim} {parent_draw.ndim} {self.model_varname} {parent.model_varname}"
            level_draws[paramname] = np.expand_dims(
                parent_draw, axis=tuple(range(1, dims_to_add + 1))
            )

        # Now draw from the current parameter
        draws = self._draw(n, level_draws)

        # Update the _drawn dictionary
        assert self not in _drawn
        _drawn[self] = draws

        # Test the ranges of the draws
        for paramname, param in self._parents.items():
            if paramname in self.POSITIVE_PARAMS:
                assert np.all(_drawn[param] >= 0)
            elif paramname in self.NEGATIVE_PARAMS:
                assert np.all(_drawn[param] <= 0)
            elif paramname in self.SIMPLEX_PARAMS:
                assert np.all((_drawn[param] >= 0) & (_drawn[param] <= 1))
                assert np.allclose(np.sum(_drawn[param], axis=-1), 1)

        return draws, _drawn

    def walk_tree(
        self, walk_down: bool = True
    ) -> list[tuple["AbstractModelComponent", "AbstractModelComponent"]]:
        """
        Walks the tree of parameters, either up or down. "up" means walking from
        the children to the parents, while "down" means walking from the parents
        to the children.

        Args:
            walk_down (bool): Whether to walk down the tree (True) or up the tree (False).

        Returns:
            list[tuple["AbstractModelComponent", "AbstractModelComponent"]]: The
            lineage. Each tuple contains the reference parameter in the first position
            and its relative (child if walking down, parent if walking up) in the
            second.
        """
        # Get the variables to loop over
        relatives = self.children if walk_down else self.parents

        # Recurse
        to_return = []
        for relative in relatives:

            # Add the current parameter and the relative parameter to the list
            to_return.append((self, relative))

            # Recurse on the relative parameter
            to_return.extend(relative.walk_tree(walk_down=walk_down))

        return to_return

    @abstractmethod
    def get_transformation_assignment(self, index_opts: tuple[str, ...]) -> str:
        """Gets the transformation assignment operation for the model component."""

    @abstractmethod
    def get_target_incrementation(self, index_opts: tuple[str, ...]) -> str:
        """Gets the target incrementation operation for the model component."""

    @abstractmethod
    def _handle_transformation_code(
        self, param: "AbstractModelComponent", index_opts: tuple[str, ...]
    ) -> str:
        """Handles code formatting for a transformed parameter."""

    @abstractmethod
    def format_stan_code(self, **to_format: str) -> str:
        """Formats the Stan code for the parameter."""

    def _get_formattables(
        self, param: "AbstractModelComponent", index_opts: tuple[str, ...]
    ) -> str:
        """Gets the inputs to `format_stan_code` for a parent parameter."""
        # If the parameter is a constant or another parameter, record
        if isinstance(param, (dms_components.Constant, dms_components.Parameter)):
            return param.get_indexed_varname(index_opts)

        # If the parameter is transformed and not named, the computation is
        # happening in the model. Otherwise, the computation has already happened
        # in the transformed parameters block. Note that THIS instance handles the
        # transformation code based on ITS type, not the type of the parameter.
        elif isinstance(param, dms_components.TransformedParameter):
            return self._handle_transformation_code(param=param, index_opts=index_opts)

        # Otherwise, raise an error
        else:
            raise TypeError(f"Unknown model component type {type(param)}")

    def get_stan_code(self, index_opts: tuple[str, ...]) -> str:
        """Gets the Stan code for the parameter."""

        # Recursively gather the transformations until we hit a non-transformed
        # parameter or a recorded variable
        return self.format_stan_code(
            **{
                name: self._get_formattables(param, index_opts)
                for name, param in self._parents.items()
            }
        )

    def _get_torch_parameter(self, paramname: str) -> torch.Tensor:
        """Returns the transformed PyTorch parameter for the given parameter name."""
        # Get the parameter
        param = self._torch_parameters[paramname]

        # If a transformed parameter, return to the original space
        if paramname in self.bounded_parameters and isinstance(
            self._parents[paramname],
            (dms_components.Parameter, dms_components.TransformedParameter),
        ):

            # First exponentiate
            param = torch.exp(param)

            # Negative parameters should be negated
            if paramname in self.NEGATIVE_PARAMS:
                param = -param

            # If a simplex, normalize
            if paramname in self.SIMPLEX_PARAMS:
                param = param / torch.sum(param, dim=-1)

        return param

    def _get_transformed_torch_parameters(self) -> dict[str, torch.Tensor]:
        """Returns the PyTorch parameters for this component, appropriately transformed."""
        return {
            paramname: self._get_torch_parameter(paramname)
            for paramname in self._torch_parameters
        }

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __contains__(self, key: str) -> bool:
        """Check if the parameter has a parent with the given key"""
        return key in self._parents

    def __getitem__(self, key: str) -> "AbstractModelComponent":
        """Get the parent parameter with the given key"""
        return self._parents[key]

    def __getattr__(self, key: str) -> "AbstractModelComponent":
        """Get the parent parameter with the given key"""
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(f"Attribute '{key}' not found in {self}") from error

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the parameter"""
        return self._shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the parameter"""
        return len(self.shape)

    @property
    def is_named(self) -> bool:
        """Return whether the parameter has a name assigned by a model"""
        return self._model_varname != ""

    @property
    def stan_bounds(self) -> str:
        """Return the Stan bounds for this parameter"""
        # Format the lower and upper bounds
        lower = (
            "" if self.STAN_LOWER_BOUND is None else f"lower={self.STAN_LOWER_BOUND}"
        )
        upper = (
            "" if self.STAN_UPPER_BOUND is None else f"upper={self.STAN_UPPER_BOUND}"
        )

        # Combine the bounds
        if lower and upper:
            bounds = f"{lower}, {upper}"
        elif lower:
            bounds = lower
        elif upper:
            bounds = upper
        else:
            return ""

        # Return the bounds
        return f"<{bounds}>"

    @property
    def stan_dtype(self) -> str:
        """Return the Stan data type for this parameter"""

        # Get the base datatype
        dtype = self.BASE_STAN_DTYPE

        # Base data type for 0-dimensional parameters. If the parameter is 0-dimensional,
        # then we can only have real or int as the data type.
        if self.ndim == 0:
            assert dtype in {"real", "int"}
            return f"{dtype}{self.stan_bounds}"

        # Convert shape to strings
        string_shape = [str(dim) for dim in self.shape]

        # Handle different data types for different dimensions
        if dtype == "real":  # Becomes vector or array of vectors
            dtype = f"vector{self.stan_bounds}[{string_shape[-1]}]"
            if self.ndim > 1:
                dtype = f"array[{','.join(string_shape[:-1])}] {dtype}"

        elif dtype == "int":  # Becomes array
            dtype = f"array[{','.join(string_shape)}] int{self.stan_bounds}"

        elif dtype == "simplex":  # Becomes array of simplexes
            dtype = f"array[{','.join(string_shape[:-1])}] simplex[{string_shape[-1]}]"

        else:
            raise AssertionError(f"Unknown data type {dtype}")

        return dtype

    @property
    def stan_code_level(self) -> int:
        """
        The level (index of the for-loop block) at which the parameter is manipulated
        in the Stan code. This is the number of dimensions of the parameter minus
        one, clipped at zero. This is because the last dimension is assumed to always
        be vectorized.
        """
        return max(self.ndim - 1, 0)

    @property
    def stan_parameter_declaration(self) -> str:
        """Returns the Stan parameter declaration for this parameter."""
        return f"{self.stan_dtype} {self.model_varname}"

    @property
    def model_varname(self) -> str:
        """Return the DMS Stan variable name for this parameter"""
        # If the _model_varname variable is set, then we return it
        if self.is_named:
            return self._model_varname

        # Otherwise, we automatically create the name. This is the name of the
        # child components and the name of this component as defined in that child
        # component separated by underscores.
        return "_".join(
            [
                f"{child.model_varname}_{name}"
                for child, name in self.get_child_paramnames().items()
            ]
        )

    @model_varname.setter
    def model_varname(self, name: str) -> None:
        """Set the DMS Stan variable name for this parameter"""
        self._model_varname = name

    @property
    def constants(self):
        """Return the constants of the component."""
        return {
            name: component
            for name, component in self._parents.items()
            if isinstance(component, dms_components.Constant)
        }

    @property
    def draw_shape(self) -> tuple[int, ...]:
        """Return the shape of the draws for the parameter."""
        return self._draw_shape

    @property
    def parents(self) -> list["AbstractModelComponent"]:
        """
        Gathers the parent parameters of the current parameter.

        Returns:
            list[AbstractModelComponent]: Parent parameters of the current parameter.
        """
        return list(self._parents.values())

    @property
    def children(self) -> list["AbstractModelComponent"]:
        """
        Gathers the children parameters of the current parameter.

        Returns:
            list[AbstractModelComponent]: Children parameters of the current parameter.
        """
        return self._children.copy()

    @property
    def bounded_parameters(self) -> set[str]:
        """
        Gets the identities of the parameters that are stored in a different space
        in PyTorch than in the bound parameter.
        """
        return self.POSITIVE_PARAMS | self.NEGATIVE_PARAMS | self.SIMPLEX_PARAMS

    @property
    def torch_parameters(self) -> dict[str, torch.Tensor]:
        """Return the PyTorch parameters for this component, appropriately transformed."""
        return self._get_transformed_torch_parameters()
