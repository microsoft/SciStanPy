"""Holds code for the maximum a posteriori (MAP) estimation of the model parameters."""

from typing import Literal, Optional, overload

import arviz as az
import hvplot.pandas  # pylint: disable=unused-import
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import xarray as xr

import dms_stan as dms

# TODO: Extend the MAP such that we can use it like the SampleResults class. We
# want to be able to do the same posterior predictive checks.


class MAPParam:
    """Holds the MAP estimate for a single parameter."""

    def __init__(
        self,
        name: str,
        value: Optional[npt.NDArray],
        distribution: dms.custom_types.DMSStanDistribution,
    ):

        # Store the inputs
        self.name = name
        self.map = value
        self.distribution = distribution

    def draw(self, n: int, seed: Optional[int] = None) -> npt.NDArray:
        """
        Sample from the MAP estimate.
        """
        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Sample from the distribution
        return self.distribution.sample((n,)).cpu().numpy()


class MAP:
    """Holds the MAP estimate for all parameters of a model."""

    def __init__(
        self,
        model: "dms.model.Model",
        map_estimate: dict[str, npt.NDArray],
        distributions: dict[str, torch.distributions.Distribution],
        losses: npt.NDArray,
        data: dict[str, npt.NDArray],
    ):

        # The keys of the map estimate should be a subset of the keys of the distributions
        if not set(map_estimate.keys()).issubset(distributions.keys()):
            raise ValueError(
                "Keys of map estimate should be a subset of the keys of the distributions"
            )

        # Record the model and data
        self.model = model
        self.data = data

        # Store inputs. Each key in the map estimate will be mapped to an instance
        # variable
        self.parameters: str = []
        for key, value in distributions.items():
            self.parameters.append(key)
            setattr(
                self,
                key,
                MAPParam(name=key, value=map_estimate.get(key), distribution=value),
            )

        # Record the loss trajectory as a pandas dataframe
        self.losses = pd.DataFrame(
            {"-log pdf/pmf": losses, "iteration": np.arange(len(losses))}
        )

    def plot_loss_curve(self, logy: bool = True):
        """Plots the loss curve of the MAP estimation."""
        # Get y-label and title
        if logy:
            ylabel = "log(-log pdf/pmf)"
            title = "Log Loss Curve"
        else:
            ylabel = "-log pdf/pmf"
            title = "Loss Curve"

        return self.losses.hvplot.line(
            x="iteration", y="-log pdf/pmf", title=title, logy=logy, ylabel=ylabel
        )

    @overload
    def draw(
        self,
        n: int,
        *,
        seed: Optional[int],
        as_xarray: Literal[True],
        as_inference_data: Literal[False]
    ) -> xr.Dataset: ...

    @overload
    def draw(
        self,
        n: int,
        *,
        seed: Optional[int],
        as_xarray: Literal[False],
        as_inference_data: Literal[True]
    ) -> az.InferenceData: ...

    @overload
    def draw(
        self,
        n: int,
        *,
        seed: Optional[int],
        as_xarray: Literal[False],
        as_inference_data: Literal[False]
    ) -> dict[str, npt.NDArray]: ...

    def draw(
        self, n: int = 1000, *, seed=None, as_xarray=False, as_inference_data=False
    ):
        """Draws samples from the MAP estimate.

        Args:
            n (int): The number of samples to draw. Defaults to 1000.
            seed (int, optional): Sets the random seed. Defaults to None.
            as_xarray (bool, optional): If `True`, results are returned as an xarray
                DataSet. Defaults to `False`, meaning results are returned as a
                dictionary of numpy arrays. This and `as_inference_data` are mutually
                exclusive.
            as_inference_data (bool, optional): If `True`, results are returned as
                an ArviZ InferenceData object. Defaults to `False`, meaning results
                are returned as a dictionary of numpy arrays. This and `as_xarray`
                are mutually exclusive.

        Returns:
            dict[str, npt.NDArray] | xr.DataSet | az.InferenceData: The samples
                drawn from the MAP estimate. If `as_xarray` is `True`, returns an
                xarray DataSet. Otherwise, returns a dictionary of numpy arrays.
        """
        # pylint: disable=protected-access
        # `as_xarray` and `as_inference_data` are mutually exclusive
        if as_xarray and as_inference_data:
            raise ValueError(
                "Cannot set both `as_xarray` and `as_inference_data` to `True`."
            )

        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Draw samples
        draws = {
            getattr(self.model, param): getattr(self, param).draw(n)
            for param in self.parameters
        }

        # If returning as an xarray or InferenceData object, convert the draws to
        # an xarray format.
        if as_xarray or as_inference_data:
            draws = self.model._dict_to_xarray(draws)

            # We are done if we are returning as an xarray
            if as_xarray:
                return draws

            # Otherwise, we also are going to want to attach the observed data
            # to the InferenceData object. First, rename the "n" dimension to "sample"
            draws = draws.rename_dims({"n": "sample"})

            # Now separate out the observables from the latent variables. Build
            # the initial inference data object with the latent variables
            inference_data = az.convert_to_inference_data(
                draws[
                    [
                        p
                        for p in self.parameters
                        if not getattr(self.model, p).observable
                    ]
                ]
            )

            # Add the observables and the observed data to the inference data object
            inference_data.add_groups(
                observed_data=xr.Dataset(
                    data_vars={
                        k: self.model._compress_for_xarray(v)[0]
                        for k, v in self.data.items()
                    }
                ),
                posterior_predictive=draws[
                    [p for p in self.parameters if getattr(self.model, p).observable]
                ],
            )
            return inference_data

        # If we make it here, we are not returning as an xarray or InferenceData
        # object, so we need to convert the parameters to their original names
        # and return them as a dictionary
        return {k.model_varname: v for k, v in draws.items()}
