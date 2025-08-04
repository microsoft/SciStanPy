"""Holds code for running MLE estimation on DMS Stan models."""

from __future__ import annotations

import warnings

from typing import Literal, Optional, overload, TYPE_CHECKING

import arviz as az
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import xarray as xr

from dms_stan.model import results

if TYPE_CHECKING:
    from dms_stan import custom_types
    from dms_stan import model as dms_model


class MLEParam:
    """Holds the MLE estimate for a single parameter."""

    def __init__(
        self,
        name: str,
        value: Optional[npt.NDArray],
        distribution: "custom_types.DMSStanDistribution",
    ):

        # Store the inputs
        self.name = name
        self.mle = value
        self.distribution = distribution

    def draw(
        self, n: int, *, seed: Optional[int] = None, batch_size: Optional[int] = None
    ) -> npt.NDArray:
        """
        Sample from the MLE estimate.
        """
        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # If the batch size is not provided, we set it to `n`
        batch_size = batch_size or n

        # Calculate the batch sizes for each sampling iteration
        batch_sizes = [batch_size] * (n // batch_size)
        if (n_remaining := n % batch_size) > 0:
            batch_sizes.append(n_remaining)

        # Sample from the distribution
        return np.concatenate(
            [
                self.distribution.sample((batch_size,)).detach().cpu().numpy()
                for batch_size in batch_sizes
            ]
        )


class MLE:
    """Holds the MLE estimate for all parameters of a model."""

    def __init__(
        self,
        model: "dms_model.Model",
        mle_estimate: dict[str, npt.NDArray],
        distributions: dict[str, torch.distributions.Distribution],
        losses: npt.NDArray,
        data: dict[str, npt.NDArray],
    ):

        # The keys of the mle estimate should be a subset of the keys of the distributions
        if not set(mle_estimate.keys()).issubset(distributions.keys()):
            raise ValueError(
                "Keys of mle estimate should be a subset of the keys of the distributions"
            )

        # Record the model and data
        self.model = model
        self.data = data

        # Store inputs. Each key in the mle estimate will be mapped to an instance
        # variable
        self.model_varname_to_mle: dict[str, MLEParam] = {
            key: MLEParam(name=key, value=mle_estimate.get(key), distribution=value)
            for key, value in distributions.items()
        }

        # Set an attribute for all MLE parameters
        for k, v in self.model_varname_to_mle.items():
            if hasattr(self, k):
                raise ValueError(
                    f"MLE parameter {k} already exists in the model. Please rename it."
                )
            setattr(self, k, v)

        # Record the loss trajectory as a pandas dataframe
        self.losses = pd.DataFrame(
            {
                "-log pdf/pmf": losses,
                "iteration": np.arange(len(losses)),
                "shifted log(-log pdf/pmf)": losses - losses.min() + 1,
            },
        )

    def plot_loss_curve(self, logy: bool = True):
        """Plots the loss curve of the MLE estimation."""
        # Get y-label and title
        y = "-log pdf/pmf"
        if logy:
            if self.losses["-log pdf/pmf"].min() <= 0:
                warnings.warn("Negative values in loss curve. Using shifted log10.")
                y = "shifted log(-log pdf/pmf)"
                ylabel = y
            else:
                ylabel = "log(-log pdf/pmf)"
            title = "Log Loss Curve"
        else:
            ylabel = "-log pdf/pmf"
            title = "Loss Curve"

        return self.losses.hvplot.line(
            x="iteration", y=y, title=title, logy=logy, ylabel=ylabel
        )

    @overload
    def draw(
        self,
        n: int,
        *,
        seed: Optional[int],
        as_xarray: Literal[True],
        as_inference_data: Literal[False],
        batch_size: Optional[int] = None,
    ) -> xr.Dataset: ...

    @overload
    def draw(
        self,
        n: int,
        *,
        seed: Optional[int],
        as_xarray: Literal[False],
        batch_size: Optional[int] = None,
    ) -> dict[str, npt.NDArray]: ...

    def draw(self, n: int, *, seed=None, as_xarray=False, batch_size=None):
        """Draws samples from the MLE estimate.

        Args:
            n (int): The number of samples to draw.
            seed (int, optional): Sets the random seed. Defaults to None.
            as_xarray (bool, optional): If `True`, results are returned as an xarray
                DataSet. Defaults to `False`, meaning results are returned as a
                dictionary of numpy arrays. This and `as_inference_data` are mutually
                exclusive.

        Returns:
            dict[str, npt.NDArray] | xr.DataSet: The samples drawn from the MLE
                estimate. If `as_xarray` is `True`, returns an xarray DataSet.
                Otherwise, returns a dictionary of numpy arrays.
        """
        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Draw samples
        draws = {
            self.model.all_model_components_dict[k]: v.draw(n, batch_size=batch_size)
            for k, v in self.model_varname_to_mle.items()
        }

        # If returning as an xarray or InferenceData object, convert the draws to
        # an xarray format.
        if as_xarray:
            return self.model._dict_to_xarray(draws)  # pylint: disable=protected-access

        # If we make it here, we are not returning as an xarray or InferenceData
        # object, so we need to convert the parameters to their original names
        # and return them as a dictionary
        return {k.model_varname: v for k, v in draws.items()}

    def get_inference_obj(
        self,
        n: int = 1000,
        *,
        seed: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> results.MLEInferenceRes:
        """Builds an inference data object from the MLE estimate."""
        # Get the samples from the posterior
        draws = self.draw(n, seed=seed, as_xarray=True, batch_size=batch_size)

        # Otherwise, we also are going to want to attach the observed data
        # to the InferenceData object. First, rename the "n" dimension to "sample"
        # and add a dummy "chain" dimension
        draws = draws.rename_dims({"n": "draw"})
        draws = draws.expand_dims("chain", 0)

        # Now separate out the observables from the latent variables. Build
        # the initial inference data object with the latent variables
        inference_data = az.convert_to_inference_data(
            draws[
                [
                    varname
                    for varname, mle_param in self.model_varname_to_mle.items()
                    if not self.model.all_model_components_dict[varname].observable
                ]
            ]
        )

        # Add the observables and the observed data to the inference data object
        # pylint: disable=protected-access
        inference_data.add_groups(
            observed_data=xr.Dataset(
                data_vars={
                    k: self.model._compress_for_xarray(v)[0]
                    for k, v in self.data.items()
                }
            ),
            posterior_predictive=draws[
                [
                    varname
                    for varname, mle_param in self.model_varname_to_mle.items()
                    if self.model.all_model_components_dict[varname].observable
                ]
            ],
        )
        return results.MLEInferenceRes(inference_data)
