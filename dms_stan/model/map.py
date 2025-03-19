"""Holds code for the maximum a posteriori (MAP) estimation of the model parameters."""

from typing import Optional

import hvplot.pandas  # pylint: disable=unused-import
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

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

    def sample(self, n: int, seed: Optional[int] = None) -> npt.NDArray:
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
        map_estimate: dict[str, npt.NDArray],
        distributions: dict[str, torch.distributions.Distribution],
        losses: npt.NDArray,
    ):

        # The keys of the map estimate should be a subset of the keys of the distributions
        if not set(map_estimate.keys()).issubset(distributions.keys()):
            raise ValueError(
                "Keys of map estimate should be a subset of the keys of the distributions"
            )

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
