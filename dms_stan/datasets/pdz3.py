"""
Modeling code for the PDZ3 dataset. See
'https://www.biorxiv.org/content/10.1101/2024.04.25.591103v1' for more details.
"""

from abc import abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd

import dms_stan.model as dms
import dms_stan.model.components as dms_components
import dms_stan.operations as dms_ops


class PDZ3Base(dms.Model):
    """Base class for the PDZ3 dataset models."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        ending_counts: npt.NDArray[np.int64],
        alpha: float = 0.75,
        inv_r_alpha: float = 2.5,
        inv_r_beta: float = 0.5,
    ):
        """Initializes the PDZ3 model with starting and ending counts.

        Args:
            starting_counts (npt.NDArray[np.int64]): Starting counts for the PDZ3 dataset.
            ending_counts (npt.NDArray[np.int64]): Ending counts for the PDZ3 dataset.
        """
        # Register default data
        super().__init__(
            default_data={
                "starting_counts": starting_counts,
                "ending_counts": ending_counts,
            }
        )

        # Inputs should be 2D arrays with equivalent dimensions (N replicates, N variants)
        assert starting_counts.ndim == 2
        assert ending_counts.shape == starting_counts.shape

        # Get the number of replicates and variants
        self.n_replicates, self.n_variants = starting_counts.shape

        # Set the total number of starting and ending counts by replicate
        self.total_starting_counts = dms_components.Constant(
            starting_counts.sum(axis=-1, keepdims=True), togglable=False
        )
        self.total_ending_counts = dms_components.Constant(
            ending_counts.sum(axis=-1, keepdims=True), togglable=False
        )

        # Every variant has a fundamental "rate" which will be the same across
        # experiments. This rate is modeled by the Gamma distribution and is the
        # inverse of the growth rate (so that we can use it as the exponential rate
        # parameter).
        self.inv_r_mean = dms_components.Gamma(
            alpha=inv_r_alpha,
            beta=inv_r_beta,
            shape=(self.n_variants,),
        )

        # The inverse rate is the beta parameter for the exponential distributions
        # describing the growth rates in each experiment.
        self.r = dms_components.Exponential(
            beta=self.inv_r_mean,
            shape=(self.n_replicates, self.n_variants),
        )

        # Starting proportions are Dirichlet distributed
        self.theta_t0 = dms_components.Dirichlet(
            alpha=alpha, shape=(self.n_replicates, self.n_variants)
        )

        # Calculate ending proportions
        self.raw_abundances_t1 = self._define_growth_function()
        self.theta_t1 = dms_ops.normalize(self.raw_abundances_t1)

        # The counts are modeled as multinomial distributions
        self.starting_counts = dms_components.Multinomial(
            theta=self.theta_t0, N=self.total_starting_counts
        )
        self.ending_counts = dms_components.Multinomial(
            theta=self.theta_t1, N=self.total_ending_counts
        )

    @abstractmethod
    def _define_growth_function(self) -> dms_components.TransformedParameter:
        """
        Defines the transformation for the model. This must be overridden in
        child classes and is used to define the transformation between starting
        and ending abundances.
        """

    @classmethod
    def from_data_file(cls, filepath: str, **kwargs) -> "PDZ3Base":
        """Loads the PDZ3 dataset from a file and initializes the model.

        Args:
            filepath (str): Path to the file to load.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            PDZ3Base: An instance of the PDZ3Base model.
        """
        # Load data and remove variants
        data = load_pdz3_dataset(filepath)
        data.pop("variants")

        return cls(**data, **kwargs)  # pylint: disable=unexpected-keyword-arg


class PDZ3Exponential(PDZ3Base):
    """Models the change in counts for the dataset resulting from exponential growth."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        ending_counts: npt.NDArray[np.int64],
        alpha: float = 0.75,
        inv_r_alpha: float = 7.0,
        inv_r_beta: float = 1.0,
    ):
        """
        Models the change in counts for the dataset resulting from exponential growth.

        Args:
            starting_counts (npt.NDArray[np.int64]): Starting counts for the PDZ3 dataset.
            ending_counts (npt.NDArray[np.int64]): Ending counts for the PDZ3 dataset.
        """
        # Initialize the base class
        super().__init__(
            starting_counts=starting_counts,
            ending_counts=ending_counts,
            alpha=alpha,
            inv_r_alpha=inv_r_alpha,
            inv_r_beta=inv_r_beta,
        )

    def _define_growth_function(self) -> dms_components.BinaryExponentialGrowth:
        return dms_components.BinaryExponentialGrowth(
            A=self.theta_t0, r=self.r, shape=(self.n_replicates, self.n_variants)
        )


class PDZ3Sigmoid(PDZ3Base):
    """Models the change in counts for the dataset resulting from sigmoid growth."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        ending_counts: npt.NDArray[np.int64],
        alpha: float = 0.75,
        inv_r_alpha: float = 2.5,
        inv_r_beta: float = 0.5,
        c_alpha: float = 4.0,
        c_beta: float = 8.0,
    ):
        """
        Models the change in counts for the dataset resulting from sigmoid growth.

        Args:
            starting_counts (npt.NDArray[np.int64]): Starting counts for the PDZ3 dataset.
            ending_counts (npt.NDArray[np.int64]): Ending counts for the PDZ3 dataset.
        """
        # Define 'c'. We assume that there might be variability in the inflection
        # point timing for different replicates.
        self.c = dms_components.Gamma(
            alpha=c_alpha, beta=c_beta, shape=(starting_counts.shape[0], 1)
        )

        # Initialize the base class
        super().__init__(
            starting_counts=starting_counts,
            ending_counts=ending_counts,
            alpha=alpha,
            inv_r_alpha=inv_r_alpha,
            inv_r_beta=inv_r_beta,
        )

    def _define_growth_function(
        self,
    ) -> dms_components.SigmoidGrowthInitParametrization:

        # Get the ending abundances
        return dms_components.SigmoidGrowthInitParametrization(
            t=dms_components.Constant(1.0, togglable=False),
            x0=self.theta_t0,
            r=self.r,
            c=self.c,
            shape=(self.n_replicates, self.n_variants),
        )
