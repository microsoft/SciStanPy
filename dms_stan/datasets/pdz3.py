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


def load_pdz3_dataset(filepath: str) -> dict[str, npt.NDArray[np.int64]]:
    """Loads a given PDZ3 dataset file and returns the starting and ending counts.
    This function is appropriate for any file whose name ends with 'norm_for_mochi.tsv'
    in the Zenodo folder.

    Args:
        filepath (str): Path to the file to load.

    Returns:
        dict[str, npt.NDArray[np.int64]]: A dictionary with two keys:
            'starting_counts' and 'ending_counts', each containing a numpy array of
            the respective counts.
    """
    # Make sure we are loading a file from the Zenodo folder
    if not filepath.endswith("norm_for_mochi.tsv"):
        raise ValueError("File must end with 'norm_for_mochi.tsv'")

    # Load the file and fill missing values
    df = pd.read_csv(
        filepath,
        sep="\t",
        usecols={
            "aa_seq",
            "count_e1_s0",
            "count_e2_s0",
            "count_e3_s0",
            "count_e1_s1",
            "count_e2_s1",
            "count_e3_s1",
        },
    )

    # No duplicate AA sequences should be present
    assert (df.aa_seq.value_counts() == 1).all()

    # Drop the aa column and fill missing counts with 0s.
    df = df.drop(columns=["aa_seq"]).fillna(0).astype(int)

    # Get two numpy arrays, one for the starting counts and one for the ending counts
    return {
        "starting_counts": df[["count_e1_s0", "count_e2_s0", "count_e3_s0"]]
        .to_numpy()
        .T,
        "ending_counts": df[["count_e1_s1", "count_e2_s1", "count_e3_s1"]].to_numpy().T,
    }


class PDZ3Base(dms.Model):
    """Base class for the PDZ3 dataset models."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        ending_counts: npt.NDArray[np.int64],
        inv_r_alpha: float = 2.5,
        inv_r_beta: float = 0.5,
        **transform_kwargs
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

        # Define our transformation. This will be overridden in the child classes
        # to define how we go from starting to ending counts. This defines theta_t0
        # and theta_t1, which are the starting and ending abundances respectively.
        self._init_transformation(**transform_kwargs)

        # The counts are modeled as multinomial distributions
        self.starting_counts = dms_components.Multinomial(
            theta=self.theta_t0, N=self.total_starting_counts
        )
        self.ending_counts = dms_components.Multinomial(
            theta=self.theta_t1, N=self.total_ending_counts
        )

    @abstractmethod
    def _init_transformation(self, **kwargs) -> None:
        """
        Defines the transformation for the model. This must be overridden in
        child classes and is used to define the transformation between starting
        and ending abundances.
        """
        self.theta_t0 = None
        self.theta_t1 = None
        raise NotImplementedError("This method must be overridden in child classes.")

    @classmethod
    def from_data_file(cls, filepath: str, **kwargs) -> "PDZ3Base":
        """Loads the PDZ3 dataset from a file and initializes the model.

        Args:
            filepath (str): Path to the file to load.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            PDZ3Base: An instance of the PDZ3Base model.
        """
        return cls(**load_pdz3_dataset(filepath), **kwargs)


class PDZ3Exponential(PDZ3Base):
    """Models the change in counts for the dataset resulting from exponential growth."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        ending_counts: npt.NDArray[np.int64],
        inv_r_alpha: float = 7.0,
        inv_r_beta: float = 1.0,
        alpha: float = 2.0,
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
            inv_r_alpha=inv_r_alpha,
            inv_r_beta=inv_r_beta,
            alpha=alpha,
        )

    def _init_transformation(  # pylint: disable=arguments-differ
        self, alpha: float
    ) -> None:

        # Alpha describes the distribution of the starting abundances. These are
        # modeled as a Dirichlet distribution with a shared value for `alpha`
        self.theta_t0 = dms_components.Dirichlet(
            alpha=alpha, shape=(self.n_replicates, self.n_variants)
        )

        # Get the ending abundances.
        self.raw_abundances = dms_components.BinaryExponentialGrowth(
            A=self.theta_t0, r=self.r
        )
        self.theta_t1 = dms_ops.normalize(self.raw_abundances)


class PDZ3Sigmoid(PDZ3Base):
    """Models the change in counts for the dataset resulting from sigmoid growth."""

    def __init__(
        self,
        starting_counts: npt.NDArray[np.int64],
        ending_counts: npt.NDArray[np.int64],
        inv_r_alpha: float = 2.5,
        inv_r_beta: float = 0.5,
        alpha: float = 0.25,
        c_mean_alpha: float = 4.0,
        c_mean_beta: float = 8.0,
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
            alpha=c_mean_alpha, beta=c_mean_beta, shape=(starting_counts.shape[0], 1)
        )

        # Initialize the base class
        super().__init__(
            starting_counts=starting_counts,
            ending_counts=ending_counts,
            inv_r_alpha=inv_r_alpha,
            inv_r_beta=inv_r_beta,
            alpha=alpha,
        )

    def _init_transformation(  # pylint: disable=arguments-differ
        self, alpha: float
    ) -> None:
        # Alpha is used to model proportions at t -> infinity. This is again modeled
        # as a Dirichlet distribution. We scale the draw from this distribution by
        # the number of variants for numerical stability.
        self.A = (  # pylint: disable=invalid-name
            dms_components.Dirichlet(
                alpha=alpha, shape=(self.n_replicates, self.n_variants)
            )
            * self.n_variants
        )

        # Get the starting and ending abundances
        self.raw_abundances_t0 = dms_components.SigmoidGrowth(
            A=self.A, r=self.r, c=self.c, t=0
        )
        self.raw_abundances_t1 = dms_components.SigmoidGrowth(
            A=self.A, r=self.r, c=self.c, t=1
        )

        self.theta_t0 = dms_ops.normalize(self.raw_abundances_t0)
        self.theta_t1 = dms_ops.normalize(self.raw_abundances_t1)
