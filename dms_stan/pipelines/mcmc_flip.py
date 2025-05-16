"""Runs Hamiltonian Monte Carlo (HMC) for FLIP datasets."""

import argparse

# Define valid combinations of dataset and subset
VALID_COMBINATIONS = {
    "trpb": {
        "LibA",
        "LibB",
        "LibC",
        "LibD",
        "LibE",
        "LibF",
        "LibG",
        "LibH",
        "LibI",
        "four-site",
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run HMC for FLIP datasets.")

    # A few required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        action="choose",
        options=["trpb", "pdz3"],
        required=True,
        help="Datset on which to run HMC.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        help="Name of the specific dataset to use (e.g., LibA for trpb)",
    )

    parser.add_argument(
        "--rate_dist",
        type=str,
        action="choose",
        options=["exponential", "gamma", "lomax"],
        required=True,
        help="Distribution that defines the mean rate of the model.",
    )
    parser.add_argument(
        "--growth_func",
        type=str,
        action="choose",
        options=["exponential", "logistic"],
        required=True,
        help="Growth function to use.",
    )
    parser.add_argument(
        "--flip_data",
        type=str,
        required=True,
        help="Path to the folder containing all raw flip data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the folder where the output will be saved.",
    )

    # Now some optionals
    parser.add_argument(
        "--seed",
        type=int,
        default=1025,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n_chains",
        type=int,
        default=4,
        help="Number of chains to run.",
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=3000,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples to draw after warmup.",
    )
    parser.add_argument(
        "--force_compile",
        action="store_true",
        help="Force compilation of the model even if it is already compiled.",
    )

    return parser.parse_args()
