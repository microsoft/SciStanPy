"""Runs Hamiltonian Monte Carlo (HMC) for FLIP datasets."""

from __future__ import annotations

import argparse
import os.path

from typing import TYPE_CHECKING

from scistanpy.model.results import SampleResults
from flipv3.trpb_models import get_trpb_instance
from flipv3.pdz3_models import get_pdz3_instance


if TYPE_CHECKING:
    from scistanpy.model import Model

# Define valid combinations of dataset and subset.
VALID_COMBINATIONS = {
    "trpb": {
        "libA",
        "libB",
        "libC",
        "libD",
        "libE",
        "libF",
        "libG",
        "libH",
        "libI",
        "four-site",
    },
    "pdz": {"cript-c", "cript-n", "cis", "trans-1", "trans-2"},
}


def define_base_parser() -> argparse.ArgumentParser:
    """Defines the base parser shared by all pipelines."""
    # Build the base parser
    parser = argparse.ArgumentParser(add_help=False)

    # A few required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["trpb", "pdz"],
        required=True,
        help="Datset on which to run HMC.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        help="Name of the specific dataset to use (e.g., libA for trpb)",
    )

    parser.add_argument(
        "--growth_rate",
        type=str,
        choices=["exponential", "gamma", "lomax"],
        required=True,
        help="Distribution that defines the mean rate of the model.",
    )
    parser.add_argument(
        "--growth_curve",
        type=str,
        choices=["exponential", "logistic"],
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
        "--ignore_od",
        action="store_false",
        dest="include_od",
        help=(
            "If set, the model will not include optical density (OD) data. This is "
            "only applicable for TrpB datasets."
        ),
    )

    return parser


def parse_args():
    """Parse command line arguments."""
    # Build the parser specifically for this pipeline
    parser = argparse.ArgumentParser(
        description="Run HMC for FLIP datasets.",
        parents=[define_base_parser()],
    )

    # Add arguments specific to this pipeline
    parser.add_argument(
        "--n_chains",
        type=int,
        default=8,
        help="Number of chains to run. Default = 8.",
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=2000,
        help="Number of warmup iterations. Default = 2000.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of samples to draw after warmup. Default = 2000.",
    )
    parser.add_argument(
        "--use_dask",
        action="store_true",
        help="Use Dask when diagnosing the model.",
    )
    parser.add_argument(
        "--force_compile",
        action="store_true",
        help="Force compilation of the model even if it is already compiled.",
    )
    parser.add_argument(
        "--conversion_catchup",
        action="store_true",
        help=(
            "If set, the pipeline will not run HMC but will instead catch up on "
            "previous runs that failed during the csv-to-nc conversion step."
        ),
    )

    return parser.parse_args()


def check_base_args(args: argparse.Namespace) -> None:
    """Checks command line arguments shared by all pipelines"""
    # Check if the dataset and subset are valid
    if args.dataset not in VALID_COMBINATIONS:
        raise ValueError(f"Invalid dataset: {args.dataset}.")
    if args.subset not in VALID_COMBINATIONS[args.dataset]:
        raise ValueError(
            f"Invalid subset for {args.dataset}: {args.subset}. Options are: "
            f"{', '.join(VALID_COMBINATIONS[args.dataset])}."
        )

    # Flip data dir must exist
    if not os.path.exists(args.flip_data):
        raise ValueError(f"Flip data directory does not exist: {args.flip_data}.")

    # Output dir must exist
    if not os.path.exists(args.output_dir):
        raise ValueError(f"Output directory does not exist: {args.output_dir}.")

    # Seed must be a positive integer
    if args.seed <= 0:
        raise ValueError("Seed must be a positive integer.")


def check_args(args: argparse.Namespace) -> None:
    """Checks command line arguments for validity."""
    # Check base arguments
    check_base_args(args)

    # Chains, warmup, and samples must be positive integers
    for arg in ("n_chains", "n_warmup", "n_samples"):
        if getattr(args, arg) <= 0:
            raise ValueError(f"{arg} must be a positive integer.")


def prep_run(args: argparse.Namespace) -> "Model":
    """
    Preps for the run by checking arguments, loading the data, and instantiating
    the model.
    """
    # Finalize the output path. This is the provided output location with additional
    # folders for the dataset and subset added on.
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.subset)
    os.makedirs(args.output_dir, exist_ok=True)

    # Build the model instance
    instance_kwargs = {
        "filepath": os.path.join(args.flip_data, "counts", args.dataset, args.subset),
        "growth_curve": args.growth_curve,
        "growth_rate": args.growth_rate,
    }
    if args.dataset == "trpb":
        instance_factory = get_trpb_instance
        instance_kwargs["filepath"] += ".csv"
        instance_kwargs["lib"] = args.subset
    elif args.dataset == "pdz":
        instance_factory = get_pdz3_instance
        instance_kwargs["filepath"] += ".tsv"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return instance_factory(**instance_kwargs)


def run_hmc(args: argparse.Namespace) -> None:
    """Run HMC for the specified dataset and model."""
    # Prepare the run
    model = prep_run(args)
    model_name = f"{args.dataset}_{args.subset}_{args.growth_rate}_{args.growth_curve}"

    # Run HMC unless we are catching up due to a failed csv-to-nc conversion
    if args.conversion_catchup:

        # Load the fit object from the csv files
        res = SampleResults(
            model=model,
            fit=os.path.join(args.output_dir, f"{model_name}*.csv"),
            use_dask=args.use_dask,
        )

    else:
        # Run HMC
        res = model.mcmc(
            output_dir=args.output_dir,
            force_compile=args.force_compile,
            model_name=model_name,
            chains=args.n_chains,
            seed=args.seed,
            iter_warmup=args.n_warmup,
            iter_sampling=args.n_samples,
            show_console=True,
            refresh=10,
            use_dask=args.use_dask,
        )

    # Run diagnostics on the results
    print("Running diagnostics...")
    _ = res.diagnose()

    # Save the inference object with diagnostics completed
    print("Saving updated results...")
    res.save_netcdf(os.path.join(args.output_dir, f"{model_name}_diagnosed.nc"))


def main():
    """Main function to run HMC for FLIP datasets."""
    # Parse command line arguments
    args = parse_args()

    # Check arguments
    check_args(args)

    # Run HMC
    run_hmc(args)


if __name__ == "__main__":
    main()
