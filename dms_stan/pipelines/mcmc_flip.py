"""Runs Hamiltonian Monte Carlo (HMC) for FLIP datasets."""

import argparse
import os.path

from dms_stan.flip_dsets import load_pdz_dataset, load_trpb_dataset
from dms_stan.model import Model
from dms_stan.model.enrichment import (
    ExponRateExponGrowth,
    ExponRateSigmoidGrowth,
    GammaInvRateExponGrowth,
    GammaInvRateSigmoidGrowth,
    LomaxRateExponGrowth,
    LomaxRateSigmoidGrowth,
    NonHierarchicalExponRateExponGrowth,
    NonHierarchicalExponRateSigmoidGrowth,
    NonHierarchicalLomaxRateExponGrowth,
    NonHierarchicalLomaxRateSigmoidGrowth,
)
from dms_stan.model.stan.stan_results import SampleResults

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

# Map dataset names to their loading functions and file extensions
LOAD_DATASET_MAP = {
    "trpb": (load_trpb_dataset, ".csv"),
    "pdz": (load_pdz_dataset, ".tsv"),
}


# Define the non-hierarchical datasets for TrpB
trpb_non_hierarchical_dsets = {"libA", "libB", "libC"}


# Map experimental conditions to the appropriate models
def build_model_map():

    # Define the appropriate dictionaries for the hierarchical and non-hierarchical
    # models
    hierarchical_models = {
        "exponential": {
            "exponential": ExponRateExponGrowth,
            "logistic": ExponRateSigmoidGrowth,
        },
        "gamma": {
            "exponential": GammaInvRateExponGrowth,
            "logistic": GammaInvRateSigmoidGrowth,
        },
        "lomax": {
            "exponential": LomaxRateExponGrowth,
            "logistic": LomaxRateSigmoidGrowth,
        },
    }
    non_hierarchical_models = {
        "exponential": {
            "exponential": NonHierarchicalExponRateExponGrowth,
            "logistic": NonHierarchicalExponRateSigmoidGrowth,
        },
        "gamma": {
            "exponential": None,
            "logistic": None,
        },
        "lomax": {
            "exponential": NonHierarchicalLomaxRateExponGrowth,
            "logistic": NonHierarchicalLomaxRateSigmoidGrowth,
        },
    }

    # Build the model map
    model_map = {}
    for dataset, subsets in VALID_COMBINATIONS.items():

        # Build an empty dictionary for the dataset
        model_map[dataset] = {}

        # Process all subsets
        for subset in subsets:

            # Get the appropriate model map
            if dataset == "trpb" and subset in trpb_non_hierarchical_dsets:
                model_map[dataset][subset] = non_hierarchical_models
            else:
                model_map[dataset][subset] = hierarchical_models

    return model_map


MODEL_MAP = build_model_map()


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
        help="Name of the specific dataset to use (e.g., LibA for trpb)",
    )

    parser.add_argument(
        "--rate_dist",
        type=str,
        choices=["exponential", "gamma", "lomax"],
        required=True,
        help="Distribution that defines the mean rate of the model.",
    )
    parser.add_argument(
        "--growth_func",
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

    # Check if the rate distribution and growth function are valid
    if args.rate_dist not in MODEL_MAP[args.dataset][args.subset]:
        raise ValueError(f"Invalid rate distribution: {args.rate_dist}.")
    if args.growth_func not in MODEL_MAP[args.dataset][args.subset][args.rate_dist]:
        raise ValueError(f"Invalid growth function: {args.growth_func}.")

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


def prep_run(args: argparse.Namespace) -> Model:
    """
    Preps for the run by checking arguments, loading the data, and instantiating
    the model.
    """
    # Finalize the output path. This is the provided output location with additional
    # folders for the dataset and subset added on.
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.subset)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the dataset and remove information about the variants present
    load_func, file_ext = LOAD_DATASET_MAP[args.dataset]
    data = load_func(
        os.path.join(
            args.flip_data, "counts", args.dataset, f"{args.subset}{file_ext}"
        ),
    )
    data.pop("variants")

    # Build an instance of the model
    model_class = MODEL_MAP[args.dataset][args.subset][args.rate_dist][args.growth_func]
    if model_class is None:
        raise ValueError(
            f"Model class not found for dataset {args.dataset}, rate distribution "
            f"{args.rate_dist}, and growth function {args.growth_func}."
        )
    return model_class(**data)


def run_hmc(args: argparse.Namespace) -> None:
    """Run HMC for the specified dataset and model."""
    # Prepare the run
    model = prep_run(args)
    model_name = f"{args.dataset}_{args.subset}_{args.rate_dist}_{args.growth_func}"

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
