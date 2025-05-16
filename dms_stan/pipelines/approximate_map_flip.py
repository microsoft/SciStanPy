"""Same layout as `mcmc_flip.py` but with the `approximate_map` pipeline."""

import argparse
import os.path

import torch

from dms_stan.defaults import DEFAULT_EARLY_STOP, DEFAULT_LR, DEFAULT_N_EPOCHS
from dms_stan.pipelines.mcmc_flip import check_base_args, define_base_parser, prep_run


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Build the parser specifically for this pipeline
    parser = argparse.ArgumentParser(
        description="Approximate the MAP for FLIP datasets.",
        parents=[define_base_parser()],
    )

    # Add arguments specific to this pipeline
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=DEFAULT_N_EPOCHS,
        help=f"Maximum number of passes over the data during optimization. Default = {DEFAULT_N_EPOCHS}.",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=DEFAULT_EARLY_STOP,
        help=f"Number of epochs to wait before stopping if the loss does not improve. Default = {DEFAULT_EARLY_STOP}.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate for the optimizer. Default = {DEFAULT_LR}.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=0 if torch.cuda.is_available() else "cpu",
        choices=["cpu"]
        + (
            list(map(str, range(torch.cuda.device_count())))
            if torch.cuda.is_available()
            else []
        ),
        help="Device to run the model on. Default = 0 if CUDA is available, otherwise 'cpu'.",
    )

    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    """Check the arguments for the pipeline."""
    # Check the base arguments
    check_base_args(args)

    # Epochs, early stopping, and learning rate must be positive
    for attr in ("n_epochs", "early_stopping", "lr"):
        if getattr(args, attr) <= 0:
            raise ValueError(f"{attr} must be greater than 0.")


def run_approximate_map(args: argparse.Namespace) -> None:
    """Run the approximate MAP pipeline."""
    # Prepare the run
    model = prep_run(args)

    # Run map approximation
    map_ = model.approximate_map(
        epochs=args.n_epochs,
        early_stop=args.early_stopping,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )

    # Draw samples from the MAP and save them
    samples = map_.get_inference_obj(seed=args.seed, batch_size=1)
    samples.save_netcdf(
        os.path.join(
            args.output_dir,
            f"{args.dataset}_{args.subset}_{args.rate_dist}_{args.growth_func}_map.nc",
        )
    )


def main():
    """Main function to run the pipeline."""
    # Parse the arguments
    args = parse_args()

    # Check the arguments
    check_args(args)

    # Run the pipeline
    run_approximate_map(args)


if __name__ == "__main__":
    main()
