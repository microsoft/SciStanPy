"""Same layout as `mcmc_flip.py` but with the `mle` pipeline."""

import argparse
import os.path

import numpy as np
import torch

from scistanpy.defaults import DEFAULT_EARLY_STOP, DEFAULT_LR, DEFAULT_N_EPOCHS
from scistanpy.pipelines.mcmc_flip import check_base_args, define_base_parser, prep_run


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Build the parser specifically for this pipeline
    parser = argparse.ArgumentParser(
        description="Maximum likelihood estimation for FLIP datasets.",
        parents=[define_base_parser()],
    )

    # Add arguments specific to this pipeline
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=DEFAULT_N_EPOCHS,
        help=(
            "Maximum number of passes over the data during optimization. "
            f"Default = {DEFAULT_N_EPOCHS}."
        ),
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=DEFAULT_EARLY_STOP,
        help=(
            "Number of epochs to wait before stopping if the loss does not improve. "
            f"Default = {DEFAULT_EARLY_STOP}."
        ),
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
        help=(
            "Device to run the model on. Default = "
            f"{'0' if torch.cuda.is_available() else 'cpu'}."
        ),
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="Batch size for sampling. Default = 1.",
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


def run_mle(args: argparse.Namespace) -> None:
    """Run the approximate MLE pipeline."""
    # Prepare the run
    model = prep_run(args)

    # Run mle approximation
    mle = model.mle(
        epochs=args.n_epochs,
        early_stop=args.early_stopping,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )

    # Save the loss curve
    base_outfile = f"{args.dataset}_{args.subset}_{args.rate_dist}_{args.growth_func}"
    mle.losses.to_csv(
        os.path.join(args.output_dir, f"{base_outfile}_loss-curve.csv"), index=False
    )

    # Save the MLE values
    np.savez(
        os.path.join(args.output_dir, f"{base_outfile}_mle.npz"),
        **{k: v.mle for k, v in mle.model_varname_to_mle.items() if v.mle is not None},
    )

    # Draw samples from the MLE and save them
    samples = mle.get_inference_obj(seed=args.seed, batch_size=args.sample_batch_size)
    samples.save_netcdf(os.path.join(args.output_dir, f"{base_outfile}_samples.nc"))


def main():
    """Main function to run the pipeline."""
    # Parse the arguments
    args = parse_args()

    # Check the arguments
    check_args(args)

    # Run the pipeline
    run_mle(args)


if __name__ == "__main__":
    main()
