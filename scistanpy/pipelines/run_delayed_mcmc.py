"""Runs the `Model.mcmc` method using a delayed object file."""

import argparse

from scistanpy.model.model import run_delayed_mcmc


def main():
    """Runs the script."""
    # Build the argument parser
    parser = argparse.ArgumentParser(
        description="Runs the `Model.mcmc` method using a delayed object file."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the delayed object file.",
    )

    # Parse the arguments and run the script
    args = parser.parse_args()
    run_delayed_mcmc(args.filepath)


if __name__ == "__main__":
    main()
