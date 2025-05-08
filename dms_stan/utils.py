"""Utility functions for the DMS Stan package."""

import glob
import os.path

from typing import Any, overload

import numpy as np
import numpy.typing as npt
import torch

from cmdstanpy.cmdstan_args import CmdStanArgs, SamplerArgs
from cmdstanpy.stanfit import CmdStanMCMC, RunSet
from cmdstanpy.utils import check_sampler_csv, scan_config


@overload
def _get_module(exponent: npt.NDArray[np.floating]) -> np: ...


@overload
def _get_module(exponent: torch.Tensor) -> torch: ...


def _get_module(exponent):
    """
    Get the module (numpy or torch) based on the type of the input.
    """
    return np if isinstance(exponent, np.ndarray) else torch


@overload
def stable_sigmoid(exponent: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...


@overload
def stable_sigmoid(exponent: torch.Tensor) -> torch.Tensor: ...


def stable_sigmoid(exponent):
    """
    Stable sigmoid function to avoid overflow.
    """
    # Are we working with torch or numpy?
    module = _get_module(exponent)

    # Empty array to store the results
    sigma_exponent = module.full_like(exponent, module.nan)

    # Different approach for positive and negative values
    mask = exponent >= 0

    # Calculate the sigmoid function for the positives
    pos_calc = module.exp(exponent[~mask])
    sigma_exponent[~mask] = pos_calc / (1 + pos_calc)

    # Calculate the sigmoid function for the negatives
    sigma_exponent[mask] = 1 / (1 + module.exp(-exponent[mask]))

    # We should have no NaN values in the result
    assert not module.any(module.isnan(sigma_exponent))
    return sigma_exponent


def from_csv_noload(path: str | list[str] | os.PathLike) -> CmdStanMCMC:
    """
    Parses the output files from Stan. This is derived from `cmdstanpy.from_csv`,
    and performs the same function, but stops short of loading the data into memory.
    This is useful for large datasets that need to be processed in chunks.
    """

    def identify_files() -> list[str]:
        """Identifies CSV files from the given path."""
        csvfiles = []
        if isinstance(path, list):
            csvfiles = path
        elif isinstance(path, str) and "*" in path:
            splits = os.path.split(path)
            if splits[0] is not None:
                if not (os.path.exists(splits[0]) and os.path.isdir(splits[0])):
                    raise ValueError(
                        f"Invalid path specification, {path} unknown directory: {splits[0]}"
                    )
            csvfiles = glob.glob(path)
        elif isinstance(path, (str, os.PathLike)):
            if os.path.exists(path) and os.path.isdir(path):
                for file in os.listdir(path):
                    if os.path.splitext(file)[1] == ".csv":
                        csvfiles.append(os.path.join(path, file))
            elif os.path.exists(path):
                csvfiles.append(str(path))
            else:
                raise ValueError(f"Invalid path specification: {path}")
        else:
            raise ValueError(f"Invalid path specification: {path}")

        if len(csvfiles) == 0:
            raise ValueError(f"No CSV files found in directory {path}")
        for file in csvfiles:
            if not (os.path.exists(file) and os.path.splitext(file)[1] == ".csv"):
                raise ValueError(
                    f"Bad CSV file path spec, includes non-csv file: {file}"
                )

        return csvfiles

    def get_config_dict() -> dict[str, Any]:
        """Reads the first CSV file and returns the configuration dictionary."""
        config_dict: dict[str, Any] = {}
        try:
            with open(csvfiles[0], "r", encoding="utf-8") as fd:
                scan_config(fd, config_dict, 0)
        except (IOError, OSError, PermissionError) as e:
            raise ValueError(f"Cannot read CSV file: {csvfiles[0]}") from e
        if "model" not in config_dict or "method" not in config_dict:
            raise ValueError(f"File {csvfiles[0]} is not a Stan CSV file.")
        if config_dict["method"] != "sample":
            raise ValueError(
                "Expecting Stan CSV output files from method sample, "
                f" found outputs from method {config_dict["method"]}"
            )

        return config_dict

    def build_sampler_args() -> SamplerArgs:
        """Builds the sampler arguments"""
        sampler_args = SamplerArgs(
            iter_sampling=config_dict["num_samples"],
            iter_warmup=config_dict["num_warmup"],
            thin=config_dict["thin"],
            save_warmup=config_dict["save_warmup"],
        )
        # bugfix 425, check for fixed_params output
        try:
            check_sampler_csv(
                csvfiles[0],
                iter_sampling=config_dict["num_samples"],
                iter_warmup=config_dict["num_warmup"],
                thin=config_dict["thin"],
                save_warmup=config_dict["save_warmup"],
            )
        except ValueError:
            try:
                check_sampler_csv(
                    csvfiles[0],
                    is_fixed_param=True,
                    iter_sampling=config_dict["num_samples"],
                    iter_warmup=config_dict["num_warmup"],
                    thin=config_dict["thin"],
                    save_warmup=config_dict["save_warmup"],
                )
                sampler_args = SamplerArgs(
                    iter_sampling=config_dict["num_samples"],
                    iter_warmup=config_dict["num_warmup"],
                    thin=config_dict["thin"],
                    save_warmup=config_dict["save_warmup"],
                    fixed_param=True,
                )
            except ValueError as e:
                raise ValueError("Invalid or corrupt Stan CSV output file, ") from e

        return sampler_args

    def build_fit() -> CmdStanMCMC:
        """Builds the CmdStanMCMC object"""
        chains = len(csvfiles)
        cmdstan_args = CmdStanArgs(
            model_name=config_dict["model"],
            model_exe=config_dict["model"],
            chain_ids=[x + 1 for x in range(chains)],
            method_args=sampler_args,
        )
        runset = RunSet(args=cmdstan_args, chains=chains)
        # pylint: disable=protected-access
        runset._csv_files = csvfiles
        for i in range(len(runset._retcodes)):
            runset._set_retcode(i, 0)
        # pylint: enable=protected-access
        fit = CmdStanMCMC(runset)
        return fit

    # Run the functions to parse the CSV files
    csvfiles = identify_files()
    config_dict = get_config_dict()
    sampler_args = build_sampler_args()
    return build_fit()
