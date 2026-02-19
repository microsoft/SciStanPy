"""Defines default values for the hyperparameters set in the FLIP models."""

from typing import Literal

# Default values for hyperparameters
DEFAULT_HYPERPARAMS = {
    "alpha": 0.1,
    "exp_beta": 10.0,
    "lambda_": 1.0,
    "lomax_alpha": 2.0,
    "c_alpha": 4.0,
    "c_beta": 8.0,
    "r_sigma": 0.1,
    "inv_r_alpha": 2.0,
    "inv_r_beta": 0.5,
    "codon_noise_alpha": 2.0,
    "codon_noise_beta": 2.0,
    "experimental_noise_sigma": 0.5,
}

# Types for model options
GrowthCurve = Literal["exponential", "sigmoid"]
GrowthRate = Literal["lomax", "exponential", "gamma"]
