"""Defines default values for the hyperparameters set in the FLIP models."""

from typing import Literal

# Default values for hyperparameters
DEFAULT_HYPERPARAMS = {
    "alpha": 0.1,
    "exp_beta": 1.0,
    "exp_beta_nuclease": 10.0,
    "lambda_": 1.0,
    "lomax_alpha": 1.0,
    "lomax_alpha_nuclease": 5.0,
    "c_alpha": 4.0,
    "c_beta": 8.0,
    "r_sigma_sigma": 0.01,
    "inv_r_alpha": 7.0,
    "inv_r_beta": 1.0,
    "alpha_alpha": 2.0,
    "alpha_beta": 2.0,
    "codon_noise_sigma": 0.1,
    "absolute_noise_sigma": 0.01,
}

# Types for model options
GrowthCurve = Literal["exponential", "sigmoid"]
GrowthRate = Literal["lomax", "exponential", "gamma"]
