"""Holds utilities for integrating DMS Stan with Pytorch"""

import warnings

from typing import Union

import numpy.typing as npt
import torch
import torch.nn as nn

from tqdm import tqdm

import dms_stan.model as dms

from dms_stan.defaults import DEFAULT_EARLY_STOP, DEFAULT_LR, DEFAULT_N_EPOCHS
from .components import Parameter


def check_observable_data(
    model: "dms.model.Model", observed_data: dict[str, torch.Tensor]
):
    """Makes sure that the correct observables are provided for a givne model."""
    # There must be perfect overlap between the keys of the provided data and the
    # expected observations
    expected_set = set(model.observable_dict.keys())
    provided_set = set(observed_data.keys())
    missing = expected_set - provided_set
    extra = provided_set - expected_set

    # If there are missing or extra, raise an error
    if missing:
        raise ValueError(
            "The provided data must match the observable distribution names."
            f"The following observables are missing: {', '.join(missing)}"
        )
    if extra:
        raise ValueError(
            "The provided data must match the observable distribution names. The "
            "following observables were provided in addition to the expected: "
            f"{', '.join(extra)}"
        )

    # Shapes must match
    for name, param in model.observable_dict.items():
        if observed_data[name].shape != param.shape:
            raise ValueError(
                f"The shape of the provided data for observable {name} does not match "
                f"the expected shape. Expected: {param.shape}, provided: "
                f"{observed_data[name].shape}"
            )


class PyTorchModel(nn.Module):
    """
    A PyTorch-trainable version of a `dms_stan.model.Model`. This should not be
    used directly, but instead accessed by calling the `to_pytorch` method on a
    `dms_stan.model.Model` instance.
    """

    def __init__(self, model: "dms.model.Model"):
        """
        Args:
            model: The `dms_stan.model.Model` instance to convert to PyTorch.
        """
        super().__init__()

        # Record the model
        self.model = model

        # Record learnable parameters such that they can be recognized by PyTorch
        self.torch_params = nn.ParameterList(self.model._torch_parameters)

    def forward(self, **observed_data: torch.Tensor) -> torch.Tensor:
        """
        Each observation is passed in as a keyword argument whose name matches the
        name of the corresponding observable distribution in the `dms_stan.model.Model`
        instance. This will calculate the log probability of the observed data given
        the parameters of the model. Stress: this is the log-probability, not the
        log loss, which is the negative log-probability.
        """
        # Sum the log-probs of the observables and parameters
        log_prob = 0.0
        for name, param in self.model.parameter_dict.items():

            # Ignore anything that is not a Parameter instance. When we call
            # `calculate_log_prob` on a parameter, any transformations needed to
            # calculate the log-probability are handled internally (i.e., the
            # transformed parameters are included)
            if not isinstance(param, Parameter):
                continue

            # Calculate the log-probability of the parameter
            log_prob += param.calculate_log_prob(observed=observed_data.get(name, None))

        return log_prob

    def fit(
        self,
        epochs: int = DEFAULT_N_EPOCHS,
        early_stop: int = DEFAULT_EARLY_STOP,
        lr: float = DEFAULT_LR,
        **observed_data: Union[torch.Tensor, npt.NDArray, float, int],
    ) -> torch.Tensor:
        """Optimizes the parameters of the model."""
        # Any observed data that is not a tensor is converted to a tensor
        observed_data = {
            k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            for k, v in observed_data.items()
        }

        # Check the observed data
        check_observable_data(self.model, observed_data)

        # Train mode. This should be a null-op.
        self.train()

        # Build the optimizer
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        # Set up for optimization
        best_loss = float("inf")  # Records the best loss
        loss_trajectory = [None] * (epochs + 1)  # Records all losses
        n_without_improvement = 0  # Epochs without improvement

        # Run optimization
        with tqdm(total=epochs, desc="Epochs", postfix={"loss": "N/A"}) as pbar:
            for epoch in range(epochs):

                # Get the loss
                log_loss = -1 * self(**observed_data)

                # Step the optimizer
                optim.zero_grad()
                log_loss.backward()
                optim.step()

                # Record loss
                log_loss = log_loss.item()
                loss_trajectory[epoch] = log_loss

                # Update best loss
                if log_loss < best_loss:
                    n_without_improvement = 0
                    best_loss = log_loss
                else:
                    n_without_improvement += 1

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"loss": f"{log_loss:.2f}"})

                # Check for early stopping
                if early_stop > 0 and n_without_improvement >= early_stop:
                    break

            # Note that early stopping was not triggered if the loop completes
            else:
                if early_stop > 0:
                    warnings.warn("Early stopping not triggered.")

        # Back to eval mode
        self.eval()

        # Get a final loss
        with torch.no_grad():
            loss_trajectory[epoch + 1] = -1 * self(**observed_data).item()

        # Trim off the None values of the loss trajectory and convert to a tensor
        return torch.tensor(loss_trajectory[: epoch + 2], dtype=torch.float32)

    def export_params(self) -> dict[str, torch.Tensor]:
        """
        Exports all parameters from the DMS Stan model used to construct the PyTorch
        model. This is primarily used once the model has been fit. Note that we
        do not gather observations for parameters that are marked as observables
        (as this is just the data). We also do not gather "observables" for transformations
        as they are implicit in the parameters.
        """
        return {
            name: param.get_torch_observables()
            for name, param in self.model.parameter_dict.items()
            if isinstance(param, Parameter) and not param.observable
        }

    def export_distributions(self) -> dict[str, torch.distributions.Distribution]:
        """
        Exports all distributions from the DMS Stan model used to construct the
        PyTorch model. This is primarily used once the model has been fit.
        """
        return {
            name: param.torch_dist_instance
            for name, param in self.model.parameter_dict.items()
            if isinstance(param, Parameter)
        }
