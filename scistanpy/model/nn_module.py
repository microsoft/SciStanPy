# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""PyTorch integration utilities for SciStanPy models.

This module provides integration between SciStanPy probabilistic models
and PyTorch's automatic differentiation and optimization framework. It enables
maximum likelihood estimation, variational inference, and other gradient-based
learning procedures on SciStanPy models.

The module's core functionality centers around converting SciStanPy models into
PyTorch nn.Module instances that preserve the probabilistic structure while
enabling efficient gradient computation and optimization. This allows users to
leverage PyTorch's ecosystem of optimizers, learning rate schedulers, and other
training utilities.

Key Features:
    - Automatic conversion of SciStanPy models to PyTorch modules
    - Gradient-based parameter optimization with various optimizers
    - Mixed precision training support for improved performance
    - Early stopping and convergence monitoring
    - GPU acceleration and device management

The module handles the complex details of parameter initialization, gradient
computation, and device management, providing a simple interface for fitting
Bayesian models using modern deep learning techniques.

Performance Considerations:
    - GPU acceleration significantly improves training speed for large models
"""

import warnings

from typing import Optional, TYPE_CHECKING, Union

import itertools
import numpy.typing as npt
import torch
import torch.nn as nn

from tqdm import tqdm

from scistanpy.defaults import DEFAULT_EARLY_STOP, DEFAULT_LR, DEFAULT_N_EPOCHS
from scistanpy.model.components import constants, parameters

if TYPE_CHECKING:
    from scistanpy import custom_types
    from scistanpy import model as ssp_model


def check_observable_data(model: "ssp_model.Model", data: dict[str, torch.Tensor]):
    """Validate that provided data matches model observable specifications.

    This function performs comprehensive validation to ensure that the observed
    data dictionary contains exactly the expected observables with correct
    shapes and types. It prevents common errors during model fitting by
    catching data mismatches early.

    :param model: SciStanPy model containing observable specifications
    :type model: ssp_model.Model
    :param data: Dictionary mapping observable names to their tensor data
    :type data: dict[str, torch.Tensor]

    :raises ValueError: If observable names don't match expected set
    :raises ValueError: If data shapes don't match observable shapes

    The validation checks:
    - Perfect correspondence between provided and expected observable names
    - Exact shape matching between data tensors and observable specifications
    - Proper tensor formatting for PyTorch computation

    Example:
        >>> data = {'y': torch.randn(100), 'x': torch.randn(100, 5)}
        >>> check_observable_data(model, data)  # Validates or raises error
    """
    # There must be perfect overlap between the keys of the provided data and the
    # expected observations
    expected_set = set(model.observable_dict.keys())
    provided_set = set(data.keys())
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
        if data[name].shape != param.shape:
            raise ValueError(
                f"The shape of the provided data for observable {name} does not match "
                f"the expected shape. Expected: {param.shape}, provided: "
                f"{data[name].shape}"
            )


class PyTorchModel(nn.Module):
    """PyTorch-trainable version of a SciStanPy Model.

    This class converts SciStanPy probabilistic models into PyTorch nn.Module
    instances that can be optimized using standard PyTorch training procedures.
    It preserves the probabilistic structure while enabling gradient-based
    parameter estimation and other machine learning techniques.

    :param model: SciStanPy model to convert to PyTorch
    :type model: ssp_model.Model
    :param seed: Random seed for reproducible parameter initialization. Defaults to None.
    :type seed: Optional[custom_types.Integer]

    :ivar model: Reference to the original SciStanPy model
    :ivar learnable_params: PyTorch ParameterList containing optimizable parameters

    The conversion process:
    - Initializes all model parameters for PyTorch optimization
    - Sets up proper gradient computation graphs
    - Configures device placement and memory management
    - Preserves probabilistic model structure and relationships

    The resulting PyTorch model can be:
    - Optimized using any PyTorch optimizer
    - Moved between devices (CPU/GPU)
    - Integrated with PyTorch training pipelines
    - Used for both maximum likelihood and variational inference

    Note:
        This class should not be instantiated directly. Instead, use the
        `to_pytorch()` method on a SciStanPy Model instance.

    Example:
        >>> pytorch_model = model.to_pytorch(seed=42)
        >>> optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.01)
        >>> loss = -pytorch_model(**observed_data)
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self, model: "ssp_model.Model", seed: Optional["custom_types.Integer"] = None
    ):
        """
        Args:
            model: The `scistanpy.model.Model` instance to convert to PyTorch.
        """
        super().__init__()

        # Record the model
        self.model = model

        # Initialize all parameters for pytorch optimization
        learnable_params = []
        for param_num, param in enumerate(self.model.parameters):
            param.init_pytorch(seed=None if seed is None else seed + param_num)
            learnable_params.append(param._torch_parametrization)

        # Record learnable parameters such that they can be recognized by PyTorch
        self.learnable_params = nn.ParameterList(learnable_params)

    def forward(self, **data: torch.Tensor) -> torch.Tensor:
        """Compute log probability of observed data given current parameters.

        This method calculates the total log probability (log-likelihood) of
        the observed data under the current model parameters. It forms the
        core objective function for maximum likelihood estimation and other
        gradient-based inference procedures.

        :param data: Observed data tensors keyed by observable parameter names
        :type data: dict[str, torch.Tensor]

        :returns: Total log probability of the observed data
        :rtype: torch.Tensor

        The computation includes:
        - Log probabilities of all observable parameters given data
        - Log probabilities of all latent parameters given their priors
        - Proper handling of different distribution types and shapes
        - Gradient computation for backpropagation

        Note:
            This returns log probability, *not* log loss (negative log probability).
            For optimization, negate the result to get the loss function.

        Example:
            >>> log_prob = pytorch_model(y=observed_y, x=observed_x)
            >>> loss = -log_prob  # Negative for minimization
            >>> loss.backward()
        """
        # Sum the log-probs of the observables and parameters
        log_prob = 0.0
        for name, param in itertools.chain(
            self.model.parameter_dict.items(), self.model.observable_dict.items()
        ):
            # Calculate the log probability of the observed data given the parameters
            temp_log_prob = param.get_torch_logprob(observed=data.get(name))

            # Log probability should be 0-dimensional if anything but a Multinomial
            assert temp_log_prob.ndim == 0 or isinstance(param, parameters.Multinomial)

            # Add to the total log probability
            log_prob += temp_log_prob.sum()

        return log_prob

    def fit(
        self,
        *,
        epochs: "custom_types.Integer" = DEFAULT_N_EPOCHS,
        early_stop: "custom_types.Integer" = DEFAULT_EARLY_STOP,
        lr: "custom_types.Float" = DEFAULT_LR,
        data: dict[
            str,
            Union[
                torch.Tensor, npt.NDArray, "custom_types.Float", "custom_types.Integer"
            ],
        ],
        mixed_precision: bool = False,
    ) -> torch.Tensor:
        """Optimize model parameters using gradient-based maximum likelihood estimation.

        This method performs complete model training using the Adam optimizer
        with configurable early stopping, learning rate, and mixed precision
        support. It automatically handles device placement, gradient computation,
        and convergence monitoring.

        :param epochs: Maximum number of training epochs. Defaults to 100000.
        :type epochs: custom_types.Integer
        :param early_stop: Epochs without improvement before stopping. Defaults to 10.
        :type early_stop: custom_types.Integer
        :param lr: Learning rate for Adam optimizer. Defaults to 0.001.
        :type lr: custom_types.Float
        :param data: Observed data for model observables
        :type data: dict[str, Union[torch.Tensor, npt.NDArray, custom_types.Float,
            custom_types.Integer]]
        :param mixed_precision: Whether to use automatic mixed precision. Defaults to False.
        :type mixed_precision: bool

        :returns: Tensor containing loss trajectory throughout training
        :rtype: torch.Tensor

        :raises UserWarning: If early stopping is not triggered within epoch limit

        Training Features:
        - Adam optimization with configurable learning rate
        - Early stopping based on loss plateau detection
        - Mixed precision training for memory efficiency and speed
        - Progress monitoring with real-time loss display
        - Automatic device placement and tensor conversion
        - Convergence tracking

        The training loop:
        1. Converts input data to appropriate tensor format
        2. Validates data compatibility with model observables
        3. Iteratively optimizes parameters using gradient descent
        4. Monitors convergence and applies early stopping
        5. Returns complete loss trajectory for analysis

        Example:
            >>> loss_history = pytorch_model.fit(
            ...     data={'y': observed_data},
            ...     epochs=5000,
            ...     lr=0.01,
            ...     early_stop=50,
            ...     mixed_precision=True
            ... )
            >>> final_loss = loss_history[-1]
        """
        # Any observed data that is not a tensor is converted to a tensor
        data = {
            k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

        # Note the device
        device = self.learnable_params[0].device

        # Check the observed data
        check_observable_data(self.model, data)

        # Train mode. This should be a null-op.
        self.train()

        # Build the optimizer
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        # If using mixed precision, we also need a scaler
        if mixed_precision:
            scaler = torch.amp.GradScaler()

        # Set up for optimization
        best_loss = float("inf")  # Records the best loss
        loss_trajectory = [None] * (epochs + 1)  # Records all losses
        n_without_improvement = 0  # Epochs without improvement

        # Run optimization
        with tqdm(total=epochs, desc="Epochs", postfix={"-log pdf/pmf": "N/A"}) as pbar:
            for epoch in range(epochs):

                # Get the loss
                with torch.autocast(device_type=device.type, enabled=mixed_precision):
                    log_loss = -1 * self(**data)

                # Step the optimizer
                optim.zero_grad()
                if mixed_precision:
                    scaler.scale(log_loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
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
                pbar.set_postfix({"-log pdf/pmf": f"{log_loss:.2f}"})

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
            loss_trajectory[epoch + 1] = -1 * self(**data).item()

        # Trim off the None values of the loss trajectory and convert to a tensor
        return torch.tensor(loss_trajectory[: epoch + 2], dtype=torch.float32)

    def export_params(self) -> dict[str, torch.Tensor]:
        """Export optimized parameter values from the fitted model.

        This method extracts the current parameter values after optimization,
        providing access to the maximum likelihood estimates or other fitted
        parameter values. It excludes observable parameters (which represent
        data) and focuses on the learnable model parameters.

        :returns: Dictionary mapping parameter names to their current tensor values
        :rtype: dict[str, torch.Tensor]

        Excluded from export:
        - Observable parameters (representing data, not learnable parameters)
        - Unnamed parameters
        - Intermediate computational results from transformations

        This is typically used after model fitting to extract the estimated
        parameter values for further analysis or model comparison.

        Example:
            >>> fitted_params = pytorch_model.export_params()
            >>> mu_estimate = fitted_params['mu']
            >>> sigma_estimate = fitted_params['sigma']
        """
        return {
            name: param.torch_parametrization
            for name, param in self.model.parameter_dict.items()
        }

    def export_distributions(self) -> dict[str, torch.distributions.Distribution]:
        """Export fitted probability distributions for all model components.

        This method returns the complete set of probability distributions
        from the fitted model, including both parameter distributions (priors)
        and observable distributions (likelihoods) with their current
        parameter values.

        :returns: Dictionary mapping component names to their distribution objects
        :rtype: dict[str, torch.distributions.Distribution]

        The exported distributions include:
        - Parameter distributions with updated hyperparameter values
        - Observable distributions with fitted parameter values
        - All distributions in their PyTorch format for further computation

        This is useful for:
        - Posterior predictive sampling
        - Model diagnostics and validation
        - Uncertainty quantification
        - Distribution comparison and analysis

        Example:
            >>> distributions = pytorch_model.export_distributions()
            >>> fitted_normal = distributions['mu']  # torch.distributions.Normal
            >>> samples = fitted_normal.sample((1000,))  # Sample from fit distribution
        """
        return {
            name: param.torch_dist_instance
            for name, param in itertools.chain(
                self.model.parameter_dict.items(), self.model.observable_dict.items()
            )
        }

    def _move_model(self, funcname: str, *args, **kwargs):
        """Internal method for device placement operations.

        This method handles the task of moving both PyTorch parameters
        and SciStanPy constant tensors to different devices or data types.
        It ensures that all model components remain synchronized during
        device transfers.

        :param funcname: Name of the PyTorch method to apply ('cuda', 'cpu', 'to')
        :type funcname: str
        :param args: Positional arguments for the device operation
        :param kwargs: Keyword arguments for the device operation

        :returns: Self reference for method chaining
        :rtype: PyTorchModel
        """
        # Apply to the model
        getattr(super(), funcname)(*args, **kwargs)

        # Apply to additional torch tensors in the model (i.e., the ones that are
        # constants and not parameters)
        # pylint: disable=protected-access
        for constant in filter(
            lambda x: isinstance(x, constants.Constant),
            self.model.all_model_components,
        ):
            constant._torch_parametrization = getattr(
                constant._torch_parametrization, funcname
            )(*args, **kwargs)

        return self

    def cuda(self, *args, **kwargs):
        """Move model to CUDA device.

        This method transfers the entire model (including SciStanPy constants)
        to a CUDA-enabled GPU device for accelerated computation.

        :param args: Arguments passed to torch.nn.Module.cuda()
        :param kwargs: Keyword arguments passed to torch.nn.Module.cuda()

        :returns: Self reference for method chaining
        :rtype: PyTorchModel

        Example:
            >>> pytorch_model = pytorch_model.cuda()  # Move to default GPU
            >>> pytorch_model = pytorch_model.cuda(1)  # Move to GPU 1
        """
        return self._move_model("cuda", *args, **kwargs)

    def cpu(self, *args, **kwargs):
        """Move model to CPU device.

        This method transfers the entire model (including SciStanPy constants)
        back to CPU memory, useful for inference or when GPU memory is limited.

        :param args: Arguments passed to torch.nn.Module.cpu()
        :param kwargs: Keyword arguments passed to torch.nn.Module.cpu()

        :returns: Self reference for method chaining
        :rtype: PyTorchModel

        Example:
            >>> pytorch_model = pytorch_model.cpu()  # Move to CPU
        """
        return self._move_model("cpu", *args, **kwargs)

    def to(self, *args, **kwargs):
        """Move model to specified device or data type.

        This method provides flexible device and dtype conversion for the
        entire model, including both PyTorch parameters and SciStanPy
        constant tensors.

        :param args: Arguments passed to torch.nn.Module.to()
        :param kwargs: Keyword arguments passed to torch.nn.Module.to()

        :returns: Self reference for method chaining
        :rtype: PyTorchModel

        Example:
            >>> pytorch_model = pytorch_model.to('cuda:0')  # Move to specific GPU
            >>> pytorch_model = pytorch_model.to(torch.float64)  # Change precision
            >>> pytorch_model = pytorch_model.to('cpu', dtype=torch.float32)
        """
        return self._move_model("to", *args, **kwargs)
