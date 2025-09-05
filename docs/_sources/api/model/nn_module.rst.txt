Neural Network Module API Reference
===================================

This reference covers the PyTorch neural network integration for SciStanPy models.

Neural Network Module
---------------------

.. automodule:: scistanpy.model.nn_module
   :undoc-members:
   :show-inheritance:

Core PyTorch Integration
------------------------

.. autoclass:: scistanpy.model.nn_module.PyTorchModel
   :members:
   :undoc-members:
   :show-inheritance:

   **Creating PyTorch Models:**

   .. code-block:: python

      import scistanpy as ssp
      import torch

      # Define SciStanPy model
      class MyModel(ssp.Model):
          def __init__(self):
              super().__init__()
              self.mu = ssp.parameters.Normal(mu=0, sigma=1)
              self.sigma = ssp.parameters.LogNormal(mu=0, sigma=0.5)
              self.y = ssp.parameters.Normal(mu=self.mu, sigma=self.sigma, observable=True)

      # Convert to PyTorch
      model = MyModel()
      pytorch_model = model.to_pytorch(seed=42)

   **Training with PyTorch Optimizers:**

   .. code-block:: python

      # Manual optimization loop
      optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.01)

      for epoch in range(1000):
          optimizer.zero_grad()
          log_prob = pytorch_model(y=observed_data)
          loss = -log_prob  # Negative log-likelihood
          loss.backward()
          optimizer.step()

   **Automated Training:**

   .. code-block:: python

      # Use built-in training method
      loss_history = pytorch_model.fit(
          data={'y': observed_data},
          epochs=5000,
          lr=0.01,
          early_stop=50,
          mixed_precision=True
      )

   **GPU Acceleration:**

   .. code-block:: python

      # Move model to GPU
      pytorch_model = pytorch_model.cuda()

      # Or specify device explicitly
      pytorch_model = pytorch_model.to('cuda:1')

      # GPU-accelerated training
      loss_history = pytorch_model.fit(
          data={'y': torch.tensor(observed_data).cuda()},
          epochs=10000,
          mixed_precision=True
      )

Validation Functions
--------------------

.. autofunction:: scistanpy.model.nn_module.check_observable_data
   :noindex:

   **Data Validation Example:**

   .. code-block:: python

      # Prepare data dictionary
      data = {
          'y': torch.randn(100),
          'x': torch.randn(100, 5)
      }

      # Validate against model expectations
      try:
          check_observable_data(model, data)
          print("Data validation passed")
      except ValueError as e:
          print(f"Data validation failed: {e}")

Training and Optimization
-------------------------

**Advanced Training Options:**

.. code-block:: python

   # Training with custom settings
   loss_trajectory = pytorch_model.fit(
       data=training_data,
       epochs=20000,           # Maximum epochs
       early_stop=100,         # Stop if no improvement for 100 epochs
       lr=0.005,              # Learning rate
       mixed_precision=True    # Use automatic mixed precision
   )

   # Check convergence
   final_loss = loss_trajectory[-1]
   print(f"Final loss: {final_loss:.4f}")

**Custom Optimization Strategies:**

.. code-block:: python

   # Advanced optimizer configuration
   optimizer = torch.optim.AdamW(
       pytorch_model.parameters(),
       lr=0.01,
       weight_decay=1e-4
   )

   # Learning rate scheduling
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, patience=50, factor=0.5
   )

   # Custom training loop with scheduling
   for epoch in range(max_epochs):
       optimizer.zero_grad()
       loss = -pytorch_model(**data)
       loss.backward()
       optimizer.step()
       scheduler.step(loss)

Parameter and Distribution Export
---------------------------------

**Extracting Fitted Parameters:**

.. code-block:: python

   # Get optimized parameter values
   fitted_params = pytorch_model.export_params()

   for name, value in fitted_params.items():
       print(f"{name}: {value.detach().cpu().numpy()}")

**Accessing Fitted Distributions:**

.. code-block:: python

   # Get fitted probability distributions
   distributions = pytorch_model.export_distributions()

   # Use for posterior predictive sampling
   fitted_normal = distributions['mu']
   posterior_samples = fitted_normal.sample((1000,))

Device Management
-----------------

**Multi-GPU Training:**

.. code-block:: python

   # Check GPU availability
   if torch.cuda.is_available():
       device = f'cuda:{torch.cuda.current_device()}'
       pytorch_model = pytorch_model.to(device)

   # Data parallel training for large models
   if torch.cuda.device_count() > 1:
       pytorch_model = torch.nn.DataParallel(pytorch_model)

**Memory-Efficient Training:**

.. code-block:: python

   # Mixed precision for memory efficiency
   pytorch_model = pytorch_model.half()  # Use FP16

   # Gradient accumulation for large effective batch sizes
   accumulation_steps = 4

   for i, batch in enumerate(data_loader):
       loss = -pytorch_model(**batch) / accumulation_steps
       loss.backward()

       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()

Integration Patterns
--------------------

**SciStanPy to PyTorch Workflow:**

.. code-block:: python

   # Complete workflow example
   def pytorch_fitting_workflow(model, observed_data):
       """Complete PyTorch fitting workflow."""

       # 1. Convert to PyTorch
       pytorch_model = model.to_pytorch(seed=42)

       # 2. Move to GPU if available
       device = 'cuda' if torch.cuda.is_available() else 'cpu'
       pytorch_model = pytorch_model.to(device)

       # 3. Prepare data
       data = {k: torch.tensor(v).to(device) for k, v in observed_data.items()}

       # 4. Fit model
       loss_history = pytorch_model.fit(
           data=data,
           epochs=10000,
           lr=0.01,
           early_stop=100,
           mixed_precision=(device == 'cuda')
       )

       # 5. Extract results
       fitted_params = pytorch_model.export_params()
       fitted_distributions = pytorch_model.export_distributions()

       return {
           'loss_history': loss_history,
           'parameters': fitted_params,
           'distributions': fitted_distributions,
           'model': pytorch_model
       }

**Variational Inference:**

.. code-block:: python

   # Use PyTorch model for variational inference
   class VariationalModel(torch.nn.Module):
       def __init__(self, original_model):
           super().__init__()
           self.model = original_model.to_pytorch()

           # Variational parameters
           self.q_mu = torch.nn.Parameter(torch.zeros(1))
           self.q_log_sigma = torch.nn.Parameter(torch.zeros(1))

       def forward(self, data):
           # Sample from variational distribution
           epsilon = torch.randn_like(self.q_mu)
           z = self.q_mu + torch.exp(self.q_log_sigma) * epsilon

           # Compute ELBO
           log_likelihood = self.model(**data)
           kl_divergence = torch.distributions.kl_divergence(
               torch.distributions.Normal(self.q_mu, torch.exp(self.q_log_sigma)),
               torch.distributions.Normal(0, 1)
           )

           return log_likelihood - kl_divergence

Performance Considerations
--------------------------

**Optimization Tips:**

1. **Use mixed precision** on GPUs to reduce memory usage and increase speed
2. **Enable early stopping** to prevent overfitting and save computation
3. **Choose appropriate learning rates** - start with 0.001 and adjust based on convergence
4. **Monitor gradient norms** to detect vanishing/exploding gradients
5. **Use GPU acceleration** for models with many parameters or large datasets

**Memory Management:**

.. code-block:: python

   # Monitor memory usage
   def get_memory_usage():
       if torch.cuda.is_available():
           return torch.cuda.memory_allocated() / 1024**2  # MB
       return 0

   print(f"Memory before training: {get_memory_usage():.1f} MB")

   # Clear cache if needed
   if torch.cuda.is_available():
       torch.cuda.empty_cache()

Error Handling and Debugging
----------------------------

**Common Issues and Solutions:**

.. code-block:: python

   # Handle common training issues
   try:
       loss_history = pytorch_model.fit(data=data, epochs=10000)
   except RuntimeError as e:
       if "out of memory" in str(e):
           print("GPU out of memory - try reducing batch size or using CPU")
           pytorch_model = pytorch_model.cpu()
           data = {k: v.cpu() for k, v in data.items()}
       else:
           raise

**Gradient Debugging:**

.. code-block:: python

   # Check for gradient issues
   def check_gradients(model):
       total_norm = 0
       for p in model.parameters():
           if p.grad is not None:
               param_norm = p.grad.data.norm(2)
               total_norm += param_norm.item() ** 2
       total_norm = total_norm ** (1. / 2)
       return total_norm

   # Monitor during training
   grad_norm = check_gradients(pytorch_model)
   if grad_norm > 10:
       print("Warning: Large gradients detected")

Best Practices
--------------

1. **Validate data format** before training to catch shape mismatches early
2. **Use reproducible seeds** for debugging and development
3. **Monitor loss trajectories** to assess convergence quality
4. **Leverage GPU acceleration** for computational efficiency
5. **Export fitted parameters** for further analysis and model comparison
6. **Use early stopping** to prevent overfitting and reduce computation time
7. **Enable mixed precision** on modern GPUs for memory and speed benefits

The PyTorch integration enables efficient gradient-based optimization while preserving the probabilistic structure of SciStanPy models.
