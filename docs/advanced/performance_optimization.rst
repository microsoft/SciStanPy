Performance Optimization
========================

This guide covers techniques for optimizing SciStanPy model performance for large-scale scientific computing.

Backend Selection and Optimization
----------------------------------

**Choosing the Right Backend:**

.. code-block:: python

   # For exploration and prototyping
   model.set_backend('numpy')
   quick_results = model.mle()

   # For gradient-based optimization
   model.set_backend('pytorch')
   vi_results = model.variational()

   # For production MCMC
   model.set_backend('stan')
   final_results = model.sample()

**GPU Acceleration:**

.. code-block:: python

   # Enable GPU acceleration for PyTorch backend
   import torch

   if torch.cuda.is_available():
       device = torch.device('cuda')
       model.set_backend('pytorch', device=device)

       # Large-scale variational inference
       results = model.variational(n_samples=10000)

Memory Optimization
------------------

**Large Dataset Handling:**

.. code-block:: python

   class ChunkedInference:
       """Process large datasets in chunks."""

       def __init__(self, model, chunk_size=1000):
           self.model = model
           self.chunk_size = chunk_size

       def process_large_dataset(self, data):
           results = []
           for i in range(0, len(data), self.chunk_size):
               chunk = data[i:i+self.chunk_size]
               chunk_result = self.model.sample(data=chunk)
               results.append(chunk_result)

           return self.combine_results(results)

**Memory-Efficient Sampling:**

.. code-block:: python

   # Reduce memory usage during sampling
   results = model.sample(
       n_samples=1000,
       save_warmup=False,      # Don't save warmup samples
       thin=2,                 # Keep every 2nd sample
       output_dir='./cache'    # Stream to disk
   )

Computational Optimization
-------------------------

**Vectorization:**

.. code-block:: python

   # Efficient vectorized operations
   def vectorized_model_function(params, data):
       # Use array operations instead of loops
       return torch.sum(params * data, dim=-1)

   # Avoid explicit Python loops
   # result = [param * datum for param, datum in zip(params, data)]

**Parallel Processing:**

.. code-block:: python

   # Parallel MCMC chains
   results = model.sample(
       n_chains=8,
       parallel_chains=True,
       n_cores=8
   )

   # Distributed computing
   distributed_config = {
       'n_nodes': 4,
       'chains_per_node': 2
   }
   results = model.distributed_sample(config=distributed_config)

Model Optimization Techniques
----------------------------

**Reparameterization for Efficiency:**

.. code-block:: python

   # Non-centered parameterization for hierarchical models
   global_mean = ssp.parameters.Normal(mu=0, sigma=1)
   global_scale = ssp.parameters.LogNormal(mu=0, sigma=0.5)

   # More efficient than centered parameterization
   group_effects_raw = ssp.parameters.Normal(mu=0, sigma=1, shape=(n_groups,))
   group_effects = global_mean + global_scale * group_effects_raw

**Numerical Stability:**

.. code-block:: python

   # Use log-space operations for numerical stability
   log_probs = ssp.operations.log_softmax(logits)

   # Stable log-sum-exp operations
   log_sum = ssp.operations.log_sum_exp(log_values)

Profiling and Benchmarking
--------------------------

**Performance Profiling:**

.. code-block:: python

   import time

   # Profile model components
   profiler = ssp.utils.ModelProfiler(model)

   start_time = time.time()
   results = model.sample()
   total_time = time.time() - start_time

   print(f"Total sampling time: {total_time:.2f} seconds")

   # Identify bottlenecks
   bottlenecks = profiler.identify_bottlenecks()
   for component, time_spent in bottlenecks.items():
       print(f"{component}: {time_spent:.2f}s")

**Memory Profiling:**

.. code-block:: python

   import psutil
   import os

   def monitor_memory_usage():
       process = psutil.Process(os.getpid())
       memory_usage = process.memory_info().rss / 1024 / 1024  # MB
       return memory_usage

   # Monitor memory during inference
   initial_memory = monitor_memory_usage()
   results = model.sample()
   peak_memory = monitor_memory_usage()

   print(f"Memory usage: {peak_memory - initial_memory:.1f} MB")

Advanced Optimization Strategies
-------------------------------

**Model Compilation:**

.. code-block:: python

   # Compile model for efficiency
   compiled_model = ssp.utils.compile_model(
       model,
       optimization_level=3,
       cache_compiled=True
   )

   # Faster subsequent runs
   results = compiled_model.sample()

**Gradient Optimization:**

.. code-block:: python

   # Optimize gradient computation
   optimizer_config = {
       'learning_rate': 0.01,
       'momentum': 0.9,
       'adaptive_learning_rate': True
   }

   vi_results = model.variational(optimizer_config=optimizer_config)

Best Practices Summary
---------------------

1. **Choose appropriate backends** for different tasks
2. **Use GPU acceleration** for large-scale problems
3. **Implement chunking** for very large datasets
4. **Vectorize operations** whenever possible
5. **Monitor memory usage** and optimize accordingly
6. **Profile code** to identify bottlenecks
7. **Use efficient parameterizations** for better sampling
8. **Cache compiled models** for repeated use
