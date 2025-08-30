Advanced Topics
===============

This guide covers advanced SciStanPy features for experienced users and complex modeling scenarios.

Custom Distribution Development
------------------------------

Creating Your Own Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SciStanPy allows you to create custom distributions for specialized scientific applications:

.. code-block:: python

   from scistanpy.model.components.transformations import UnaryTransformedParameter

   class TruncatedExponential(UnaryTransformedParameter):
       """Exponential distribution truncated at upper bound."""

       def __init__(self, rate, upper_bound):
           self.rate = rate
           self.upper_bound = upper_bound

           # Base exponential distribution
           base_dist = ssp.parameters.Exponential(rate=rate)
           super().__init__(base_dist)

       def run_np_torch_op(self, x):
           # Truncate at upper bound
           return torch.clamp(x, max=self.upper_bound)

       def write_stan_operation(self, x: str) -> str:
           return f"fmin({x}, {self.upper_bound})"

**Domain-Specific Distributions:**

.. code-block:: python

   class WeibullReliability(ssp.custom_distributions.CustomDistribution):
       """Weibull distribution for reliability analysis."""

       def __init__(self, shape, scale):
           self.shape = shape
           self.scale = scale

       def log_prob(self, value):
           # Weibull log-probability implementation
           return (torch.log(self.shape) - torch.log(self.scale) +
                   (self.shape - 1) * torch.log(value / self.scale) -
                   (value / self.scale) ** self.shape)

       def sample(self, sample_shape=torch.Size()):
           # Inverse transform sampling
           u = torch.rand(sample_shape)
           return self.scale * (-torch.log(1 - u)) ** (1 / self.shape)

Pruned Content
--------------

Removed unsupported features: variational inference, custom samplers, parallel /
distributed execution, GPU context managers, automatic backend switching APIs,
normalizing flows, advanced cross‑validation helpers, sensitivity frameworks,
automatic performance profilers.

Current Focus
-------------

- Structuring larger models via composition (parameters + arithmetic)
- Using mle() for quick prototyping before mcmc()
- Manual sensitivity: modify priors and re‑run

Example Pattern
---------------

.. code-block:: python

   base_results = model.mle(epochs=5000, early_stop=5)
   mcmc_results = model.mcmc(chains=4, iter_warmup=500, iter_sampling=1000)
   sample_failures, variable_failures = mcmc_results.diagnose()

Extensibility (Conceptual)
--------------------------

You can prototype a new distribution by wrapping existing parameter
components and combining them arithmetically; formal extension hooks
beyond this are future work.

Accuracy Note
-------------

Large sections referencing unimplemented APIs (variational(), backend
set_backend switching, distributed sampling, posterior predictive, LOO/WAIC,
normalizing flows, custom sampler classes) removed to reflect current scope.

This advanced topics guide provides the tools and techniques needed for sophisticated scientific modeling with SciStanPy.
   results = model.sample(config=stan_config)

Advanced Model Architectures
----------------------------

Hierarchical Model Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Varying Intercepts and Slopes:**

.. code-block:: python

   class HierarchicalRegression(ssp.Model):
       def __init__(self, groups, predictors, outcomes):
           n_groups = len(np.unique(groups))

           # Population-level parameters
           global_intercept = ssp.parameters.Normal(mu=0, sigma=1)
           global_slope = ssp.parameters.Normal(mu=0, sigma=1)

           # Group-level variance
           sigma_intercept = ssp.parameters.LogNormal(mu=0, sigma=0.5)
           sigma_slope = ssp.parameters.LogNormal(mu=0, sigma=0.5)

           # Correlated group effects
           correlation = ssp.parameters.LkjCorr(eta=2, dim=2)
           group_effects = ssp.parameters.MultivariateNormal(
               mu=torch.stack([global_intercept, global_slope]),
               correlation=correlation,
               scale=torch.stack([sigma_intercept, sigma_slope]),
               shape=(n_groups,)
           )

           # Predictions
           group_intercepts = group_effects[:, 0]
           group_slopes = group_effects[:, 1]

           predictions = (group_intercepts[groups] +
                         group_slopes[groups] * predictors)

           # Likelihood
           sigma = ssp.parameters.LogNormal(mu=0, sigma=0.5)
           self.likelihood = ssp.parameters.Normal(mu=predictions, sigma=sigma)
           self.likelihood.observe(outcomes)

**Mixture Models:**

.. code-block:: python

   class GaussianMixture(ssp.Model):
       def __init__(self, data, n_components=3):
           # Mixture weights
           weights = ssp.parameters.Dirichlet(alpha=torch.ones(n_components))

           # Component parameters
           means = ssp.parameters.Normal(mu=0, sigma=5, shape=(n_components,))
           stds = ssp.parameters.LogNormal(mu=0, sigma=1, shape=(n_components,))

           # Mixture distribution
           components = [
               ssp.parameters.Normal(mu=means[k], sigma=stds[k])
               for k in range(n_components)
           ]

           mixture = ssp.parameters.Mixture(
               weights=weights,
               components=components
           )

           mixture.observe(data)

Meta-Learning and Transfer Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Prior Learning from Related Datasets:**

.. code-block:: python

   class TransferLearningModel(ssp.Model):
       def __init__(self, source_results, target_data):
           # Use source model results as informative priors
           source_mean = source_results['parameter'].mean()
           source_std = source_results['parameter'].std()

           # Shrinkage parameter for transfer strength
           transfer_strength = ssp.parameters.Beta(alpha=2, beta=2)

           # Adaptive prior based on transfer strength
           prior_mean = transfer_strength * source_mean
           prior_std = (transfer_strength * source_std +
                       (1 - transfer_strength) * 2.0)  # Default std

           # Target model parameter
           target_param = ssp.parameters.Normal(mu=prior_mean, sigma=prior_std)

           # Target likelihood
           likelihood = ssp.parameters.Normal(mu=target_param, sigma=0.1)
           likelihood.observe(target_data)

Advanced Inference Techniques
-----------------------------

Custom MCMC Algorithms
~~~~~~~~~~~~~~~~~~~~~~

**Adaptive Metropolis-Hastings:**

.. code-block:: python

   class AdaptiveMCMC(ssp.inference.CustomSampler):
       def __init__(self, model, adaptation_window=100):
           super().__init__(model)
           self.adaptation_window = adaptation_window
           self.proposal_cov = None
           self.acceptance_history = []

       def adapt_proposal(self, samples):
           # Adapt proposal covariance based on sample history
           if len(samples) > self.adaptation_window:
               recent_samples = samples[-self.adaptation_window:]
               self.proposal_cov = np.cov(recent_samples.T)

       def sample_step(self, current_state):
           # Custom sampling step with adaptation
           proposal = self.propose(current_state)
           acceptance_prob = self.compute_acceptance_prob(current_state, proposal)

           if np.random.rand() < acceptance_prob:
               self.acceptance_history.append(1)
               return proposal
           else:
               self.acceptance_history.append(0)
               return current_state

**Parallel Tempering:**

.. code-block:: python

   class ParallelTempering(ssp.inference.CustomSampler):
       def __init__(self, model, temperatures):
           super().__init__(model)
           self.temperatures = np.array(temperatures)
           self.n_chains = len(temperatures)
           self.chains = [self.initialize_chain() for _ in range(self.n_chains)]

       def tempered_log_prob(self, state, temperature):
           return self.model.log_prob(state) / temperature

       def swap_proposal(self, i, j):
           # Propose swap between chains i and j
           temp_i, temp_j = self.temperatures[i], self.temperatures[j]
           state_i, state_j = self.chains[i], self.chains[j]

           log_ratio = (
               (1/temp_j - 1/temp_i) * self.model.log_prob(state_i) +
               (1/temp_i - 1/temp_j) * self.model.log_prob(state_j)
           )

           if np.log(np.random.rand()) < log_ratio:
               self.chains[i], self.chains[j] = state_j, state_i

Variational Inference Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Normalizing Flows:**

.. code-block:: python

   class NormalizingFlowVI(ssp.inference.VariationalInference):
       def __init__(self, model, flow_layers=5):
           super().__init__(model)
           self.flow = self.build_flow(flow_layers)

       def build_flow(self, n_layers):
           # Build normalizing flow for flexible approximation
           layers = []
           for _ in range(n_layers):
               layers.append(ssp.inference.RealNVPLayer(
                   input_dim=self.model.n_parameters
               ))
           return ssp.inference.NormalizingFlow(layers)

       def approximate_posterior(self, n_samples):
           # Sample from flow-based approximation
           base_samples = torch.randn(n_samples, self.model.n_parameters)
           return self.flow.forward(base_samples)

Model Debugging and Profiling
-----------------------------

Advanced Debugging Tools
~~~~~~~~~~~~~~~~~~~~~~~~

**Computational Graph Inspection:**

.. code-block:: python

   # Visualize model computational graph
   model_graph = ssp.utils.build_computation_graph(model)
   ssp.utils.visualize_graph(model_graph, save_path='model_graph.png')

   # Identify computational bottlenecks
   profiler = ssp.utils.ModelProfiler(model)
   profile_results = profiler.profile_operations()

   print("Operation timing:")
   for op, time in profile_results.items():
       print(f"{op}: {time:.3f}s")

**Gradient Analysis:**

.. code-block:: python

   # Analyze gradient flow
   gradient_analyzer = ssp.diagnostics.GradientAnalyzer(model)

   # Check for vanishing/exploding gradients
   gradient_norms = gradient_analyzer.compute_gradient_norms()

   if gradient_norms.max() > 100:
       print("Warning: Potential exploding gradients")
   if gradient_norms.min() < 1e-6:
       print("Warning: Potential vanishing gradients")

Memory and Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Large-Scale Model Optimization:**

.. code-block:: python

   # Memory-efficient inference for large models
   class LargeModelInference:
       def __init__(self, model, chunk_size=1000):
           self.model = model
           self.chunk_size = chunk_size

       def chunked_inference(self, data):
           results = []
           for i in range(0, len(data), self.chunk_size):
               chunk = data[i:i+self.chunk_size]
               chunk_result = self.model.sample(data=chunk)
               results.append(chunk_result)

           return self.combine_results(results)

**Distributed Computing:**

.. code-block:: python

   # Distributed MCMC across multiple machines
   distributed_config = ssp.inference.DistributedConfig(
       n_nodes=4,
       chains_per_node=2,
       communication_backend='mpi'
   )

   distributed_sampler = ssp.inference.DistributedMCMC(
       model, config=distributed_config
   )

   results = distributed_sampler.sample()

Advanced Model Validation
-------------------------

Cross-Validation for Complex Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Time Series Cross-Validation:**

.. code-block:: python

   class TimeSeriesCV:
       def __init__(self, model, initial_window=100, step_size=10):
           self.model = model
           self.initial_window = initial_window
           self.step_size = step_size

       def expanding_window_cv(self, time_series):
           cv_scores = []

           for i in range(self.initial_window, len(time_series), self.step_size):
               # Train on expanding window
               train_data = time_series[:i]
               test_data = time_series[i:i+self.step_size]

               # Fit model
               model_copy = self.model.copy()
               model_copy.fit(train_data)

               # Evaluate on test set
               predictions = model_copy.predict(test_data)
               score = self.evaluate_predictions(test_data, predictions)
               cv_scores.append(score)

           return cv_scores

**Hierarchical Cross-Validation:**

.. code-block:: python

   def hierarchical_cv(hierarchical_model, groups, n_folds=5):
       """Cross-validation respecting hierarchical structure."""

       # Split at group level, not individual observation level
       unique_groups = np.unique(groups)
       group_folds = np.array_split(unique_groups, n_folds)

       cv_scores = []
       for fold_groups in group_folds:
           # Hold out entire groups
           test_mask = np.isin(groups, fold_groups)
           train_mask = ~test_mask

           # Fit on training groups
           train_model = hierarchical_model.copy()
           train_model.fit(data[train_mask])

           # Evaluate on test groups
           test_score = train_model.evaluate(data[test_mask])
           cv_scores.append(test_score)

       return cv_scores

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~

**Global Sensitivity Analysis:**

.. code-block:: python

   class SobolSensitivityAnalysis:
       def __init__(self, model, parameter_ranges):
           self.model = model
           self.parameter_ranges = parameter_ranges

       def compute_sobol_indices(self, n_samples=1000):
           # Generate Sobol sequences
           sobol_samples = self.generate_sobol_samples(n_samples)

           # Evaluate model at sample points
           model_outputs = []
           for sample in sobol_samples:
               output = self.model.predict(sample)
               model_outputs.append(output)

           # Compute Sobol indices
           first_order, total_order = self.calculate_indices(
               sobol_samples, model_outputs
           )

           return {
               'first_order': first_order,
               'total_order': total_order
           }

Integration with Scientific Software
-----------------------------------

External Library Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Integration with Scientific Python Stack:**

.. code-block:: python

   # Integration with scikit-learn
   class ScikitLearnBayesian:
       def __init__(self, sklearn_model):
           self.sklearn_model = sklearn_model

       def bayesian_wrapper(self, X, y):
           # Use sklearn predictions as mean function
           predictions = self.sklearn_model.predict(X)

           # Add Bayesian uncertainty
           uncertainty = ssp.parameters.LogNormal(mu=np.log(0.1), sigma=0.3)

           likelihood = ssp.parameters.Normal(mu=predictions, sigma=uncertainty)
           likelihood.observe(y)

           return ssp.Model(likelihood)

**Integration with Domain-Specific Libraries:**

.. code-block:: python

   # Integration with BioPython for sequence analysis
   def phylogenetic_bayesian_model(sequences, tree_structure):
       # Use BioPython for sequence alignment
       aligned_sequences = align_sequences(sequences)

       # Bayesian phylogenetic model
       branch_lengths = ssp.parameters.Exponential(rate=1.0, shape=tree_structure.n_branches)
       substitution_rate = ssp.parameters.LogNormal(mu=0, sigma=1)

       # Felsenstein pruning algorithm likelihood
       likelihood = PhylogeneticLikelihood(
           sequences=aligned_sequences,
           tree=tree_structure,
           branch_lengths=branch_lengths,
           substitution_rate=substitution_rate
       )

       return ssp.Model(likelihood)

Best Practices for Advanced Usage
---------------------------------

1. **Modular Design**: Break complex models into reusable components
2. **Computational Efficiency**: Profile and optimize bottlenecks
3. **Numerical Stability**: Use log-space computations when appropriate
4. **Robust Validation**: Implement comprehensive testing strategies
5. **Documentation**: Document custom components thoroughly
6. **Version Control**: Track model versions and dependencies
7. **Reproducibility**: Use seeds and configuration management

**Advanced Workflow Example:**

.. code-block:: python

   def advanced_modeling_workflow(data, model_class):
       """Complete advanced modeling workflow."""

       # 1. Model development with debugging
       with ssp.utils.debug_mode():
           model = model_class(data)
           model.validate()

       # 2. Multi-backend optimization
       backends = ['numpy', 'pytorch', 'stan']
       backend_results = {}

       for backend in backends:
           model.set_backend(backend)
           result = model.sample()
           backend_results[backend] = result

       # 3. Advanced validation
       cv_scores = hierarchical_cv(model, data.groups)
       sensitivity = SobolSensitivityAnalysis(model).compute_sobol_indices()

       # 4. Production deployment
       optimized_model = ssp.utils.optimize_for_production(model)

       return {
           'model': optimized_model,
           'results': backend_results,
           'validation': cv_scores,
           'sensitivity': sensitivity
       }

This advanced topics guide provides the tools and techniques needed for sophisticated scientific modeling with SciStanPy.
