Integration Patterns
====================

This guide covers advanced patterns for integrating SciStanPy with other scientific computing tools, workflows, and infrastructure.

Scientific Computing Ecosystem Integration
------------------------------------------

NumPy and SciPy Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Advanced Array Operations:**

.. code-block:: python

   import numpy as np
   from scipy import integrate, optimize
   import scistanpy as ssp

   class IntegratedModel:
       """Model that combines SciStanPy with SciPy numerical methods."""

       def __init__(self, data):
           self.data = data

           # SciStanPy parameters
           self.amplitude = ssp.parameters.LogNormal(mu=0, sigma=1)
           self.frequency = ssp.parameters.LogNormal(mu=np.log(1), sigma=0.5)
           self.phase = ssp.parameters.Uniform(low=0, high=2*np.pi)

           # Derived quantities using SciPy
           self.model_predictions = self._compute_predictions()

       def _compute_predictions(self):
           """Compute model predictions using SciPy integration."""
           def integrand(t, amp, freq, phase):
               return amp * np.sin(freq * t + phase)

           # Create transformation that uses SciPy
           return ssp.operations.scipy_integrate(
               integrand,
               bounds=(0, 2*np.pi),
               parameters=(self.amplitude, self.frequency, self.phase)
           )

**Custom SciPy-Based Transformations:**

.. code-block:: python

   from scistanpy.model.components.transformations.transformed_parameters import UnaryTransformedParameter

   class ScipyOptimizedTransform(UnaryTransformedParameter):
       """Transformation using SciPy optimization."""

       def __init__(self, dist1, target_function):
           self.target_function = target_function
           super().__init__(dist1=dist1)

       def run_np_torch_op(self, dist1):
           """Apply SciPy optimization to parameter values."""
           results = []
           for value in np.atleast_1d(dist1):
               # Use SciPy optimization
               result = optimize.minimize_scalar(
                   lambda x: self.target_function(x, value),
                   bounds=(0, 10),
                   method='bounded'
               )
               results.append(result.x)

           return np.array(results)

Pandas Integration
~~~~~~~~~~~~~~~~~

**DataFrame-Aware Models:**

.. code-block:: python

   import pandas as pd

   class PandasIntegratedModel:
       """SciStanPy model that works naturally with pandas DataFrames."""

       def __init__(self, dataframe):
           self.df = dataframe
           self._validate_dataframe()
           self._build_model()

       def _validate_dataframe(self):
           """Validate DataFrame structure for modeling."""
           required_columns = ['response', 'predictor1', 'predictor2']
           missing_columns = set(required_columns) - set(self.df.columns)
           if missing_columns:
               raise ValueError(f"Missing required columns: {missing_columns}")

       def _build_model(self):
           """Build SciStanPy model from DataFrame."""
           # Categorical variables
           if 'group' in self.df.columns:
               self.group_effects = ssp.parameters.Normal(
                   mu=0, sigma=1,
                   shape=(self.df['group'].nunique(),)
               )

           # Continuous predictors
           X = self.df[['predictor1', 'predictor2']].values
           self.coefficients = ssp.parameters.Normal(mu=0, sigma=1, shape=(X.shape[1],))

           # Model predictions
           predictions = X @ self.coefficients
           if hasattr(self, 'group_effects'):
               group_indices = pd.Categorical(self.df['group']).codes
               predictions += self.group_effects[group_indices]

           # Likelihood
           self.sigma = ssp.parameters.LogNormal(mu=0, sigma=1)
           self.likelihood = ssp.parameters.Normal(mu=predictions, sigma=self.sigma)
           self.likelihood.observe(self.df['response'].values)

       def predict(self, new_data):
           """Make predictions on new DataFrame."""
           # Ensure same structure as training data
           X_new = new_data[['predictor1', 'predictor2']].values

           # Generate predictions
           predictions = X_new @ self.posterior_coefficients

           return pd.DataFrame({
               'prediction_mean': predictions.mean(axis=0),
               'prediction_std': predictions.std(axis=0),
               'lower_ci': np.percentile(predictions, 2.5, axis=0),
               'upper_ci': np.percentile(predictions, 97.5, axis=0)
           })

Visualization Integration
------------------------

Matplotlib Advanced Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Custom Plotting Classes:**

.. code-block:: python

   import matplotlib.pyplot as plt
   from matplotlib.patches import Rectangle
   import seaborn as sns

   class SciStanPyPlotter:
       """Advanced plotting for SciStanPy models."""

       def __init__(self, model, results):
           self.model = model
           self.results = results

       def plot_comprehensive_diagnostics(self):
           """Create comprehensive diagnostic plot."""
           fig = plt.figure(figsize=(16, 12))
           gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

           # Trace plots
           self._plot_traces(fig, gs[0, :2])

           # Posterior distributions
           self._plot_posteriors(fig, gs[0, 2:])

           # Autocorrelation
           self._plot_autocorrelation(fig, gs[1, :2])

           # Energy plots
           self._plot_energy(fig, gs[1, 2:])

           # Posterior predictive
           self._plot_posterior_predictive(fig, gs[2, :])

           return fig

       def _plot_traces(self, fig, gs):
           """Plot MCMC traces with advanced diagnostics."""
           ax = fig.add_subplot(gs)

           for param_name, samples in self.results.items():
               if samples.ndim == 1:  # Scalar parameters only
                   # Plot traces with different colors for chains
                   n_chains = samples.shape[0] if samples.ndim > 1 else 1
                   colors = plt.cm.tab10(np.linspace(0, 1, n_chains))

                   for chain in range(n_chains):
                       chain_samples = samples[chain] if n_chains > 1 else samples
                       ax.plot(chain_samples, color=colors[chain % len(colors)], alpha=0.7)

           ax.set_xlabel('Iteration')
           ax.set_ylabel('Parameter Value')
           ax.set_title('MCMC Traces')

       def plot_model_comparison(self, models_dict):
           """Compare multiple models visually."""
           n_models = len(models_dict)
           fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 8))

           for i, (name, model_results) in enumerate(models_dict.items()):
               # LOO comparison
               loo = model_results['loo']
               axes[0, i].bar(['ELPD'], [loo['elpd_loo']],
                            yerr=[loo['se_elpd_loo']], capsize=5)
               axes[0, i].set_title(f'{name}\nLOO-CV')

               # Posterior predictive
               post_pred = model_results['posterior_predictive']
               axes[1, i].hist(post_pred.flatten(), alpha=0.7, bins=30)
               axes[1, i].set_title(f'{name}\nPosterior Predictive')

           return fig

Plotly Interactive Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Interactive Diagnostics:**

.. code-block:: python

   import plotly.graph_objects as go
   from plotly.subplots import make_subplots
   import plotly.express as px

   class InteractiveDiagnostics:
       """Interactive diagnostic plots using Plotly."""

       def __init__(self, model, results):
           self.model = model
           self.results = results

       def create_interactive_dashboard(self):
           """Create interactive dashboard for model diagnostics."""
           # Create subplots
           fig = make_subplots(
               rows=2, cols=2,
               subplot_titles=('Trace Plot', 'Posterior Distribution',
                             'Autocorrelation', 'Energy Plot'),
               specs=[[{"secondary_y": False}, {"secondary_y": False}],
                     [{"secondary_y": False}, {"secondary_y": False}]]
           )

           # Add interactive traces
           self._add_interactive_traces(fig)
           self._add_posterior_distributions(fig)
           self._add_autocorrelation(fig)
           self._add_energy_plot(fig)

           # Update layout for interactivity
           fig.update_layout(
               title="Interactive Model Diagnostics",
               showlegend=True,
               hovermode='x unified'
           )

           return fig

       def create_parameter_explorer(self):
           """Create parameter exploration interface."""
           # Create parameter selection dropdown
           parameters = list(self.results.keys())

           fig = go.Figure()

           # Add traces for each parameter (initially hidden)
           for param in parameters:
               samples = self.results[param]
               if samples.ndim == 1:
                   fig.add_trace(
                       go.Histogram(
                           x=samples,
                           name=param,
                           visible=(param == parameters[0])  # Only first visible
                       )
                   )

           # Create dropdown menu
           dropdown_buttons = []
           for i, param in enumerate(parameters):
               visibility = [False] * len(parameters)
               visibility[i] = True

               dropdown_buttons.append(
                   dict(
                       args=[{"visible": visibility}],
                       label=param,
                       method="restyle"
                   )
               )

           fig.update_layout(
               updatemenus=[
                   dict(
                       active=0,
                       buttons=dropdown_buttons,
                       direction="down",
                       showactive=True,
                       x=1.0,
                       xanchor="left",
                       y=1.0,
                       yanchor="top"
                   )
               ],
               title="Interactive Parameter Explorer"
           )

           return fig

Machine Learning Integration
---------------------------

Scikit-learn Pipeline Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SciStanPy as Scikit-learn Estimator:**

.. code-block:: python

   from sklearn.base import BaseEstimator, RegressorMixin
   from sklearn.model_selection import cross_val_score
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler

   class SciStanPyRegressor(BaseEstimator, RegressorMixin):
       """Scikit-learn compatible SciStanPy regressor."""

       def __init__(self, inference_method='vi', n_samples=1000):
           self.inference_method = inference_method
           self.n_samples = n_samples
           self.model_ = None
           self.results_ = None

       def fit(self, X, y):
           """Fit SciStanPy model using scikit-learn interface."""
           # Build SciStanPy model
           n_features = X.shape[1]

           # Priors
           intercept = ssp.parameters.Normal(mu=0, sigma=1)
           coefficients = ssp.parameters.Normal(mu=0, sigma=1, shape=(n_features,))
           sigma = ssp.parameters.LogNormal(mu=0, sigma=1)

           # Model
           predictions = intercept + X @ coefficients
           likelihood = ssp.parameters.Normal(mu=predictions, sigma=sigma)
           likelihood.observe(y)

           self.model_ = ssp.Model(likelihood)

           # Fit using specified method
           if self.inference_method == 'vi':
               self.results_ = self.model_.variational(n_samples=self.n_samples)
           elif self.inference_method == 'mcmc':
               self.results_ = self.model_.sample(n_samples=self.n_samples)
           else:
               self.results_ = self.model_.mle()

           return self

       def predict(self, X):
           """Make predictions using fitted model."""
           if self.model_ is None:
               raise ValueError("Model not fitted yet")

           # Extract posterior means for point predictions
           if isinstance(self.results_, dict) and 'intercept' in self.results_:
               intercept_mean = self.results_['intercept'].mean()
               coef_mean = self.results_['coefficients'].mean(axis=0)
               return intercept_mean + X @ coef_mean
           else:
               # MLE case
               return self.model_.predict(X, self.results_)

       def predict_with_uncertainty(self, X):
           """Predict with uncertainty quantification."""
           # Generate posterior predictive samples
           post_pred = self.model_.posterior_predictive(self.results_, X_new=X)

           return {
               'mean': post_pred.mean(axis=0),
               'std': post_pred.std(axis=0),
               'lower_ci': np.percentile(post_pred, 2.5, axis=0),
               'upper_ci': np.percentile(post_pred, 97.5, axis=0)
           }

**Integration with Scikit-learn Pipelines:**

.. code-block:: python

   # Create scikit-learn pipeline with SciStanPy
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('regressor', SciStanPyRegressor(inference_method='vi'))
   ])

   # Use standard scikit-learn workflow
   scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')

   # Fit and predict
   pipeline.fit(X_train, y_train)
   predictions = pipeline.predict(X_test)

TensorFlow/Keras Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Neural Network Prior Integration:**

.. code-block:: python

   import tensorflow as tf
   import tensorflow_probability as tfp

   class NeuralNetworkPrior:
       """Use neural network as prior for SciStanPy parameters."""

       def __init__(self, input_dim, output_dim):
           self.nn_model = self._build_network(input_dim, output_dim)

       def _build_network(self, input_dim, output_dim):
           """Build neural network for prior specification."""
           model = tf.keras.Sequential([
               tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
               tf.keras.layers.Dense(32, activation='relu'),
               tf.keras.layers.Dense(output_dim * 2)  # Mean and log-std
           ])
           return model

       def get_prior_parameters(self, features):
           """Get prior parameters from neural network."""
           nn_output = self.nn_model(features)

           # Split into mean and log-std
           mean = nn_output[:, :nn_output.shape[1]//2]
           log_std = nn_output[:, nn_output.shape[1]//2:]

           return mean, tf.exp(log_std)

       def create_scistanpy_prior(self, features):
           """Create SciStanPy prior from neural network output."""
           mean, std = self.get_prior_parameters(features)

           # Convert to SciStanPy parameters
           return ssp.parameters.Normal(
               mu=mean.numpy(),
               sigma=std.numpy()
           )

Cloud and Distributed Computing
------------------------------

Dask Integration
~~~~~~~~~~~~~~~

**Distributed Computing with Dask:**

.. code-block:: python

   import dask
   from dask.distributed import Client, as_completed
   from dask import delayed

   class DistributedSciStanPy:
       """Distributed SciStanPy computation using Dask."""

       def __init__(self, client=None):
           self.client = client or Client()

       @delayed
       def fit_model_chunk(self, data_chunk, model_builder):
           """Fit model on a chunk of data."""
           model = model_builder(data_chunk)
           results = model.sample(n_samples=500)  # Smaller samples per chunk
           return results

       def distributed_bootstrap(self, data, model_builder, n_bootstrap=100):
           """Perform distributed bootstrap analysis."""
           bootstrap_tasks = []

           for i in range(n_bootstrap):
               # Create bootstrap sample
               bootstrap_indices = np.random.choice(
                   len(data), size=len(data), replace=True
               )
               bootstrap_data = data[bootstrap_indices]

               # Create delayed task
               task = self.fit_model_chunk(bootstrap_data, model_builder)
               bootstrap_tasks.append(task)

           # Execute in parallel
           bootstrap_results = dask.compute(*bootstrap_tasks)

           return self._combine_bootstrap_results(bootstrap_results)

       def parallel_model_comparison(self, data, model_builders):
           """Compare multiple models in parallel."""
           model_tasks = []

           for name, builder in model_builders.items():
               task = self.fit_model_chunk(data, builder)
               model_tasks.append((name, task))

           # Execute and collect results
           results = {}
           for name, task in model_tasks:
               results[name] = task.compute()

           return results

Cloud Deployment Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~

**AWS Integration:**

.. code-block:: python

   import boto3
   import json

   class AWSIntegration:
       """Integration with AWS services for SciStanPy."""

       def __init__(self):
           self.s3_client = boto3.client('s3')
           self.lambda_client = boto3.client('lambda')

       def deploy_model_to_lambda(self, model, function_name):
           """Deploy SciStanPy model as AWS Lambda function."""

           # Serialize model
           model_data = self._serialize_model(model)

           # Create Lambda deployment package
           lambda_code = self._create_lambda_package(model_data)

           # Deploy to Lambda
           response = self.lambda_client.create_function(
               FunctionName=function_name,
               Runtime='python3.9',
               Role='arn:aws:iam::account:role/lambda-execution-role',
               Handler='lambda_function.lambda_handler',
               Code={'ZipFile': lambda_code},
               Timeout=300,
               MemorySize=1024
           )

           return response

       def batch_inference(self, model, data_s3_path, output_s3_path):
           """Run batch inference using AWS Batch."""

           job_definition = {
               'jobDefinitionName': 'scistanpy-batch-inference',
               'type': 'container',
               'containerProperties': {
                   'image': 'scistanpy/inference:latest',
                   'vcpus': 4,
                   'memory': 8192,
                   'environment': [
                       {'name': 'INPUT_S3_PATH', 'value': data_s3_path},
                       {'name': 'OUTPUT_S3_PATH', 'value': output_s3_path}
                   ]
               }
           }

           # Submit batch job
           response = boto3.client('batch').submit_job(
               jobName='scistanpy-inference-job',
               jobQueue='scistanpy-queue',
               jobDefinition=job_definition['jobDefinitionName']
           )

           return response

Database Integration
-------------------

SQL Database Integration
~~~~~~~~~~~~~~~~~~~~~~~

**Database-Backed Models:**

.. code-block:: python

   import sqlalchemy as sa
   from sqlalchemy.orm import sessionmaker

   class DatabaseIntegratedModel:
       """SciStanPy model with database integration."""

       def __init__(self, connection_string):
           self.engine = sa.create_engine(connection_string)
           self.Session = sessionmaker(bind=self.engine)

       def load_data_from_db(self, query):
           """Load data directly from database."""
           with self.Session() as session:
               result = session.execute(sa.text(query))
               data = pd.DataFrame(result.fetchall(), columns=result.keys())
           return data

       def store_results_to_db(self, results, table_name):
           """Store model results back to database."""

           # Convert results to DataFrame
           results_df = self._results_to_dataframe(results)

           # Store to database
           results_df.to_sql(
               table_name,
               self.engine,
               if_exists='replace',
               index=False
           )

       def incremental_learning(self, table_name, batch_size=1000):
           """Implement incremental learning from database."""

           # Get total number of records
           with self.Session() as session:
               count_result = session.execute(
                   sa.text(f"SELECT COUNT(*) FROM {table_name}")
               ).scalar()

           # Process in batches
           for offset in range(0, count_result, batch_size):
               # Load batch
               query = f"""
                   SELECT * FROM {table_name}
                   LIMIT {batch_size} OFFSET {offset}
               """
               batch_data = self.load_data_from_db(query)

               # Update model with batch
               self._update_model_with_batch(batch_data)

Time Series Database Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**InfluxDB Integration for Time Series:**

.. code-block:: python

   from influxdb_client import InfluxDBClient

   class TimeSeriesModel:
       """SciStanPy model for time series data from InfluxDB."""

       def __init__(self, influx_config):
           self.client = InfluxDBClient(**influx_config)
           self.query_api = self.client.query_api()

       def load_time_series(self, measurement, field, start_time, end_time):
           """Load time series data from InfluxDB."""

           query = f'''
               from(bucket: "scientific_data")
               |> range(start: {start_time}, stop: {end_time})
               |> filter(fn: (r) => r["_measurement"] == "{measurement}")
               |> filter(fn: (r) => r["_field"] == "{field}")
               |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
           '''

           result = self.query_api.query_data_frame(query)
           return result

       def build_temporal_model(self, time_series_data):
           """Build temporal model from time series data."""

           # Extract time and values
           times = time_series_data['_time'].values
           values = time_series_data[field].values

           # Time-varying parameters
           trend = ssp.parameters.Normal(mu=0, sigma=1)
           seasonality_amplitude = ssp.parameters.LogNormal(mu=0, sigma=1)
           noise_level = ssp.parameters.LogNormal(mu=0, sigma=1)

           # Temporal model
           time_numeric = (times - times[0]).astype('timedelta64[h]').astype(float)

           seasonal_component = seasonality_amplitude * ssp.operations.sin(
               2 * np.pi * time_numeric / 24  # Daily seasonality
           )

           predictions = trend * time_numeric + seasonal_component

           # Likelihood
           likelihood = ssp.parameters.Normal(mu=predictions, sigma=noise_level)
           likelihood.observe(values)

           return ssp.Model(likelihood)

Real-time Processing Integration
------------------------------

Apache Kafka Integration
~~~~~~~~~~~~~~~~~~~~~~~

**Streaming Model Updates:**

.. code-block:: python

   from kafka import KafkaConsumer, KafkaProducer
   import json

   class StreamingModelUpdater:
       """Real-time model updating with Kafka streams."""

       def __init__(self, kafka_config):
           self.consumer = KafkaConsumer(
               'scientific_data',
               bootstrap_servers=kafka_config['servers'],
               value_deserializer=lambda x: json.loads(x.decode('utf-8'))
           )

           self.producer = KafkaProducer(
               bootstrap_servers=kafka_config['servers'],
               value_serializer=lambda x: json.dumps(x).encode('utf-8')
           )

           self.model = None
           self.online_stats = OnlineStatistics()

       def stream_process(self):
           """Process streaming data and update model."""

           for message in self.consumer:
               data_point = message.value

               # Update online statistics
               self.online_stats.update(data_point)

               # Check if model update is needed
               if self._should_update_model():
                   # Rebuild model with updated data
                   self._rebuild_model()

                   # Send updated predictions
                   predictions = self._make_predictions(data_point)
                   self.producer.send('model_predictions', predictions)

       def _should_update_model(self):
           """Determine if model needs updating based on streaming data."""
           # Implement adaptive updating logic
           return (self.online_stats.n_samples % 1000 == 0 or
                   self.online_stats.drift_detected())

WebSocket Integration
~~~~~~~~~~~~~~~~~~~

**Real-time Model Serving:**

.. code-block:: python

   import asyncio
   import websockets
   import json

   class RealTimeModelServer:
       """WebSocket server for real-time SciStanPy predictions."""

       def __init__(self, model, results):
           self.model = model
           self.results = results

       async def handle_prediction_request(self, websocket, path):
           """Handle real-time prediction requests."""

           async for message in websocket:
               try:
                   # Parse request
                   request = json.loads(message)
                   features = np.array(request['features'])

                   # Make prediction
                   prediction = self._make_fast_prediction(features)

                   # Send response
                   response = {
                       'prediction': prediction.tolist(),
                       'timestamp': datetime.now().isoformat(),
                       'status': 'success'
                   }

                   await websocket.send(json.dumps(response))

               except Exception as e:
                   error_response = {
                       'error': str(e),
                       'status': 'error'
                   }
                   await websocket.send(json.dumps(error_response))

       def start_server(self, host='localhost', port=8765):
           """Start the WebSocket server."""
           return websockets.serve(
               self.handle_prediction_request,
               host,
               port
           )

This comprehensive guide provides patterns for integrating SciStanPy with the broader scientific computing ecosystem, enabling sophisticated workflows and deployment scenarios.
