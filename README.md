# SciStanPy: Intuitive Bayesian Modeling for Scientists

SciStanPy is a Python framework that makes Bayesian statistical modeling accessible to scientists and researchers. Whether you're analyzing experimental data, building predictive models, or exploring uncertainty in your research, SciStanPy provides an intuitive interface that bridges the gap between scientific thinking and advanced statistical computation.

## Why SciStanPy?

**Think in Science, Compute with Confidence**

Traditional Bayesian modeling often requires deep statistical programming knowledge. SciStanPy changes this by letting you express your scientific models naturally in Python, while automatically handling the complex computational details behind the scenes.

- **Intuitive Model Building**: Express your scientific hypotheses using familiar Python syntax
- **Automatic Optimization**: Your models are automatically compiled to high-performance Stan code
- **Multi-Backend Flexibility**: Compute with NumPy, PyTorch, or Stan depending on your needs
- **Built-in Validation**: Comprehensive error checking helps catch modeling mistakes early
- **Scientific Focus**: Designed specifically for scientific applications and data analysis workflows

## Quick Example: From Science to Statistics

Let's say you're studying the relationship between temperature and reaction rates in your lab:

```python
import scistanpy as ssp
import numpy as np

# Your experimental data
temperatures = np.array([20, 25, 30, 35, 40])  # °C
reaction_rates = np.array([0.1, 0.3, 0.8, 2.1, 5.2])  # units/sec

# Express your scientific model naturally
class MyModel(ssp.Model):
    def __init__(self, temperatures, reaction_rates):

        # Record default data
        super().__init__(default_data = {"reaction_rates": reaction_rates})

        # Define priors
        self.baseline_rate = ssp.parameters.LogNormal(mu=0, sigma=1)  # Rates are always positive
        self.temperature_effect = ssp.parameters.Normal(mu=0, sigma=0.5)  # Effect of temperature

        # Model the relationship (Arrhenius-like behavior with noise)
        self.reaction_rates = ssp.parameters.Normal(
            mu = self.baseline_rate * ssp.operations.exp(self.temperature_effect * temperatures),
            sigma = 0.1
        )

# Build the model
model = MyModel(temperatures, reaction_rates)

# Prior predictive checks
model.prior_predictive() # <--- Interactive plots created in Jupyter environment

# Maximum likelihood estimation using PyTorch
mle = model.mle()

# Hamiltonian monte carlo using Stan
hmc = model.mcmc()

# Evaluate model fit
results.diagnose()
```

SciStanPy handles the statistical complexity while you focus on the science.

# Getting Started

## Installation

SciStanPy works with Python 3.8+ and can be installed via pip:

```bash
pip install scistanpy
```

For development or advanced features, you can install from source:

```bash
git clone https://github.com/microsoft/SciStanPy.git
cd SciStanPy
pip install -e .
```

## Dependencies

SciStanPy builds on proven scientific Python libraries:

- **NumPy**: For numerical computations and array operations
- **SciPy**: For statistical distributions and mathematical functions
- **PyTorch**: For automatic differentiation and GPU acceleration
- **CmdStanPy**: For interfacing with the Stan probabilistic programming language
- **Matplotlib**: For visualization and plotting

All dependencies are automatically installed with SciStanPy.

## Your First Model

Here's a simple example to get you started - analyzing measurement uncertainty:

```python
import scistanpy as ssp
import numpy as np

# Simulated measurement data with known uncertainty
measurements = np.array([10.2, 9.8, 10.1, 9.9, 10.3])

# Model: what's the true value?
true_value = ssp.Normal(mu=10, sigma=1)  # Prior belief
measurement_noise = 0.2  # Known measurement uncertainty

# Likelihood: each measurement follows the true value plus noise
likelihood = ssp.Normal(mu=true_value, sigma=measurement_noise)
likelihood.observe(measurements)

# Build and sample from the model
model = ssp.Model(likelihood)
posterior_samples = model.sample()

print(f"Estimated true value: {posterior_samples['true_value'].mean():.2f}")
print(f"95% credible interval: [{np.percentile(posterior_samples['true_value'], 2.5):.2f}, "
      f"{np.percentile(posterior_samples['true_value'], 97.5):.2f}]")
```

## Key Concepts for Scientists

**Parameters vs Observations**: In SciStanPy, parameters represent unknown quantities you want to learn about, while observations are your measured data.

**Distributions**: These encode your knowledge and uncertainty. Use them to express:
- Prior knowledge about parameters
- Expected relationships between variables
- Measurement uncertainty and noise

**Model Building**: Combine parameters using standard mathematical operations. SciStanPy automatically tracks relationships and generates efficient computation code.

**Inference**: Once your model is built, SciStanPy handles the complex sampling mathematics to give you posterior distributions representing your updated knowledge.

## Common Scientific Use Cases

- **Parameter Estimation**: Determine values and uncertainties for physical constants
- **Model Comparison**: Compare different theoretical models using your data
- **Experimental Design**: Predict outcomes and optimize future experiments
- **Data Fusion**: Combine multiple data sources with different uncertainties
- **Time Series Analysis**: Model temporal dynamics and make forecasts
- **Hierarchical Modeling**: Account for grouping structure in your data

## Documentation and Learning

- **User Guide**: Comprehensive tutorials and examples for scientific applications
- **API Reference**: Complete documentation of all functions and classes
- **Example Gallery**: Real-world scientific modeling examples
- **Best Practices**: Guidelines for effective Bayesian modeling in research

## Advanced Features

**Multi-Backend Computing**: Start prototyping with NumPy, scale up with PyTorch for GPU acceleration, or compile to Stan for maximum performance.

**Custom Distributions**: Extend SciStanPy with domain-specific distributions for your field.

**Automatic Differentiation**: Built-in gradient computation enables advanced algorithms like variational inference and Hamiltonian Monte Carlo.

**Model Diagnostics**: Comprehensive tools to validate your models and check convergence.

# Build and Test

For developers and contributors:

```bash
# Clone the repository
git clone https://github.com/microsoft/SciStanPy.git
cd SciStanPy

# Install in development mode
pip install -e .[dev]

# Run the test suite
pytest tests/

# Build documentation
cd docs/
make html
```

The test suite includes unit tests, integration tests, and scientific validation examples to ensure reliability across different use cases.

# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.