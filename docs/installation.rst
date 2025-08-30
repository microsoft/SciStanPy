Installation Guide
==================

System Requirements
-------------------

SciStanPy requires Python 3.8 or later and is compatible with:

- **Operating Systems**: Windows, macOS, Linux
- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Architecture**: x86_64, ARM64 (including Apple Silicon)

Basic Installation
------------------

Install SciStanPy using pip:

.. code-block:: bash

   pip install scistanpy

This will automatically install all required dependencies.

Development Installation
------------------------

For development work or to get the latest features:

.. code-block:: bash

   git clone https://github.com/microsoft/SciStanPy.git
   cd SciStanPy
   pip install -e .[dev]

Optional Dependencies
---------------------

**GPU Support** (PyTorch with CUDA):

.. code-block:: bash

   pip install torch[cuda]

**Plotting and Visualization**:

.. code-block:: bash

   pip install matplotlib seaborn plotly

**Jupyter Notebook Support**:

.. code-block:: bash

   pip install jupyter ipywidgets

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   import scistanpy as ssp
   print(ssp.__version__)

   # Quick functionality test
   import numpy as np
   data = np.random.normal(0, 1, 100)
   param = ssp.parameters.Normal(mu=0, sigma=1)
   print("Installation successful!")

Troubleshooting
---------------

**Common Issues**:

1. **Stan Installation**: If Stan compilation fails, ensure you have a C++ compiler installed
2. **PyTorch Issues**: For GPU support, install PyTorch with appropriate CUDA version
3. **Permission Errors**: Use ``pip install --user`` if you don't have admin privileges

**Getting Help**:

- Check the FAQ section
- Open an issue on GitHub
- Join our community discussions
