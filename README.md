SciStanPy: Intuitive Bayesian Modeling for Scientists
=====================================================

Welcome to SciStanPy, a Python framework that makes Bayesian statistical modeling accessible to scientists and researchers. This page covers installation only. For comprehensive documentation along with examples, see the [documentation](https://microsoft.github.io/scistanpy/).

# Installation
To start, clone SciStanPy from GitHub:

```bash
git clone https://github.com/microsoft/SciStanPy.git
```

If you are not planning to contribute to SciStanPy, you can install with `pip` as below from the root directory.

```bash
pip install .
```

Once Python dependencies are installed with `pip`, install `cmdstan` with the command `install_cmdstan`. Installation of SciStanPy is now complete! Take a look [here](https://microsoft.github.io/scistanpy/) for a quick-start guide.

If you are a developer looking to contribute to SciStanPy (or if you are a casual user who wants to use the development environment), install via `conda` as below:

```bash
conda env create -f conda.yml
conda activate scistanpy
install_cmdstanpy
```
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