# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SciStanPy"
copyright = "2025, Bruce Wittmann"
author = "Bruce Wittmann"
release = "Alpha"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",  # add links to source code
    "sphinx.ext.intersphinx",  # optional cross-references
    "sphinx.ext.githubpages",  # Support GitHub pages
]

autosummary_generate = True
autoclass_content = "both"  # include class and __init__ docstrings
autodoc_mock_imports = ["torch"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
