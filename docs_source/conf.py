# Configuration file for the Sphinx documentation builder.

# Basic information about the project.
project = "SciStanPy"
copyright = "%Y, Microsoft Corporation"
author = "Bruce Wittmann"
release = "Alpha"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",  # add links to source code
    "sphinx.ext.intersphinx",  # optional cross-references
    "sphinx.ext.githubpages",  # Support GitHub pages
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_mock_imports = ["torch"]  # Mock torch

templates_path = ["_templates"]  # Path to templates
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]  # Patterns to ignore

# HTML output settings
html_theme = "alabaster"
html_static_path = ["_static"]

# Set the maximum line length for function signatures in the documentation
maximum_signature_line_length = 100
