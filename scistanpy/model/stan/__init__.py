"""Code for interfacing with the Stan probabilistic programming language."""

import os.path

# We need the path of the directory of the current file. This is used to include
# the custom stan functions.
STAN_INCLUDE_PATHS = [os.path.abspath(os.path.dirname(__file__))]
