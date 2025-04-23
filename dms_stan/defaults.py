"""Holds default values shared by multiple components of the DMS package"""

import string

# Fitting defaults
DEFAULT_N_EPOCHS = 100000
DEFAULT_EARLY_STOP = 10
DEFAULT_LR = 0.001

# Default order for index variable names
DEFAULT_INDEX_ORDER = ("i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t")

# Default names for the dimensions of the data
DEFAULT_DIM_NAMES = tuple(l for l in string.ascii_lowercase if l != "n")

# Defaults for the Stan model
DEFAULT_FORCE_COMPILE = False
DEFAULT_STANC_OPTIONS = None
DEFAULT_CPP_OPTIONS = {"STAN_THREADS": True}
DEFAULT_USER_HEADER = None

# Defaults for Stan diagnostics
DEFAULT_EBFMI_THRESH = 0.2
DEFAULT_ESS_THRESH = 400
DEFAULT_MAX_TREE_DEPTH = 10
DEFAULT_RHAT_THRESH = 1.01
