"""Holds default values shared by multiple components of the DMS package"""

# Fitting defaults
DEFAULT_N_EPOCHS = 100000
DEFAULT_EARLY_STOP = 10
DEFAULT_LR = 0.001

# Default order for index variable names
DEFAULT_INDEX_ORDER = ("i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t")

# Defaults for the Stan model
DEFAULT_FORCE_COMPILE = False
DEFAULT_STANC_OPTIONS = None
DEFAULT_CPP_OPTIONS = None
DEFAULT_USER_HEADER = None
