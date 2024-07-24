"""
Defines code needed for building models in Stan for evaluating deep mutational scanning
data.
"""

from typeguard import install_import_hook

# Set up type checking
install_import_hook("dms_stan")
