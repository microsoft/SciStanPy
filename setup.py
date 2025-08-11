"""
Installs SciStanPy
"""

import re

from setuptools import find_packages, setup


# Get package information
def get_package_info():
    """
    Gets version information for the installation.
    """
    # Set up variables
    package_version = None

    # Open the file containing version info
    with open("scistanpy/__init__.py", "r", encoding="utf-8") as file:
        for line in file:
            # Check version
            if match_obj := re.match(r"__version__.+([0-9]+\.[0-9]+\.[0-9]+)", line):
                package_version = match_obj.group(1)

    # Checks on variables
    if package_version is None:
        raise IOError("Could not find information on version.")

    return package_version


# Run setup
setup(version=get_package_info(), packages=find_packages())
