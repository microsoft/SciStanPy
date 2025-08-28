"""Custom exception classes for the SciStanPy package.

This module defines a hierarchy of custom exceptions used throughout the
SciStanPy package to provide clear error reporting and exception handling.
All custom exceptions inherit from the base SciStanPyError class to allow
for unified exception handling when needed.

The exception hierarchy follows Python best practices by providing both
general and specific exception types for different error conditions that
may arise during model construction, fitting, and analysis.
"""


class SciStanPyError(Exception):
    """Base class for all exceptions in the SciStanPy package.

    This exception serves as the root of the SciStanPy exception hierarchy,
    allowing users to catch all package-specific exceptions with a single
    except clause. All other SciStanPy exceptions should inherit from this
    base class.

    :param message: Error message describing the exception
    :type message: str

    Example:
        >>> try:
        ...     # SciStanPy operations
        ...     pass
        ... except SciStanPyError as e:
        ...     print(f"SciStanPy error occurred: {e}")
    """


class NumpySampleError(SciStanPyError):
    """Raised when a NumPy sample is not valid for the intended operation.

    This exception is raised when sample data provided as NumPy arrays
    fails validation checks, such as having incorrect dimensions, invalid
    values, or incompatible data types for the requested operation.

    :param message: Error message describing why the sample is invalid
    :type message: str
    """
