"""Custom exceptions for the SciStanPy package."""


class SciStanPyError(Exception):
    """Base class for exceptions in the SciStanPy package."""


class NumpySampleError(SciStanPyError):
    """Raised when a numpy sample is not valid."""
