"""Custom exceptions for the DMS Stan package."""


class DMSStanError(Exception):
    """Base class for exceptions in the DMS Stan package."""


class NumpySampleError(DMSStanError):
    """Raised when a numpy sample is not valid."""
