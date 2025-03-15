"""
Exceptions that are used throughout
"""

from __future__ import annotations


class MissingOptionalDependencyError(ImportError):
    """
    Raised when an optional dependency is missing

    For example, plotting dependencies like matplotlib
    """

    def __init__(self, callable_name: str, requirement: str) -> None:
        """
        Initialise the error

        Parameters
        ----------
        callable_name
            The name of the callable that requires the dependency

        requirement
            The name of the requirement
        """
        error_msg = f"`{callable_name}` requires {requirement} to be installed"
        super().__init__(error_msg)
