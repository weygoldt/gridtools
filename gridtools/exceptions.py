"""Specific exceptions for the gridtools package."""

from typing import Self


class GridDataMismatchError(Exception):
    """Raise if params of the dataset do not match each other."""

    def __init__(self: Self, message: str) -> None:
        self.message = message
        super().__init__(message)


class GridDataMissingError(Exception):
    """If parameters of the dataset are missing for the specific operation."""

    def __init__(self: Self, message: str) -> None:
        self.message = message
        super().__init__(message)


class NotOnTimeError(Exception):
    """When the time point that is searched for is not on the time array."""

    def __init__(self: Self, message: str) -> None:
        self.message = message
        super().__init__(message)


class BadOutputDirError(Exception):
    """When files in the output would be overwritten."""

    def __init__(self: Self, message: str) -> None:
        self.message = message
        super().__init__(message)
