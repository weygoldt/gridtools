class GridDataMismatch(Exception):
    """
    GridDataMismatchError to raise an exception if
    parameters of the dataset do not match each other.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class GridDataMissing(Exception):
    """
    To raise an exception if
    parameters of the dataset are missing for the specific operation.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class NotOnTimeError(Exception):
    """
    Error is called when the time point that is searched for is not on the supplied time array.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class BadOutputDir(Exception):
    """
    Error is called when the save directory either contains raw data or is the directory from which the function is called. This prevents accidentally overwriting data.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)