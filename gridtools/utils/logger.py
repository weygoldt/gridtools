"""A simple logger for the gridtools package."""

import logging
from typing import Self
from rich.console import Console

import time
from contextlib import ContextDecorator


class Timer(ContextDecorator):
    """A simple timer class to time the execution of a block of code."""

    def __init__(self: Self, console: Console, message: str) -> None:
        """Initialize the timer."""
        self.console = console
        self.message = message

    def __enter__(self: Self) -> Self:
        """Start the timer."""
        self.start_time = time.time()
        return self

    def __exit__(
            self: Self, exc_type: None, exc_value: None, traceback: None
        ) -> None:
        """Stop the timer and log the elapsed time."""
        elapsed_time = time.time() - self.start_time
        msg = f"[bold]Timing:[/bold] {self.message} - [bold green]Execution time:[/bold green] {elapsed_time:.4f} seconds"
        self.console.log(msg)

        if exc_type is not None:
            self.console.log(
                f"[bold red]Exception:[/bold red] {exc_type.__name__}: {exc_value}"
            )
            self.console.log(f"[bold red]Traceback:[/bold red] {traceback}")


def make_logger(name: str) -> logging.Logger:
    """Create a logger for the gridtools package."""
    file_formatter = logging.Formatter(
        "[ %(levelname)s ] ~ %(asctime)s ~ %(module)s.%(funcName)s: %(message)s" # noqa
    )
    console_formatter = logging.Formatter(
        "[ %(levelname)s ] in %(module)s.%(funcName)s: %(message)s"
    )

    # create stream handler for terminal output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # create stream handler for file output
    file_handler = logging.FileHandler("logfile.log")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    # create script specific logger
    logger = logging.getLogger(name)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger


if __name__ == "__main__":
    mylogger = make_logger(__name__)
    mylogger.debug("This is for debugging!")
    mylogger.info("This is an info.")
    mylogger.warning("This is a warning.")
    mylogger.error("This is an error.")
    mylogger.critical("This is a critical error!")
