import os
import typing

import yaml

from ..logger import makeLogger

logger = makeLogger(__name__)


class ConfLoader:
    """
    Opens a YAML file and unpacks dict keys to instance variables.
    """

    def __init__(self, path: str) -> None:
        with open(path) as file:
            try:
                conf = yaml.safe_load(file)
                for key in conf:
                    setattr(self, key, conf[key])

            except yaml.YAMLError as error:
                logger.error("Failed to open yaml file!")
                raise error


class ListRecordings:
    """
    Lists subdiretories of data recordings in a given root directory to iterate over. Directory names specified as strings in the exclude list are ignored (e.g. directories containing metadata, etc.).
    """

    def __init__(
        self, path: str, exclude: typing.Optional[list] = None
    ) -> None:
        logger.debug("Listing recordings ...")

        # set correct paths and ids based on script setup parameters
        self.dataroot = path
        self.recordings = []

        if exclude is not None:
            self.exclude = exclude
        else:
            self.exclude = []

        # create list of recordings in dataroot
        for recording in os.listdir(self.dataroot):
            if os.path.isdir(os.path.join(self.dataroot, recording)):
                if recording not in self.exclude:
                    self.recordings.append(recording)


def makeOutputdir(path: str) -> str:
    """
    Creates a new directory where the path leads if it does not already exist.

    Parameters
    ----------
    path : string
        path to the new output directory

    Returns
    -------
    string
        path of the newly created output directory
    """

    if os.path.isdir(path) is False:
        os.mkdir(path)
        print("new output directory created")
    else:
        print("using existing output directory")

    return path
