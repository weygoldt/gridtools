import os

import yaml

from ..logger import makeLogger

logger = makeLogger(__name__)


class ConfLoader:
    """
    Opens a YAML file and unpacks it into the class namespace.
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

    def __init__(self, path, exclude=[], verbose=False) -> None:

        logger.debug("Listing recordings ...")

        # set correct paths and ids based on script setup parameters
        self.verbose = verbose
        self.dataroot = path
        self.exclude = exclude
        self.recordings = []

        # create list of recordings in dataroot
        for recording in os.listdir(self.dataroot):
            if os.path.isdir(os.path.join(self.dataroot, recording)):
                if recording not in self.exclude:
                    self.recordings.append(recording)

# def loadYaml(path: str) -> dict:    

#     with open(path) as file:
#         try:
#             conf = yaml.safe_load(file)
#             return conf

#         except yaml.YAMLError as error:
#             logger.error("Failed to open yaml file!")
#             raise error


def makeOutputdir(path: str):
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

    if os.path.isdir(path) == False:
        os.mkdir(path)
        print("new output directory created")
    else:
        print("using existing output directory")

    return path