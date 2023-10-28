#!/usr/bin/env python3

"""
This module contains functions and classes for converting data from one format
to another.
"""

import pathlib

from .datasets import Dataset, load


def chirps_fasterrcnn_trainingdata(data: Dataset) -> None:
    """
    Convert a dataset containing ground truth chirps (simulated or annotated
    by hand) to a dataset of images and bounding boxes for training a
    Faster-RCNN model.

    Parameters
    ----------
    data : Dataset
        The input dataset
    """
    pass
