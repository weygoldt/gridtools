#!/usr/bin/env python3

"""
This module contains the preprocessing functions for the gridtools package.
The functions act on just the tracked data and should thus be usable on all 
dataset classes.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from rich import print
from rich.progress import track

from ..utils.datasets import WTPreprocessing
from ..utils.logger import make_logger

logger = make_logger(__name__)


def remove_unassigned_tracks(data: WTPreprocessing) -> WTPreprocessing:
    """Removes unassinged frequencies from the dataset.

    Parameters
    ----------
    data : WTPreprocessing
        The dataset to be processed.

    Returns
    -------
    WTPreprocessing
        The processed dataset.
    """
    logger.info("Removing unassigned IDs from dataset.")

    data.track_indices = np.delete(data.track_indices, np.isnan(data.track_idents))
    data.track_freqs = np.delete(data.track_freqs, np.isnan(data.track_idents))
    data.track_powers = np.delete(data.track_powers, np.isnan(data.track_idents), axis=0)
    data.track_idents = np.delete(data.track_idents, np.isnan(data.track_idents))
    data.ids = np.delete(data.ids, np.isnan(data.ids))
    print("Removed unassigned IDs from dataset.")

    return data


def remove_short_tracks(data: WTPreprocessing, min_length: int) -> WTPreprocessing:
    """Removes tracks shorter than a given threshold.

    Parameters
    ----------
    data : WTPreprocessing
        The dataset to be processed.
    min_length : int
        The minimum length of a track.

    Returns
    -------
    WTPreprocessing
        The processed dataset.
    """
    logger.info("Removing tracks shorter than %s.", min_length)

    counter = 0
    index_ids = np.arange(data.ids.shape[0])
    index_ident = np.arange(data.track_idents.shape[0])
    index_ids_del = []
    index_ident_del = []

    for track_id in track(data.ids, description="Removing short tracks"):
        track_times = data.track_times[data.track_indices[data.track_idents == track_id]]
        dur = track_times[-1] - track_times[0]
        if dur < min_length:
            index_ids_del.extend(index_ids[data.ids == track_id])
            index_ident_del.extend(index_ident[data.track_idents == track_id])
            counter += 1
    
    mask_ident = np.ones(data.track_idents.shape[0], dtype=bool)
    mask_ids = np.ones(data.ids.shape[0], dtype=bool)
    mask_ident[index_ident_del] = False
    mask_ids[index_ids_del] = False

    data.track_indices = data.track_indices[mask_ident]
    data.track_freqs = data.track_freqs[mask_ident]
    data.track_powers = data.track_powers[mask_ident, :]
    data.track_idents = data.track_idents[mask_ident]
    data.ids = data.ids[mask_ids]
    print(f"Removed {counter} short tracks.")

    return data


def remove_low_power_tracks(data: WTPreprocessing, min_power: float) -> WTPreprocessing:
    """Removes tracks with a maximum power below a given threshold.

    Parameters
    ----------
    data : WTPreprocessing
        The dataset to be processed.
    min_power : float
        The minimum power of a track.

    Returns
    -------
    WTPreprocessing
        The processed dataset.
    """
    logger.info("Removing tracks with a maximum power below %s.", min_power)

    counter = 0
    index_ids = np.arange(data.ids.shape[0])
    index_ident = np.arange(data.track_idents.shape[0])
    index_ids_del = []
    index_ident_del = []

    for track_id in track(data.ids, description="Removing bad tracks"):
        track_powers = data.track_powers[data.track_idents == track_id, :]
        if np.max(track_powers) < min_power:
            index_ids_del.extend(index_ids[data.ids == track_id])
            index_ident_del.extend(index_ident[data.track_idents == track_id])
            counter += 1
    
    mask_ident = np.ones(data.track_idents.shape[0], dtype=bool)
    mask_ids = np.ones(data.ids.shape[0], dtype=bool)
    mask_ident[index_ident_del] = False
    mask_ids[index_ids_del] = False

    data.track_indices = data.track_indices[mask_ident]
    data.track_freqs = data.track_freqs[mask_ident]
    data.track_powers = data.track_powers[mask_ident, :]
    data.track_idents = data.track_idents[mask_ident]
    data.ids = data.ids[mask_ids]
    print(f"Removed {counter} tracks with a low maximum power.")

    return data

def remove_poorly_tracked_tracks(data: WTPreprocessing, min_coverage: float) -> WTPreprocessing:
    """Removes tracks with a coverage below a given threshold.
    
    Parameters
    ----------
    data : WTPreprocessing
        The dataset to be processed.
    min_coverage : float
        The minimum coverage of a track.

    Returns
    -------
    WTPreprocessing
        The processed dataset.
    """

    logger.info("Removing tracks with a coverage below %s.", min_coverage)

    counter = 0
    for track_id in track(data.ids, description="Removing poorly tracked tracks"):
        track_times = data.track_times[data.track_indices[data.track_idents == track_id]]
        tmin, tmax = track_times[0], track_times[-1]
        tmin_idx = np.argmin(np.abs(data.track_times - tmin))
        tmax_idx = np.argmin(np.abs(data.track_times - tmax))
        true_times = data.track_times[tmin_idx:tmax_idx]
        tracking_perf = len(track_times) / len(true_times)

        if tracking_perf < min_coverage:
            data.track_indices = np.delete(data.track_indices, data.track_idents == track_id)
            data.track_freqs = np.delete(data.track_freqs, data.track_idents == track_id)
            data.track_powers = np.delete(data.track_powers, data.track_idents == track_id, axis=0)
            data.track_idents = np.delete(data.track_idents, data.track_idents == track_id)
            data.ids = np.delete(data.ids, data.ids == track_id)
            counter += 1
        
    print(f"Removed {counter} tracks with a low coverage.")

    return data