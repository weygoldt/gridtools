"""Contains the preprocessing functions for the gridtools package.

The functions act on just the tracked data and should thus be usable on all
dataset classes, whether they contain raw data or chirp data or not.
"""

import numpy as np
from rich import print as rprint
from rich.progress import track

from .datasets import Dataset
from .utils.logger import make_logger

logger = make_logger(__name__)


def remove_unassigned_tracks(data: Dataset) -> Dataset:
    """Remove unassinged frequencies from the dataset.

    Parameters
    ----------
    data : Dataset
        The dataset to be processed.

    Returns
    -------
    WTPreprocessing
        The processed dataset.
    """
    logger.info("Removing unassigned IDs from dataset.")

    data.track.indices = np.delete(
        data.track.idents, np.isnan(data.track.idents)
    )
    data.track.freqs = np.delete(data.track.freqs, np.isnan(data.track.idents))
    data.track.powers = np.delete(
        data.track.powers, np.isnan(data.track.idents), axis=0
    )
    data.track.idents = np.delete(
        data.track.idents, np.isnan(data.track.idents)
    )
    data.track.ids = np.delete(data.track.ids, np.isnan(data.track.ids))
    rprint("Removed unassigned IDs from dataset.")

    return data


def remove_short_tracks(data: Dataset, min_length: int) -> Dataset:
    """Remove tracks shorter than a given threshold.

    Parameters
    ----------
    data : Dataset
        The dataset to be processed.
    min_length : int
        The minimum length of a track.

    Returns
    -------
    Dataset
        The processed dataset.
    """
    logger.info("Removing tracks shorter than %s.", min_length)

    counter = 0
    index_ids = np.arange(data.track.ids.shape[0])
    index_ident = np.arange(data.track.idents.shape[0])
    index_ids_del = []
    index_ident_del = []

    for track_id in track(data.track.ids, description="Removing short tracks"):
        track_times = data.track.times[
            data.track.indices[data.track.idents == track_id]
        ]
        dur = track_times[-1] - track_times[0]
        if dur < min_length:
            index_ids_del.extend(index_ids[data.track.ids == track_id])
            index_ident_del.extend(index_ident[data.track.idents == track_id])
            counter += 1

    mask_ident = np.ones(data.track.idents.shape[0], dtype=bool)
    mask_ids = np.ones(data.track.ids.shape[0], dtype=bool)
    mask_ident[index_ident_del] = False
    mask_ids[index_ids_del] = False

    data.track.indices = data.track.indices[mask_ident]
    data.track.freqs = data.track.freqs[mask_ident]
    data.track.powers = data.track.powers[mask_ident, :]
    data.track.idents = data.track.idents[mask_ident]
    data.track.ids = data.track.ids[mask_ids]
    rprint(f"Removed {counter} short tracks.")
    return data


def remove_low_power_tracks(data: Dataset, min_power: float) -> Dataset:
    """Remove tracks with a maximum power below a given threshold.

    Parameters
    ----------
    data : Dataset
        The dataset to be processed.
    min_power : float
        The minimum power of a track.

    Returns
    -------
    Dataset
        The processed dataset.
    """
    logger.info("Removing tracks with a maximum power below %s.", min_power)

    counter = 0
    index_ids = np.arange(data.track.ids.shape[0])
    index_ident = np.arange(data.track.idents.shape[0])
    index_ids_del = []
    index_ident_del = []

    for track_id in track(data.track.ids, description="Removing bad tracks"):
        track_powers = data.track.powers[data.track.idents == track_id, :]
        if np.max(track_powers) < min_power:
            index_ids_del.extend(index_ids[data.track.ids == track_id])
            index_ident_del.extend(index_ident[data.track.idents == track_id])
            counter += 1

    mask_ident = np.ones(data.track.idents.shape[0], dtype=bool)
    mask_ids = np.ones(data.track.ids.shape[0], dtype=bool)
    mask_ident[index_ident_del] = False
    mask_ids[index_ids_del] = False

    data.track.indices = data.track.indices[mask_ident]
    data.track.freqs = data.track.freqs[mask_ident]
    data.track.powers = data.track.powers[mask_ident, :]
    data.track.idents = data.track.idents[mask_ident]
    data.track.ids = data.track.ids[mask_ids]
    rprint(f"Removed {counter} tracks with a low maximum power.")
    return data


def remove_poorly_tracked_tracks(
        data: Dataset, min_coverage: float
) -> Dataset:
    """Remove tracks with a coverage below a given threshold.

    Parameters
    ----------
    data : Dataset
        The dataset to be processed.
    min_coverage : float
        The minimum coverage of a track.

    Returns
    -------
    Dataset
        The processed dataset.
    """
    logger.info("Removing tracks with a coverage below %s.", min_coverage)

    counter = 0
    for track_id in track(
        data.track.ids, description="Removing poorly tracked tracks"
    ):
        track_times = data.track.times[
            data.track.indices[data.track.idents == track_id]
        ]
        tmin, tmax = track_times[0], track_times[-1]
        tmin_idx = np.argmin(np.abs(data.track.times - tmin))
        tmax_idx = np.argmin(np.abs(data.track.times - tmax))
        true_times = data.track.times[tmin_idx:tmax_idx]
        tracking_perf = len(track_times) / len(true_times)

        if tracking_perf < min_coverage:
            data.track.indices = np.delete(
                data.track.indices, data.track.idents == track_id
            )
            data.track.freqs = np.delete(
                data.track.freqs, data.track.idents == track_id
            )
            data.track.powers = np.delete(
                data.track.powers, data.track.idents == track_id, axis=0
            )
            data.track.idents = np.delete(
                data.track.idents, data.track.idents == track_id
            )
            data.track.ids = np.delete(
                data.track.ids, data.track.ids == track_id
            )
            counter += 1

    rprint(f"Removed {counter} tracks with a low coverage.")

    return data
