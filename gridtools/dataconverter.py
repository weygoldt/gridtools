#!/usr/bin/env python3

"""
This module contains functions and classes for converting data from one format
to another.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print as rprint
from rich.console import Console
from rich.progress import track

from .datasets import Dataset, load, subset
from .utils.spectrograms import (
    decibel,
    freqres_to_nfft,
    overlap_to_hoplen,
    sint,
    specshow,
    spectrogram,
)

# Notes
# dataset/
# ├── train/
# │   ├── images/
# │   └── labels/
# └── val/
#     ├── images/
#     └── labels/

con = Console()

freq_resolution = 6
overlap_fraction = 0.9
spectrogram_freq_limits = (0, 2200)


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

    # For each fish ...

    # Get frequency point for each time point of the chirps of this fish

    # From the chirp params, estimate bounding box using amp, std and nfft of
    # the spectrogram

    # save bounding box freq and time points to a dataframe with fish id

    # iterate through the raw signal and save a sum of the spectrogram of 10 seconds

    # save the spectrogram as images

    pass


def chirp_bounding_boxes(data: Dataset) -> pd.DataFrame:
    assert hasattr(
        data.com.chirp, "params"
    ), "Dataset must have a chirp attribute with a params attribute"

    nfft = 4096

    # Bounding box adjustment paramters as factors
    time_padding = 0.4
    freq_padding = 0.4

    pad_time = nfft / data.grid.samplerate
    rprint(pad_time)

    boxes = []
    ids = []
    for fish_id in data.track.ids:
        freqs = data.track.freqs[data.track.idents == fish_id]
        times = data.track.times[
            data.track.indices[data.track.idents == fish_id]
        ]
        chirps = data.com.chirp.times[data.com.chirp.idents == fish_id]
        params = data.com.chirp.params[data.com.chirp.idents == fish_id]

        for chirp, param in zip(chirps, params):
            # take the two closest frequency points
            f_closest = freqs[np.argsort(np.abs(times - chirp))[:2]]

            # take the two closest time points
            t_closest = times[np.argsort(np.abs(times - chirp))[:2]]

            # compute the weighted average of the two closest frequency points
            # using the dt between chirp time and sampled time as weights
            f_closest = np.average(f_closest, weights=np.abs(t_closest - chirp))

            # we now have baseline eodf and time point of the chirp. Now
            # we get some parameters from the params to build the bounding box
            # for the chirp
            amp = param[1]
            std = param[2]

            # now define bounding box as center coordinates, width and height
            t_center = chirp
            f_center = f_closest + amp / 2

            # approximate width from std by taking approx 3 stds which should
            # cover 99.7% of the gaussian
            # width = std * 3 + ((std * 3) * time_padding)
            height = amp + (amp * freq_padding)
            width = std + pad_time

            boxes.append((t_center, f_center, width, height))
            ids.append(fish_id)

    df = pd.DataFrame(
        boxes, columns=["t_center", "f_center", "width", "height"]
    )
    df["fish_id"] = ids
    return df


def make_spectrograms(data: Dataset) -> None:
    assert hasattr(data, "grid"), "Dataset must have a grid attribute"

    n_electrodes = data.grid.rec.shape[1]

    # How much time to put into each spectrogram
    time_window = 20  # seconds
    window_overlap = 1  # seconds
    window_overlap_samples = window_overlap * data.grid.samplerate  # samples

    # Spectrogram computation parameters
    nfft = freqres_to_nfft(freq_resolution, data.grid.samplerate)  # samples
    rprint(nfft)
    hop_len = overlap_to_hoplen(overlap_fraction, nfft)  # samples
    chunksize = time_window * data.grid.samplerate  # samples
    n_chunks = np.ceil(data.grid.rec.shape[0] / chunksize).astype(int)

    for chunk_no in track(
        range(n_chunks), description="Computing spectrograms"
    ):
        con.log(f"Chunk {chunk_no + 1} of {n_chunks}")

        # get start and stop indices for the current chunk
        # including some overlap to compensate for edge effects
        # this diffrers for the first and last chunk

        if chunk_no == 0:
            idx1 = sint(chunk_no * chunksize)
            idx2 = sint((chunk_no + 1) * chunksize + window_overlap_samples)
        elif chunk_no == n_chunks - 1:
            idx1 = sint(chunk_no * chunksize - window_overlap_samples)
            idx2 = sint((chunk_no + 1) * chunksize)
        else:
            idx1 = sint(chunk_no * chunksize - window_overlap_samples)
            idx2 = sint((chunk_no + 1) * chunksize + window_overlap_samples)

        # idx1 and idx2 now determine the window I cut out of the raw signal
        # to compute the spectrogram of.

        # compute the time and frequency axes of the spectrogram now that we
        # include the start and stop indices of the current chunk and thus the
        # right start and stop time. The `spectrogram` function does not know
        # about this and would start every time axis at 0.
        spec_times = np.arange(idx1, idx2 + 1, hop_len) / data.grid.samplerate
        spec_freqs = np.arange(0, nfft / 2 + 1) * data.grid.samplerate / nfft

        # create a subset from the grid dataset
        chunk = subset(data, idx1, idx2, mode="index")

        # compute the spectrogram for each electrode of the current chunk
        for el in range(n_electrodes):
            # get the signal for the current electrode
            sig = chunk.grid.rec[:, el]

            # compute the spectrogram for the current electrode
            chunk_spec, _, _ = spectrogram(
                data=sig.copy(),
                samplingrate=data.grid.rec.samplerate,
                nfft=nfft,
                hop_length=hop_len,
            )

            # sum spectrogram over all electrodes
            # the spec is a tensor
            if el == 0:
                spec = chunk_spec
            else:
                spec += chunk_spec

        # normalize spectrogram by the number of electrodes
        # the spec is still a tensor
        spec /= n_electrodes

        # convert the spectrogram to dB
        # .. still a tensor
        spec = decibel(spec)

        # cut off everything outside the upper frequency limit
        # the spec is still a tensor
        spec = spec[
            (spec_freqs >= spectrogram_freq_limits[0])
            & (spec_freqs <= spectrogram_freq_limits[1]),
            :,
        ]
        spec_freqs = spec_freqs[
            (spec_freqs >= spectrogram_freq_limits[0])
            & (spec_freqs <= spectrogram_freq_limits[1])
        ]

        # normalize the spectrogram to zero mean and unit variance
        # the spec is still a tensor
        spec = (spec - spec.mean()) / spec.std()

        # plot the bounding boxes for this chunk
        bboxes = chirp_bounding_boxes(chunk)

        # plot the spectrogram
        _, ax = plt.subplots()
        specshow(
            spec,
            spec_times,
            spec_freqs,
            ax=ax,
            aspect="auto",
            origin="lower",
            cmap="gray",
        )

        for box in bboxes.itertuples():
            ax.add_patch(
                plt.Rectangle(
                    (
                        box.t_center - box.width / 2,
                        box.f_center - box.height / 2,
                    ),
                    box.width,
                    box.height,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
            )
        plt.show()


def main():
    path = pathlib.Path(
        "/home/weygoldt/Projects/mscthesis/data/raw/local/gridsimulations/simulated_grid_000"
    )
    data = load(path, grid=True)
    make_spectrograms(data)
