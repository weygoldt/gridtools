#!/usr/bin/env python3

"""
Simulate an electrode grid recording of wave-type weakly electric fish based on 
the parameters from the configuration file.
"""

import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import embed
from rich.console import Console
from rich.progress import track
from scipy.signal import resample

from thunderfish.fakefish import wavefish_eods

from .datasets import (
    ChirpData,
    CommunicationData,
    Dataset,
    GridData,
    WavetrackerData,
    save,
)
from .simulations import (
    MovementParams,
    chirp_model_v4,
    make_grid,
    make_positions,
    make_steps,
)

# from .utils.files import Config
from .utils.filters import lowpass_filter

# conf = Config("config.yml")
con = Console()

np.random.seed(42)
model = chirp_model_v4


def get_random_timestamps(start_t, stop_t, n_timestamps, min_dt):
    """
    Generate an array of random timestamps between start_t and stop_t with a minimum time difference of min_dt.

    Parameters
    ----------
    start_t : float
        The start time for the timestamps.
    stop_t : float
        The stop time for the timestamps.
    n_timestamps : int
        The number of timestamps to generate.
    min_dt : float
        The minimum time difference between timestamps.

    Returns
    -------
    numpy.ndarray
        An array of random timestamps between start_t and stop_t with a minimum time difference of min_dt.
    """
    timestamps = np.sort(np.random.uniform(start_t, stop_t, n_timestamps))
    while True:
        time_diffs = np.diff(timestamps)
        if np.all(time_diffs >= min_dt):
            return timestamps
        else:
            # Resample the timestamps that don't meet the minimum time difference criteria
            invalid_indices = np.where(time_diffs < min_dt)[0]
            num_invalid = len(invalid_indices)
            new_timestamps = np.random.uniform(start_t, stop_t, num_invalid)
            timestamps[invalid_indices] = new_timestamps
            timestamps.sort()


def fakegrid() -> None:
    """
    Simulate a grid of electrodes and fish EOD signals.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    This function simulates a grid of electrodes and fish EOD signals. It generates
    random parameters for a set of chirps, and simulates the EOD signals for a
    random number of fish. The fish are assigned random positions within a
    predefined boundary, and their EOD signals are attenuated by the distance
    between the fish and each electrode on the grid. The resulting signals are
    collected and downsampled for tracking.

    The function saves the simulated signals to disk in the specified output
    directory.
    """

    chirp_param_path = pathlib.Path(
        "/home/weygoldt/Projects/mscthesis/data/processed/chirpsimulations/chirp_fits_interpolated.csv"
    )
    output_path = pathlib.Path(
        "/home/weygoldt/Projects/mscthesis/data/raw/local/grid_simulations"
    )

    ngrids = 100  # number of grids to simulate as long as there are chirps left
    samplerate = 20000  # Hz
    wavetracker_samplingrate = 3
    duration = 300  # s
    min_chirp_dt = 0.5  # s
    max_chirp_freq = 0.5  # Hz
    max_chirp_contrast = 0.6
    eodfrange = np.array([300, 1600])
    simulated_chirp_params = pd.read_csv(chirp_param_path).to_numpy()
    np.random.shuffle(simulated_chirp_params)

    gridx, gridy = make_grid(
        origin=(0, 0),
        shape=(4, 4),
        spacing=0.5,
        style="square",
    )
    nelectrodes = len(np.ravel(gridx))

    for griditer in range(ngrids):
        nfish = np.random.randint(1, 5)
        stop = False

        track_freqs = []
        track_powers = []
        track_idents = []
        track_indices = []
        xpos = []
        ypos = []

        chirp_times = []
        chirp_idents = []
        chirp_params = []
        detector = "gt"

        con.log(f"Grid {griditer}: Simulating {nfish} fish")

        for fishiter in range(nfish):
            eodf = np.random.uniform(eodfrange[0], eodfrange[1])

            # simulate chirps
            nchirps = np.random.randint(1, duration * max_chirp_freq)
            con.log(f"Fish {fishiter}: Simulating {nchirps} chirps")

            # get n_chirps random chirp parameters and delete them from the list
            # of chirp parameters after each iteration to avoid duplicates
            if len(simulated_chirp_params) < nchirps:
                nchirps = len(simulated_chirp_params)
                stop = True

            cp = simulated_chirp_params[:nchirps]
            simulated_chirp_params = simulated_chirp_params[nchirps:]

            # make random chirp times at least max_chirp_dt apart
            ctimes = get_random_timestamps(
                start_t=0,
                stop_t=duration,
                n_timestamps=nchirps,
                min_dt=min_chirp_dt,
            )

            # make chirps with the extracted parameters at these times
            ftrace = np.zeros(duration * samplerate)
            contrasts = np.random.uniform(0, max_chirp_contrast, size=nchirps)
            amtrace = np.ones_like(ftrace)
            time = np.arange(len(ftrace)) / samplerate
            for i, ctime in enumerate(ctimes):
                ftrace += model(
                    time,
                    ctime,
                    *cp[i, 1:],
                )
                amtrace += model(
                    time,
                    ctime,
                    -contrasts[i],
                    *cp[i, 2:],
                )

            # shift the freq trace to the eodf
            ftrace += eodf

            # make an eod from the frequency trace
            eod = wavefish_eods(
                fish="Alepto",
                frequency=ftrace,
                samplerate=samplerate,
                duration=duration,
                phase0=0,
                noise_std=0.001,
            )

            # modulate the eod with the amplitude trace
            eod *= amtrace

            # simulate positions for this fish
            # pick a random origin
            boundaries = (-2, 2, -2, 2)
            origin = (
                np.random.uniform(boundaries[0], boundaries[1]),
                np.random.uniform(boundaries[2], boundaries[3]),
            )
            mvm = MovementParams(
                duration=duration,
                origin=origin,
                boundaries=boundaries,
                target_fs=samplerate,
            )
            t, s = make_steps(mvm)
            x, y = make_positions(t, s, mvm)

            # compute the distance at every position to every electrode
            dists = np.sqrt(
                (x[:, None] - gridx[None, :]) ** 2
                + (y[:, None] - gridy[None, :]) ** 2
            )

            # make the distance sqared and invert it
            dists = -(dists**2)

            # normalize the distances between 0 and 1
            dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))

            # add the fish signal onto all electrodes
            grid_signals = np.tile(eod, (nelectrodes, 1)).T

            # attentuate the signals by the squared distances
            attenuated_signals = grid_signals * dists

            # collect signals
            if fishiter == 0:
                signal = attenuated_signals
            else:
                signal += attenuated_signals

            # downsample the tracking arrays
            num = int(
                np.round(wavetracker_samplingrate / samplerate * len(ftrace))
            )
            f = resample(np.ones_like(ftrace) * eodf, num)
            p = resample(dists, num, axis=0)
            x = resample(x, num)
            y = resample(y, num)

            # filter to remove resampling artifacts, particularly when there
            # are rises
            f = lowpass_filter(f, 10, wavetracker_samplingrate)
            f[f < eodf] = eodf  # for filter oscillations
            p = np.vstack(
                [lowpass_filter(pi, 10, wavetracker_samplingrate) for pi in p.T]
            ).T
            p[p < 0] = 0

            track_freqs.append(f)
            track_powers.append(p)
            track_idents.append(np.ones_like(f) * fishiter)
            track_indices.append(np.arange(len(f)))
            xpos.append(x)
            ypos.append(y)

            chirp_times.append(ctimes)
            chirp_idents.append(np.ones_like(ctimes) * fishiter)
            chirp_params.append(cp)

            if stop:
                break
        if stop:
            con.log(
                f"Grid {griditer}: Stopped after {fishiter} fish, no more chirps left"
            )
            break

        track_freqs = np.concatenate(track_freqs)
        track_powers = np.concatenate(track_powers)
        track_idents = np.concatenate(track_idents)
        track_indices = np.concatenate(track_indices)
        xpos = np.concatenate(xpos)
        ypos = np.concatenate(ypos)
        track_times = np.arange(len(track_freqs)) / wavetracker_samplingrate

        chirp_times = np.concatenate(chirp_times)
        chirp_idents = np.concatenate(chirp_idents)
        chirp_params = np.concatenate(chirp_params)

        # norm the signal between -1 and 1
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

        wt = WavetrackerData(
            freqs=track_freqs,
            powers=track_powers,
            idents=track_idents,
            indices=track_indices,
            ids=np.unique(track_idents),
            times=track_times,
            xpos=xpos,
            ypos=ypos,
        )

        chps = ChirpData(
            times=chirp_times,
            idents=chirp_idents,
            params=chirp_params,
            detector=detector,
        )

        com = CommunicationData(chirp=chps)

        grid = GridData(
            rec=signal,
            samplerate=samplerate,
        )

        path = pathlib.Path(f"{output_path}/simulated_grid_{griditer:03d}")

        data = Dataset(
            path=path,
            grid=grid,
            track=wt,
            com=com,
        )

        if path.exists():
            shutil.rmtree(path)
        save(data, output_path)

        # plot dataset
        _, ax = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)
        for fish_id in np.unique(data.track.ids):
            ax[0].plot(
                data.track.times[
                    data.track.indices[data.track.idents == fish_id]
                ],
                data.track.freqs[data.track.idents == fish_id],
                label=f"Fish {fish_id}",
            )
            ax[1].plot(
                data.track.xpos[data.track.idents == fish_id],
                data.track.ypos[data.track.idents == fish_id],
            )

        ax[0].set_ylim(300, 2000)
        ax[0].set_title("Frequency")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Frequency (Hz)")
        ax[0].legend()
        ax[1].set_xlim(boundaries[0], boundaries[1])
        ax[1].set_ylim(boundaries[2], boundaries[3])
        ax[1].set_title("Position")
        ax[1].set_xlabel("x (m)")
        ax[1].set_ylabel("y (m)")
        plt.savefig(path / "overview.png")
