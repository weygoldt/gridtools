"""Simulate an electrode grid recording of wave-type weakly electric fish."""

import gc
import pathlib
import shutil
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track
from scipy.signal import resample
from thunderfish.fakefish import wavefish_eods

from gridtools.datasets.models import (
    ChirpData,
    CommunicationData,
    Dataset,
    GridData,
    WavetrackerData,
    RiseData,
)
from gridtools.datasets.loaders import load
from gridtools.datasets.savers import save

from .simulations import (
    MovementParams,
    fold_space,
    gaussian,
    interpolate_positions,
    make_grid,
    make_positions,
    make_steps,
)
from gridtools.utils.configfiles import SimulationConfig, load_sim_config
from gridtools.utils.filters import lowpass_filter
from gridtools.simulations.noise import band_limited_noise
from gridtools.simulations.utils import get_random_timestamps
from gridtools.utils.logger import Timer

con = Console()
model = gaussian # Model used for chirp generation




def fakegrid(config: SimulationConfig, output_path: pathlib.Path) -> None:
    """
    Simulate a grid of electrodes and fish EOD signals.

    Parameters
    ----------
    config : SimulationConfig
        The simulation configuration, to be placed in the output_path.
    output_path : pathlib.Path
        Path to save the simulated recordings to.

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
    # general parameters
    ngrids = config.meta.ngrids
    samplerate = config.grid.samplerate
    wavetracker_samplingrate = config.grid.wavetracker_samplerate
    duration = config.grid.duration
    eodfrange = config.fish.eodfrange
    eodfnoise_std = config.fish.eodfnoise_std
    eodfnoise_band = config.fish.eodfnoise_band
    noise_std = config.fish.noise_std
    downsample_lowpass = config.grid.downsample_lowpass

    # chirp parameters
    min_chirp_dt = config.chirps.min_chirp_dt
    max_chirp_freq = config.chirps.max_chirp_freq
    max_chirp_contrast = config.chirps.max_chirp_contrast
    chirpnoise_std = config.chirps.chirpnoise_std
    chirpnoise_band = config.chirps.chirpnoise_band

    # load chirp parameters from model fits
    chirp_param_path = pathlib.Path(config.chirps.chirp_params_path)
    simulated_chirp_params = pd.read_csv(chirp_param_path).to_numpy()
    np.random.shuffle(simulated_chirp_params)

    gridx, gridy = make_grid(
        origin=config.grid.origin,
        shape=config.grid.shape,
        spacing=config.grid.spacing,
        style=config.grid.style,
    )
    nelectrodes = len(np.ravel(gridx))
    boundaries = config.grid.boundaries

    msg = f"Simulating {ngrids} grids with {nelectrodes} electrodes each ..."
    con.log(msg)

    for griditer in range(ngrids):
        nfish = np.random.randint(1, 5)

        # space possible baseline eodfs apart by at least 20 Hz
        eodfs = get_random_timestamps(eodfrange[0], eodfrange[1], nfish, 20)
        stop = False

        track_freqs = []
        track_powers = []
        track_idents = []
        track_indices = []
        xpos_orig = []
        ypos_orig = []
        xpos_fine = []
        ypos_fine = []
        xpos = []
        ypos = []

        chirp_times = []
        chirp_idents = []
        chirp_params = []
        detector = "gt"

        fishiter = 0

        msg = f"Grid {griditer + 1}: Simulating {nfish} fish ..."
        con.log(msg)

        signal = None

        for fishiter in range(nfish):
            eodf = eodfs[fishiter]

            # simulate chirps
            nchirps = np.random.randint(1, int(duration * max_chirp_freq))

            # get n_chirps random chirp parameters and delete them from the list
            # of chirp parameters after each iteration to avoid duplicates
            if len(simulated_chirp_params) < nchirps:
                nchirps = len(simulated_chirp_params)
                stop = True

            cp = simulated_chirp_params[:nchirps]
            simulated_chirp_params = simulated_chirp_params[nchirps:]

            # make random chirp times at least max_chirp_dt apart
            with Timer(con, "Generating chirp times"):
                ctimes = get_random_timestamps(
                    start_t=0,
                    stop_t=duration,
                    n_timestamps=nchirps,
                    min_dt=min_chirp_dt,
                )

            # make chirps with the extracted parameters at these times
            ftrace = np.zeros(int(duration * samplerate))
            contrasts = np.random.uniform(0, max_chirp_contrast, size=nchirps)
            amtrace = np.ones_like(ftrace)
            time = np.arange(len(ftrace)) / samplerate

            with Timer(con, f"Simulating {nchirps} chirps"):
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

            # make chirps particularly noisy
            # to do this make noise and multiply it with the chirp
            with Timer(con, "Adding noise to frequency trace"):
                chirpnoise = (
                    band_limited_noise(
                        chirpnoise_band[0],
                        chirpnoise_band[1],
                        len(ftrace),
                        samplerate,
                        1,
                    )
                    * ftrace
                )
                # now scale it back down because chirps are too strong
                chirpnoise = (chirpnoise - np.mean(chirpnoise)) / np.std(chirpnoise)
                chirpnoise *= chirpnoise_std

                # add the noise to the frequency trace
                ftrace += chirpnoise

                # add band limited noise to the frequency trace
                ftrace += band_limited_noise(
                    eodfnoise_band[0],
                    eodfnoise_band[1],
                    len(ftrace),
                    samplerate,
                    eodfnoise_std,
                )  # EOD fluctuations

                # shift the freq trace to the eodf
                ftrace += eodf

            # make an eod from the frequency trace
            with Timer(con, "Simulating EOD"):
                eod = wavefish_eods(
                    fish="Alepto",
                    frequency=ftrace,
                    samplerate=samplerate,
                    duration=duration,
                    phase0=0,
                    noise_std=noise_std,
                )

            # modulate the eod with the amplitude trace
            eod *= amtrace

            # simulate positions for this fish
            # pick a random origin
            origin = (
                np.random.uniform(boundaries[0], boundaries[1]),
                np.random.uniform(boundaries[2], boundaries[3]),
            )
            # check if duration and samplerate can make int number of samples
            if (duration * samplerate) % 1 != 0:
                msg = (
                    "Duration and samplerate must make an integer number of samples"
                    "Please modify the parameters in the config file so"
                    "that the movement target fs multiplied by the duration"
                    "is an integer."
                )
                raise ValueError(msg)
            duration = int(duration)
            samplerate = int(samplerate)
            mvm = MovementParams(
                duration=duration,
                origin=origin,
                boundaries=boundaries,
                target_fs=samplerate,
            )

            with Timer(con, "Simulating positions"):
                origx = np.random.uniform(boundaries[0], boundaries[1])
                origy = np.random.uniform(boundaries[2], boundaries[3])
                con.log(f"Random origin: {origx}, {origy}")
                trajectories, steps = make_steps(mvm)
                x, y = make_positions(trajectories, steps, (origx, origy))

            con.log(f"Max x: {np.max(x)}")
            con.log(f"Min x: {np.min(x)}")
            con.log(f"Max y: {np.max(y)}")
            con.log(f"Min y: {np.min(y)}")
            con.log(f"Bounaries: {boundaries}")
            con.log(f"Grid xmin and xmax: {np.min(gridx)} {np.max(gridx)}")
            con.log(f"Grid ymin and ymax: {np.min(gridy)} {np.max(gridy)}")

            with Timer(con, "Folding space"):
                boundaries = np.array(boundaries)
                x_orig, y_orig = fold_space(x, y, boundaries)

            with Timer(con, "Interpolating positions"):
                x_fine, y_fine = interpolate_positions(
                    x_orig, y_orig, duration, mvm.measurement_fs, mvm.target_fs
                )

            with Timer(con, "Attenuating signals with distance to electrodes"):
                # compute the distance at every position to every electrode
                dists = np.sqrt(
                    (x_fine[:, None] - gridx[None, :]) ** 2
                    + (y_fine[:, None] - gridy[None, :]) ** 2
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
            # signal = None
            if fishiter == 0:
                signal = attenuated_signals
            else:
                signal += attenuated_signals

            with Timer(con, "Downsampling signals"):
                # downsample the tracking arrays
                num = int(
                    np.round(wavetracker_samplingrate / samplerate * len(ftrace))
                )
                f = resample(np.ones_like(ftrace) * eodf, num)
                p = resample(dists, num, axis=0)
                x = resample(x_orig, num)
                y = resample(y_orig, num)

                # filter to remove resampling artifacts, particularly when there
                # are rises
                f = lowpass_filter(f, downsample_lowpass, wavetracker_samplingrate)
                f[f < eodf] = eodf  # for filter oscillations
                p = np.vstack(
                    [
                        lowpass_filter(
                            pi, downsample_lowpass, wavetracker_samplingrate
                        )
                        for pi in p.T
                    ]
                ).T
                p[p < 0] = 0

            track_freqs.append(f)
            track_powers.append(p)
            track_idents.append(np.ones_like(f) * fishiter)
            track_indices.append(np.arange(len(f)))
            xpos.append(x)
            ypos.append(y)
            xpos_orig.append(x_orig)
            ypos_orig.append(y_orig)
            xpos_fine.append(x_fine)
            ypos_fine.append(y_fine)

            chirp_times.append(ctimes)
            chirp_idents.append(np.ones_like(ctimes) * fishiter)
            chirp_params.append(cp)

            msg = f"Grid {griditer}: Simulated {fishiter + 1} fish"
            con.log(msg)

            if stop:
                break
        if stop:
            msg = f"Grid {griditer}: Stopped after {fishiter} fish, no more chirps left"
            con.log(msg)
            break

        with Timer(con, "Concatenating arrays"):
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
            has_positions=True,
        )

        chps = ChirpData(
            times=chirp_times,
            idents=chirp_idents,
            params=chirp_params,
            detector=detector,
            are_detected=True,
            have_params=True,
        )

        rs = RiseData(
            times=np.array([]),
            idents=np.array([]),
            params=np.array([]),
            detector="None",
            are_detected=False,
            have_params=False,
        )

        com = CommunicationData(
            chirp=chps,
            rise=rs,
            are_detected=True,
        )

        grid = GridData(
            rec=signal,
            samplerate=samplerate,
            shape=signal.shape,
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
        for idx, fish_id in enumerate(np.unique(data.track.ids)):
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
            ax[1].plot(
                xpos_orig[idx],
                ypos_orig[idx],
                color="black",
                linestyle="dashed",
                linewidth=1,
                alpha=0.5,
                label="Original",
            )
            ax[1].plot(
                xpos_fine[idx],
                ypos_fine[idx],
                color="black",
                linestyle="dotted",
                linewidth=1,
                alpha=0.5,
                label="Interpolated",
            )

        ax[1].scatter(
            gridx,
            gridy,
            color="black",
            marker="o",
            label="Electrodes",
        )

        ax[0].set_ylim(300, 2000)
        ax[0].set_title("Frequency")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Frequency (Hz)")
        ax[0].legend()
        ax[1].set_xlim(boundaries[0], boundaries[2])
        ax[1].set_ylim(boundaries[1], boundaries[3])
        ax[1].set_title("Position")
        ax[1].set_xlabel("x (m)")
        ax[1].set_ylabel("y (m)")
        plt.savefig(path / "overview.png")
        plt.show()


def augment_grid(sd: Dataset, rd: Dataset) -> Dataset:
    """Augment a simulated dataset with a random snippet from a real recording.

    Parameters
    ----------
    sd : Dataset
        The simulated dataset to augment.
    rd : Dataset
        The real recording dataset to use for augmentation.

    Returns
    -------
    Dataset
        The augmented dataset.

    Raises
    ------
    GridDataMismatch
        If the number of electrodes in the simulated dataset is larger than
        the number of electrodes in the real dataset.

    Notes
    -----
    This function normalizes the simulated dataset, takes a random snippet from the
    real recording, scales the simulated dataset to match the real recording, and
    combines the two datasets. This introduces realistic background noise to the
    simulated dataset. A future version should choose a snipped from the real
    recording that contains as little communication as possible.
    """
    # normalize the simulated dataset
    sd.track.powers = (sd.track.powers - np.mean(sd.track.powers)) / np.std(
        sd.track.powers
    )
    sd.grid.rec = (sd.grid.rec[:] - np.mean(sd.grid.rec[:])) / np.std(
        sd.grid.rec[:]
    )

    # take a random snippet from the real recording
    target_shape = sd.grid.rec.shape
    source_shape = rd.grid.rec.shape

    # check if number of electrodes of the simulated dataset is larger
    # than the number of electrodes of the real dataset
    if target_shape[1] > source_shape[1]:
        raise GridDataMismatch(
            "The number of electrodes of the simulated dataset is larger than the number of electrodes of the real dataset."
        )

    random_electrodes = np.random.choice(
        np.arange(source_shape[1]), size=target_shape[1], replace=False
    )
    # random_start = np.random.randint(
    #     0, source_shape[0] - target_shape[0], size=1
    # )[0]
    # random_end = random_start + target_shape[0]

    # get a window where little chirps are produced
    chirps = np.sort(rd.com.chirp.times)
    cdiffs = np.diff(chirps)

    # get start and stop time of the longest chirpless window
    start, stop = np.argmax(cdiffs), np.argmax(cdiffs) + 1
    start, stop = chirps[start], chirps[stop]
    time = np.arange(rd.grid.rec.shape[0]) / rd.grid.samplerate
    index = np.where((time >= start) & (time <= stop))[0]

    if len(index) < target_shape[0]:
        return None

    # now get random snippet from the indices
    random_start = np.random.randint(
        index[0], index[-1] - target_shape[0], size=1
    )[0]
    random_end = random_start + target_shape[0]

    real_snippet = rd.grid.rec[random_start:random_end, random_electrodes]

    # get mean and std of the real recording
    mean_power, std_power = np.nanmean(rd.track.powers), np.nanstd(
        rd.track.powers
    )
    mean_amp, std_amp = np.mean(real_snippet), np.std(real_snippet)

    # scale the simulated dataset to match the real recording
    sd.track.powers = sd.track.powers * std_power + mean_power
    sd.grid.rec = sd.grid.rec * std_amp + mean_amp

    # combine the simulated dataset with the real recording
    sd.grid.rec = sd.grid.rec + real_snippet

    del rd

    # save the hybrid dataset
    return sd


def hybridgrid(
    fakegrid_path: pathlib.Path,
    realgrid_path: pathlib.Path,
    save_path: pathlib.Path,
) -> None:
    """
    Take a simulated dataset and add realistic background noise to it.

    Parameters
    ----------
    fakegrid_path : pathlib.Path
        The path to the simulated dataset.
    realgrid_path : pathlib.Path
        The path to the real dataset.
    save_path : pathlib.Path
        The path to save the hybrid dataset to.

    Returns
    -------
    None

    Notes
    -----
    This function takes a simulated dataset and adds realistic background noise
    to it. It adds realistic background noise to a simulated dataset and scales
    it to match a real recording by combining the simulated dataset with a real
    recording.

    The function saves the hybrid dataset to disk in the specified output
    directory.
    """
    # list subdirectories of fakegrid_path
    fake_datasets = list(fakegrid_path.iterdir())
    real_datasets = list(realgrid_path.iterdir())

    # load csv with notes
    chirp_notes = pd.read_csv(
        realgrid_path / "chirp_notes.csv",
    )

    # get indices 25 to 29 as real datasets
    # real_datasets = chirp_notes.iloc[25:30]["recording"]
    real_datasets = chirp_notes["recording"]
    real_datasets = [realgrid_path / rd for rd in real_datasets]

    for fake_dataset in track(fake_datasets):
        fake_dataset = load(fake_dataset, grid=True)
        real_dataset = load(np.random.choice(real_datasets), grid=True)
        augmented_fake_dataset = augment_grid(fake_dataset, real_dataset)

        # I use a while loop to grab random snippets from random real datasets
        # until I find one that has a long enough chirpless window
        while augmented_fake_dataset is None:
            real_dataset = load(np.random.choice(real_datasets), grid=True)
            augmented_fake_dataset = augment_grid(fake_dataset, real_dataset)

            msg = "No valid snippet found, trying again..."
            con.log(msg)

            del real_dataset

        save(augmented_fake_dataset, save_path)

        del fake_dataset
        del real_dataset
        del augmented_fake_dataset
        gc.collect()


def fakegrid_cli(output_path):
    """
    Command line interface for the fakegrid function.
    """
    config_path = (pathlib.Path(output_path) / "gridtools_simulations.toml",)
    config_path = config_path[0].resolve()
    config = load_sim_config(str(config_path))

    fakegrid(
        config=config,
        output_path=output_path,
    )


def hybridgrid_cli(input_path, real_path, output_path):
    """
    Run the hybridgrid simulation using the command line interface.
    """
    input_path = pathlib.Path(input_path)
    real_path = pathlib.Path(real_path)
    output_path = pathlib.Path(output_path)

    if output_path.exists():
        shutil.rmtree(output_path)

    hybridgrid(
        fakegrid_path=input_path,
        realgrid_path=real_path,
        save_path=output_path,
    )
