"""Simulate an electrode grid recording of wave-type weakly electric fish."""

import gc
import pathlib
import shutil
from typing import Callable, Self

import numpy as np
from rich.console import Console
from scipy.ndimage import median_filter
from scipy.signal import resample
from thunderfish.fakefish import wavefish_eods

from gridtools.datasets.loaders import load
from gridtools.datasets.models import (
    ChirpData,
    CommunicationData,
    Dataset,
    GridData,
    GridDataMismatchError,
    RiseData,
    WavetrackerData,
)
from gridtools.datasets.savers import save
from gridtools.simulations.communication import (
    biphasic_chirp,
    monophasic_chirp,
)
from gridtools.simulations.eod import ChirpGenerator
from gridtools.simulations.movement import (
    MovementParams,
    fold_space,
    interpolate_positions,
    make_grid,
    make_positions,
    make_steps,
)
from gridtools.simulations.noise import band_limited_noise
from gridtools.simulations.utils import get_random_timestamps
from gridtools.simulations.visualizations import (
    plot_freq_tracks,
    plot_positions,
)
from gridtools.utils.configfiles import SimulationConfig, load_sim_config
from gridtools.utils.filters import lowpass_filter
from gridtools.utils.logger import Timer

con = Console()
rng = np.random.default_rng(42)


class GridSimulator:
    """Simulate a synthetic grid recording."""

    def __init__(
        self: Self,
        config: SimulationConfig,
        chirper: ChirpGenerator,
        output_path: pathlib.Path,
        verbosity: int = 0,
    ) -> None:
        """Initialize fake grid."""
        self.config = config
        self.chirper = chirper
        self.output_path = output_path
        self.verbosity = verbosity
        self.gridx, self.gridy = make_grid(
            origin=self.config.grid.origin,
            shape=self.config.grid.shape,
            spacing=self.config.grid.spacing,
            style=self.config.grid.style,
        )
        self.nelectrodes = len(np.ravel(self.gridx))
        # chirp_data_path = pathlib.Path(self.config.chirps.chirp_params_path)
        # self.chirp_params = pd.read_csv(chirp_data_path).to_numpy()
        # rng.shuffle(self.chirp_params)

        msg = f"with {self.nelectrodes} electrodes each ..."
        con.log(msg)

    @property
    def chirp_model(self: Self) -> Callable:
        """Select the chirp model specified in the config file."""
        if self.config.chirps.model == "monophasic":
            return monophasic_chirp
        if self.config.chirps.model == "biphasic":
            return biphasic_chirp
        msg = f"Chirp model {self.config.chirps.model} not supported."
        raise ValueError(msg)

    @property
    def nfish(self: Self) -> int:
        """Get the number of fish to simulate."""
        lowerbound = self.config.fish.nfish[0]
        upperbound = self.config.fish.nfish[1] + 1
        nfish = rng.integers(lowerbound, upperbound)
        if self.verbosity > 1:
            msg = f"Simulating {nfish} fish."
            con.log(msg)
        return nfish

    def run_simulations(self: Self) -> None:
        """Run the grid simulations."""
        for griditer in range(self.config.meta.ngrids)[17:]:
            self.make_grid(griditer)
            gc.collect()

    def make_grid(self: Self, griditer: int) -> None:
        """Simulate a grid of electrodes and fish EOD signals."""
        nfish = self.nfish

        eodfs = get_random_timestamps(
            self.config.fish.eodfrange[0],
            self.config.fish.eodfrange[1],
            nfish,
            self.config.fish.min_delta_eodf,
        )

        stop = False  # stop simulation when no chirp params are left
        track_freqs = []  # track frequencies are stored here
        track_freqs_orig = []  # track frequencies are stored here
        track_powers = []  # track powers are stored here
        track_idents = []  # track idents are stored here
        track_indices = []  # fish indices for time array are stored here
        xpos_orig = []  # Original 30 Hz fish positions
        ypos_orig = []  # Original 30 Hz fish positions
        xpos_fine = []  # fish positions are stored after upsampling to 20kHz
        ypos_fine = []  # fish positions are stored after upsampling to 20kHz
        xpos = []  # fish positions are stored here after downsampling to 3Hz
        ypos = []  # fish positions are stored here after downsampling to 3Hz
        chirp_times = []  # chirp times are stored here
        chirp_idents = []  # fish idents are stored here
        chirp_params = []  # chirp parameters are stored here
        detector = self.config.chirps.detector_str  # how to name chirp file
        signal = np.array([])  # Here we store the electric signal later

        for fishiter in range(nfish):
            # this is the baseline eodf of the current fishs
            eodf = eodfs[fishiter]

            # initialize frequency trace
            # ftrace = np.zeros(
            #     int(self.config.grid.duration * self.config.grid.samplerate)
            # )

            # initialize the am trace
            # amtrace = np.ones_like(ftrace)

            # initialize the time array
            # time = np.arange(len(ftrace)) / self.config.grid.samplerate

            if self.verbosity > 1:
                msg = f"Fish {fishiter + 1} gets EODf of {eodf}."
                con.log(msg)

            ### Chirps --------------------------------------------------------
            con.log("Generating chirps ...")
            ctimes, cp, ftrace, amtrace = self.chirper()
            con.log("Chirps generated.")

            # # choose a random number of chirps for this fish
            # nchirps = rng.integers(
            #     1, int(self.config.grid.duration * self.config.chirps.max_chirp_freq)
            # )
            #
            # # Check how many chirp params left and delete used from array
            # if len(self.chirp_params) < nchirps:
            #     nchirps = len(self.chirp_params)
            #     stop = True
            # cp = self.chirp_params[:nchirps]
            # self.chirp_params = self.chirp_params[nchirps:]
            #
            # if self.verbosity > 1:
            #     msg = f"Fish {fishiter + 1} gets {nchirps} chirps."
            #     con.log(msg)
            #
            # # generate random time stamp for each chirp that follows some rules
            # with Timer(con, "Generating chirp times", self.verbosity):
            #     ctimes = get_random_timestamps(
            #         start_t=0,
            #         stop_t=self.config.grid.duration,
            #         n_timestamps=nchirps,
            #         min_dt=self.config.chirps.min_chirp_dt,
            #     )
            #
            # # choose a random contrast for each chirp
            # contrasts = rng.uniform(
            #     0, self.config.chirps.max_chirp_contrast, size=nchirps
            # )
            #
            # # evaluate the chirp model for every chirp parameter at every
            # # chirp time and just use the but flipped as the amplitude trace
            # with Timer(con, f"Simulating {nchirps} chirps", self.verbosity):
            #     for i, ctime in enumerate(ctimes):
            #         ftrace += self.chirp_model(
            #             time,
            #             ctime,
            #             *cp[i, 1:],
            #         )
            #         amtrace += self.chirp_model(
            #             time,
            #             ctime,
            #             -contrasts[i],
            #             *cp[i, 2:],
            #         )

            ### Augment with noise --------------------------------------------
            # add noise to the frequency trace
            with Timer(con, "Adding noise to frequency trace", self.verbosity):
                # make noise strong at chirps by multiplying it with the chirp
                chirpnoise = (
                    band_limited_noise(
                        self.config.chirps.chirpnoise_band[0],
                        self.config.chirps.chirpnoise_band[1],
                        len(ftrace),
                        self.config.grid.samplerate,
                        1,
                    )
                    # * ftrace
                )

                # scale back to std of 1
                chirpnoise = (chirpnoise - np.mean(chirpnoise)) / np.std(
                    chirpnoise
                )

                # add std from config
                chirpnoise *= self.config.chirps.chirpnoise_std

                # add it to the chirps
                ftrace += chirpnoise

                # also modulate baseline eodf with noise
                ftrace += band_limited_noise(
                    self.config.fish.eodfnoise_band[0],
                    self.config.fish.eodfnoise_band[1],
                    len(ftrace),
                    self.config.grid.samplerate,
                    self.config.fish.eodfnoise_std,
                )

                # import matplotlib.pyplot as plt
                # import matplotlib as mpl
                # mpl.use("TkAgg")
                # plt.plot(ftrace)
                # plt.plot(chirpnoise)
                # plt.show()
                # plt.plot(amtrace)
                # plt.show()

                # shift the frequency trace up to the baseline eodf of fish
                ftrace += eodf

                # store the original frequency trace
                track_freqs_orig.append(ftrace)

            ### Build the EOD -------------------------------------------------
            # make the eod
            with Timer(con, "Simulating EOD", self.verbosity):
                eod = wavefish_eods(
                    fish="Alepto",
                    frequency=ftrace,
                    samplerate=self.config.grid.samplerate,
                    duration=self.config.grid.duration,
                    phase0=0,
                    noise_std=self.config.fish.noise_std,
                )

            # modulate the eod with the amplitude trace
            eod *= amtrace

            ### Build the positions -------------------------------------------
            # now lets make the positions
            # pick a random initial position for the fish
            origin = (
                rng.uniform(
                    low=self.config.grid.boundaries[0],
                    high=self.config.grid.boundaries[2],
                    size=1,
                )[0],
                rng.uniform(
                    low=self.config.grid.boundaries[1],
                    high=self.config.grid.boundaries[3],
                    size=1,
                )[0],
            )
            if self.verbosity > 1:
                msg = f"Random origin: {origin}"
                con.log(msg)

            # This is important: Check if duration and samplerate can make
            # int number of samples
            if (
                self.config.grid.duration * self.config.grid.samplerate
            ) % 1 != 0:
                msg = (
                    "Duration and samplerate must make an integer number of "
                    "samples. Please modify the parameters in the config file "
                    "so that the movement target fs multiplied by the "
                    "duration is an integer."
                )
                raise ValueError(msg)

            # make a movement params object. This is a class containing
            # parameters to simulate movements
            mvm = MovementParams(
                duration=int(self.config.grid.duration),
                origin=origin,
                boundaries=self.config.grid.boundaries,
                target_fs=int(self.config.grid.samplerate),
            )

            # Now simulate the positions
            with Timer(con, "Simulating positions", self.verbosity):
                trajectories, steps = make_steps(mvm)
                x, y = make_positions(trajectories, steps, origin)

            # Now restrict generated positions to the boundaries
            # of the simulated world
            with Timer(con, "Folding space", self.verbosity):
                boundaries = np.array(self.config.grid.boundaries)
                x_orig, y_orig = fold_space(x, y, boundaries)

            # Now interpoalte the positions to sampling rate of grid recording
            # so that we can use the positions to modulate the AM of the
            # raw signal with distance to electrodes
            with Timer(con, "Interpolating positions", self.verbosity):
                x_fine, y_fine = interpolate_positions(
                    x_orig,
                    y_orig,
                    self.config.grid.duration,
                    mvm.measurement_fs,
                    mvm.target_fs,
                )

            ### Combine positions and EOD -------------------------------------
            # Now attenuate the signals with distance to electrodes

            # TODO: This is where a dipole model should be introduced.
            # With the current version, a fish is just a monopole

            # TODO: This is where different conductivities
            # Should be introduced.

            with Timer(
                con,
                "Attenuating signals with distance to electrodes",
                self.verbosity,
            ):
                # Compute distance between fish and each electrode
                # for every point in time
                dists = np.sqrt(
                    (x_fine[:, None] - self.gridx[None, :]) ** 2
                    + (y_fine[:, None] - self.gridy[None, :]) ** 2
                )

                # Square the distance as field decreases with distance squared
                # and invert the distance as larger distance means smaller
                # field
                dists = -(dists**2)  # Add term for conductivity maybe here

                # Normalize the distances between 0 and 1 (this also
                # needs to change when we introduce conductivity)
                dists = (dists - np.min(dists)) / (
                    np.max(dists) - np.min(dists)
                )

                # Add the fish signal onto all electrodes
                # This essentially just copies the fish signal for n
                # electrodes
                grid_signals = np.tile(eod, (self.nelectrodes, 1)).T

                # Attentuate the signals by the squared distances
                attenuated_signals = grid_signals * dists

            # Collect signals
            if fishiter == 0:
                signal = attenuated_signals
            else:
                signal += attenuated_signals

            ### Downsample the signals to resemble wave tracker ---------------
            # Downsample the tracking arrays i.e. frequency tracks, powers,
            # postions, etc. so that they have the same resolution as the
            # output of the wavetracker
            with Timer(con, "Downsampling tracking arrays", self.verbosity):
                num = int(
                    np.round(
                        self.config.grid.wavetracker_samplerate
                        / self.config.grid.samplerate
                        * len(ftrace)
                    )
                )
                f = resample(ftrace, num)
                f = median_filter(f, 100)
                p_unfilt = resample(dists, num, axis=0)
                x = resample(x_orig, num)
                y = resample(y_orig, num)

                # also lowpass filter the powers
                p = np.vstack(
                    [
                        lowpass_filter(
                            pi,
                            self.config.grid.downsample_lowpass,
                            self.config.grid.wavetracker_samplerate,
                        )
                        for pi in p_unfilt.T
                    ]
                ).T
                # After filtering some vals can be negative
                p[p < 0] = 0  # rectify negative values (powers should be 0-1)

            # and now save everything
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

            msg = f"Grid {griditer}: Finished simulating fish {fishiter + 1}"
            con.log(msg)

            if stop:
                break

        # Now concatenate all the arrays
        with Timer(con, "Concatenating arrays", self.verbosity):
            track_freqs_concat = np.concatenate(track_freqs)
            track_powers = np.concatenate(track_powers)
            track_idents = np.concatenate(track_idents)
            track_indices = np.concatenate(track_indices)
            xpos_concat = np.concatenate(xpos)
            ypos_concat = np.concatenate(ypos)
            track_times = (
                np.arange(len(track_freqs[0]))
                / self.config.grid.wavetracker_samplerate
            )
            chirp_times = np.concatenate(chirp_times)
            chirp_idents = np.concatenate(chirp_idents)
            chirp_params = np.concatenate(chirp_params)

        # Normalize signal on electrodes between -1 and 1
        # (as we added many EODs this can get large) but if we dont norm it
        # everything outside is clipped by the saving function
        with Timer(con, "Normalizing signal", self.verbosity):
            signal = (signal - np.min(signal)) / (
                np.max(signal) - np.min(signal)
            )

        if self.verbosity > 0:
            msg = "Assembling dataset ..."
            con.log(msg)

        # Assemble the dataset class to be able to use the saving functions
        print(f"track_freqs_concat: {track_freqs_concat.shape}")
        print(f"track_powers: {track_powers.shape}")
        print(f"track_idents: {track_idents.shape}")
        print(f"track_indices: {track_indices.shape}")
        print(f"track_times: {track_times.shape}")
        print(f"xpos_concat: {xpos_concat.shape}")
        print(f"ypos_concat: {ypos_concat.shape}")

        wt = WavetrackerData(
            freqs=track_freqs_concat,
            powers=track_powers,
            idents=track_idents,
            indices=track_indices,
            ids=np.unique(track_idents),
            times=track_times,
            xpos=xpos_concat,
            ypos=ypos_concat,
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
            samplerate=self.config.grid.samplerate,
            shape=signal.shape,
        )

        path = pathlib.Path(
            f"{self.output_path}/simulated_grid_{griditer:03d}"
        )

        data = Dataset(
            path=path,
            grid=grid,
            track=wt,
            com=com,
        )

        if path.exists():
            msg = f"Grid {griditer}: Removing existing directory"
            con.log(msg)
            shutil.rmtree(path)

        with Timer(con, "Saving dataset", self.verbosity):
            save(data, self.output_path)

        with Timer(con, "Plotting dataset", self.verbosity):
            pltpath = path / "plots"
            pltpath.mkdir(parents=True, exist_ok=True)
            plot_positions(
                original=(xpos_orig, ypos_orig),
                upsampled=(xpos_fine, ypos_fine),
                downsampled=(xpos, ypos),
                grid=(self.gridx, self.gridy),
                boundaries=self.config.grid.boundaries,
                path=pltpath / "positions.pdf",
            )
            plot_freq_tracks(
                track_freqs,
                track_times,
                track_freqs_orig,
                pltpath / "frequency_tracks.pdf",
            )

        if stop:
            msg = f"Grid {griditer}: Stopped after {fishiter} fish, no more chirps left"
            con.log(msg)


class GridAugmenter:
    """Scales simulated grids to measured values and adds realistic noise."""

    def __init__(
        self: Self,
        simulation_path: pathlib.Path,
        real_path: pathlib.Path,
        output_path: pathlib.Path,
    ) -> None:
        """Initialize class."""
        self.sim_path = simulation_path
        self.real_path = real_path
        self.output_path = output_path
        self.fake_datasets = list(self.sim_path.iterdir())
        self.fake_datasets = [
            path
            for path in self.fake_datasets
            if len(list(path.glob("*traces*"))) > 0
        ]
        self.real_datasets = list(self.real_path.iterdir())
        self.real_datasets = [
            path
            for path in self.real_datasets
            if len(list(path.glob("*traces*"))) > 0
        ]
        msg = "Successfully initialized GridAugmenter."
        con.log(msg)

    def run_augmentation(self: Self) -> None:
        """Run the augmentation."""
        msg = "Starting augmentation ..."
        con.log(msg)

        for fake_dataset_path in self.fake_datasets:
            # load the fake dataset
            fake_dataset = load(fake_dataset_path)
            # load a random real dataset
            indices = np.arange(len(self.real_datasets))
            real_dataset = load(self.real_datasets[rng.choice(indices)])

            msg = f"Augmenting {fake_dataset_path.name} with {real_dataset.path.name}"
            con.log(msg)

            # check if the real dataset has detected chirps
            if not real_dataset.com.chirp.are_detected:
                msg = (
                    "No chirps detected in real dataset, skipping "
                    "for simulation augmentation as no chirpless windows "
                    "can be located."
                    "Augmenting with a recording without chirpdetections is "
                    "not implemented yet."
                )
                raise ValueError(msg)

            augmented_dataset = self.augment_grid(fake_dataset, real_dataset)

            # keep iterating until self.augment_grid() finds a good window
            # without chirps
            while augmented_dataset is None:
                real_dataset = load(rng.choice(self.real_datasets))
                augmented_dataset = self.augment_grid(
                    fake_dataset, real_dataset
                )

                msg = "No valid snippet found, trying again..."
                con.log(msg)

                del real_dataset

            save(augmented_dataset, self.output_path)

            del fake_dataset
            del real_dataset
            del augmented_dataset
            gc.collect()

    def augment_grid(self: Self, sd: Dataset, rd: Dataset) -> Dataset:
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
        This function normalizes the simulated dataset, takes a random snippet
        from the real recording, scales the simulated dataset to match the real
        recording, and combines the two datasets. This introduces realistic
        background noise to the simulated dataset. A future version should
        choose a snipped from the real recording that contains as little
        communication as possible.
        """
        # normalize the simulated dataset
        sd.track.powers = (
            sd.track.powers - np.mean(sd.track.powers)
        ) / np.std(sd.track.powers)
        sd.grid.rec = (sd.grid.rec[:] - np.mean(sd.grid.rec[:])) / np.std(
            sd.grid.rec[:]
        )

        # take a random snippet from the real recording
        target_shape = sd.grid.rec.shape
        source_shape = rd.grid.rec.shape

        # check if number of electrodes of the simulated dataset is larger
        # than the number of electrodes of the real dataset
        if target_shape[1] > source_shape[1]:
            msg = (
                "The number of electrodes of the simulated dataset is "
                "larger than the number of electrodes of the real dataset."
            )
            raise GridDataMismatchError(msg)

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
        mean_power, std_power = (
            np.nanmean(rd.track.powers),
            np.nanstd(rd.track.powers),
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


def fakegrid_cli(output_path: pathlib.Path) -> None:
    """Command line interface for the fakegrid function."""
    config_path = output_path / "gridtools_simulations.toml"
    config_path = config_path.resolve()
    config = load_sim_config(config_path)

    chirper = ChirpGenerator(config)
    gs = GridSimulator(config, chirper, output_path, 3)

    gs.run_simulations()


def hybridgrid_cli(
    input_path: pathlib.Path,
    real_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    """Run the hybridgrid simulation using the command line interface."""
    input_path = pathlib.Path(input_path)
    real_path = pathlib.Path(real_path)
    output_path = pathlib.Path(output_path)

    if output_path.exists():
        shutil.rmtree(output_path)

    ga = GridAugmenter(input_path, real_path, output_path)
    ga.run_augmentation()
