"""Classes and functios for generating electric organ discharges (EODs)."""

from typing import Union, List, Tuple, Optional, Any, TypeVar, Generic, Self, Callable
import numpy as np
import matplotlib.pyplot as plt
from gridtools.utils.configfiles import SimulationConfig
from gridtools.simulations.utils import get_random_timestamps
from gridtools.utils.logger import Timer
from gridtools.simulations.communication import monophasic_chirp, biphasic_chirp
from scipy.signal.windows import tukey
from gridtools.simulations.utils import get_width_heigth

rng = np.random.default_rng()


class ChirpGenerator:
    """Class for generating chirps with different methods."""

    def __init__(self: Self, config: SimulationConfig) -> None:
        """Initialize the chirp generator."""
        self.config = config
        self.data = np.load(self.config.chirps.chirp_data_path)
        rng.shuffle(self.data)
        self.nchirps = None

    @property
    def method(self: Self) -> str:
        """Return the method to use for chirp generation."""
        return self.config.chirps.method

    @property
    def chirp_model(self: Self) -> Optional[Callable]:
        """Return the model to use for chirp generation."""
        if self.method == "extracted":
            msg = "Extracted chirps do not have a model."
            raise AttributeError(msg)
        if self.method == "simulated":
            if self.config.chirps.model == "monophasic":
                return monophasic_chirp
            if self.config.chirps.model == "biphasic":
                return biphasic_chirp

            msg = (
                f"Chirp model {self.config.chirps.model} not recognized."
                "Please choose between 'monophasic' and 'biphasic'."
            )
            raise ValueError(msg)

        msg = (
            f"Chirp generation method {self.method} not recognized."
            "Please choose between 'extracted' and 'simulated'."
        )
        raise ValueError(msg)

    def _decimate_chirp_data(self: Self, nchirps: int) -> np.ndarray:
        """Decimate the chirp data to the desired number of chirps."""
        print(f"Decimating {nchirps} samples from chirp data.")
        return self.data[nchirps:]

    # def _chirps_from_simulation(self: Self) -> np.ndarray:
    #     """Generate chirps from interpolated fits."""
    #     pass
    # TODO: Implement based on commented out block in GridSimulator

    def _chirps_from_extractions(
        self: Self,
        ctimes: np.ndarray,
        contrasts: np.ndarray,
        ftrace: np.ndarray,
        amtrace: np.ndarray,
        time: np.ndarray,
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """Generate chirps from extracted data."""
        center = len(self.data[0, :]) // 2
        chirp_time = np.arange(-center, center) / self.config.grid.samplerate
        chirp_params = []

        rng.shuffle(self.data)
        indices = np.arange(len(ctimes))
        print(f"Making {len(ctimes)} chirps from extracted data.")
        for i, ctime, contrast  in zip(indices, ctimes, contrasts):
            chirp = self.data[i, :]

            if self.config.chirps.replace is False:
                # delete chirp from data
                self.data = np.delete(self.data, i, axis=0)

            mx, w, _ = get_width_heigth(chirp_time, chirp)
            mn = np.min(chirp)
            chirp_params.append((w, mx, mn, contrast))

            # taper ends with a tukey window
            window = tukey(len(chirp), alpha=0.3)
            chirp *= window

            # add chirp to frequency trace at ctime
            tidx = np.argmin(np.abs(time - ctime))

            # check if chirp fits on the trace or if it needs to be cut
            # this happens when the chirp is too close to the beginning or end
            # of the trace
            if tidx - center < 0:
                chirp = chirp[center - tidx :]
                center = len(chirp) // 2
            if tidx + center > len(ftrace):
                chirp = chirp[: -(tidx + center - len(ftrace))]
                center = len(chirp) // 2
            ftrace[tidx - center : tidx + center] += chirp

            # turn chirp into amp trough with specified contrast
            amtrough = -(chirp / np.max(chirp)) * contrast
            amtrace[tidx - center : tidx + center] += amtrough

        if self.config.chirps.replace is False:
            self.data = self._decimate_chirp_data(len(ctimes))

        return chirp_params, ftrace, amtrace

    def __call__(self: Self) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray]:
        """Generate an EODf trace with chirps."""
        self.nchirps = rng.integers(
            1, int(self.config.grid.duration * self.config.chirps.max_chirp_freq)
        )

        ftrace = np.zeros(
            int(self.config.grid.duration * self.config.grid.samplerate)
        )
        amtrace = np.ones(
            int(self.config.grid.duration * self.config.grid.samplerate)
        )
        time = np.arange(len(ftrace)) / self.config.grid.samplerate

        ctimes = get_random_timestamps(
            start_t = 0,
            stop_t = self.config.grid.duration,
            n_timestamps = self.nchirps,
            min_dt = self.config.chirps.min_chirp_dt,
        )

        contrasts = rng.uniform(
            0.3, self.config.chirps.max_chirp_contrast, size=self.nchirps
        )

        if self.method == "extracted":
            params, ftrace, amtrace = self._chirps_from_extractions(
                ctimes, contrasts, ftrace, amtrace, time
            )
            return ctimes, params, ftrace, amtrace
        # if self.method == "simulated":
        #     return self._chirps_from_simulation(
        #         ctimes, contrasts, ftrace, amtrace, time
        #     )
        msg = (
            f"Chirp generation method {self.method} not recognized."
            "Please choose between 'extracted' and 'simulated'."
        )
        raise ValueError(msg)











