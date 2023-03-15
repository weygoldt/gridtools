import fakefish as ff
from IPython import embed
import numpy as np
from typing import Tuple
from scipy.signal import savgol_filter
from scipy.stats import norm
from movement import make_positions

np.random.seed(13)

def make_fish(
    samplerate: float, duration: float, eodf: float, nchirps: int, nrises: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    time = np.arange(0, duration, 1 / samplerate)
    chirp_times = np.random.uniform(0, duration, nchirps)
    rise_times = np.random.uniform(0, duration, nrises)

    # pick random parameters for chirps 
    chirp_size = np.random.uniform(60, 300, nchirps)
    chirp_width = np.random.uniform(0.01, 0.1, nchirps)
    chirp_kurtosis = np.random.uniform(1, 1, nchirps)
    chirp_contrast = np.random.uniform(0.1, 0.5, nchirps)

    # pick random parameters for rises
    rise_size = np.random.uniform(10, 100, nrises)
    rise_tau = np.random.uniform(0.5, 1.5, nrises)
    decay_tau = np.random.uniform(5, 15, nrises)

    # generate frequency trace with chirps
    chirp_trace, chirp_amp = ff.chirps(
        eodf=0.0,
        samplerate=samplerate,
        duration=duration,
        chirp_times=chirp_times,
        chirp_size=chirp_size,
        chirp_width=chirp_width,
        chirp_kurtosis=chirp_kurtosis,
        chirp_contrast=chirp_contrast,
    )

    # generate frequency trace with rises
    rise_trace = ff.rises(
        eodf=0.0,
        samplerate=samplerate,
        duration=duration,
        rise_times=rise_times,
        rise_size=rise_size,
        rise_tau=rise_tau,
        decay_tau=decay_tau,
    )

    # combine traces to one
    full_trace = chirp_trace + rise_trace + eodf

    # make the EOD from the frequency trace
    fish = ff.wavefish_eods(
        fish="Alepto",
        frequency=full_trace,
        samplerate=samplerate,
        duration=duration,
        phase0=0.0,
        noise_std=0.05,
    )

    signal = fish * chirp_amp

    return full_trace, time, signal


class Recording:

    def __init__(self, fishcount: int, duration: int, grid: Tuple[int, int], electrode_spacing: int, step_size: float):

        self.samplerate = 20000

        # EOD parameters
        mineodf = 500
        maxeodf = 1000
        maxchirprate = 0.2
        maxriserate = 0.1
        nmaxchirps = int(np.floor(duration*maxchirprate))
        nmaxrises = int(np.floor(duration*maxriserate))

        # grid parameters
        electrode_number = grid[0] * grid[1]
        electrode_index = np.arange(0, electrode_number)
        electrode_x = np.mod(electrode_index, grid[0]) * electrode_spacing
        electrode_y = np.floor(electrode_index / grid[0]) * electrode_spacing
        center_x = np.mean(electrode_x)
        center_y = np.mean(electrode_y)
        
        # save the electrode positions
        self.electrode_x = electrode_x
        self.electrode_y = electrode_y
        
        # shift every second row to mage a hexagonal grid
        electrode_y[1::2] += electrode_spacing / 2
        
        self.traces = []
        self.x = []
        self.y = []

        for i in range(fishcount):

            fs = 30  # sampling frequency in Hz
            tmax = duration # duration in seconds
            time = np.arange(0, tmax, 1/fs) # time vector
            peak_veloc = 0.2 # most common velocity in m/s
            directions = np.arange(0, 2*np.pi, 0.001) # directions vector in radians
            origin = [0, 0] # starting point
            boundaries = [(0, 10), (0, 10)] # boundaries of the arena

            # make probablitiy distribution of directions
            sigma = 7/fs
            p1 = norm.pdf(directions, 0, sigma) 
            p2 = norm.pdf(directions, np.max(directions), sigma)
            probabilities = (p1 + p2)
            probabilities = probabilities / np.sum(probabilities)

            # make random step lengths according to a gamma distribution
            step_lengths = np.random.default_rng().gamma(peak_veloc*100, 1, tmax*fs)/100
            
            # remove outliers
            step_lengths[step_lengths > 1] = 1

            # normalize to sampling rate
            step_lengths = step_lengths / fs
            
            # draw random directions according to the probability distribution
            trajectories = np.random.choice(directions, size = (tmax*fs), p = probabilities)

            # make positions
            x, y = make_positions(origin, boundaries, trajectories, step_lengths)

            # resample positions to 20000 Hz
            xi = np.interp(np.arange(0, tmax, 1/20000), time, x)
            yi = np.interp(np.arange(0, tmax, 1/20000), time, y)

            # save the position data
            self.x.append(x)
            self.y.append(y)

            # calculate the distance to every electrode
            distances = np.sqrt((xi[:, None] - electrode_x[None, :]) ** 2 + (yi[:, None] - electrode_y[None, :]) ** 2)

            # make the distance squared and invert it
            distances = distances ** 2
            distances = np.max(distances) - distances            

            # generate random EOD parameters
            if nmaxchirps > 0:
                nchirps = np.random.randint(0, nmaxchirps)
            else:
                nchirps = 0
            if nmaxrises > 0:
                nrises = np.random.randint(0, nmaxrises)
            else:
                nrises = 0
            eodf = np.random.uniform(mineodf, maxeodf)

            # make the EOD
            trace, time, signal = make_fish(self.samplerate, duration, eodf, nchirps, nrises)

            # put the signal on every electrode
            grid_signals = np.tile(signal, (electrode_number, 1)).T

            # attenuate the signal with the distance
            attenuated_signals = grid_signals * distances

            # normalize by the number of simulated fish
            attenuated_signals = attenuated_signals / fishcount

            # add the signal to the recording
            if i == 0:
                self.signal = attenuated_signals
            else:
                self.signal += attenuated_signals

            self.traces.append(trace + eodf)
            self.time = time


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import thunderfish.powerspectrum as ps
    
    rec = Recording(fishcount=2, duration=60, grid=(1, 1), electrode_spacing=0.5, step_size=0.001)

    # set electrode and fish of interest
    eoi = 0
    foi = 0
    
    # compute powerspectrum for each electrode and sum them up 
    for i in range(rec.signal.shape[1]):
        spec, freqs, spectime = ps.spectrogram(
            data=rec.signal[:, i],
            ratetime=rec.samplerate,
            freq_resolution=0.5,
            overlap_frac=0.5,
        )
        if i == 0:
            summed_spec = spec
        else:
            summed_spec += spec

    fig, ax = plt.subplots(2, 2, sharex='col')

    ax[0,0].plot(rec.time, rec.signal[:, eoi], color="black", alpha=1, lw=1)
    ax[1,0].imshow(
        X=ps.decibel(summed_spec),
        aspect="auto",
        origin="lower",
        extent=(spectime.min(), spectime.max(), freqs.min(), freqs.max()),
    )

    for trace in rec.traces:
        ax[1,0].plot(rec.time, trace, color="black", alpha=1, lw=1)
    
    ax[0,1].scatter(rec.electrode_x, rec.electrode_y, color="red", alpha=1, lw=1, zorder=10)

    # plot positions of all fish
    for i in range(len(rec.x)):
        ax[0,1].plot(rec.x[i], rec.y[i], alpha=1, lw=1)

    ax[1,0].set_ylim(400, 1100)
    plt.savefig('exampleplot.png')
    plt.show()
