import fakefish as ff
from IPython import embed
import numpy as np
from typing import Tuple
from scipy.signal import savgol_filter
from scipy.stats import norm
from movement import FishMovement

np.random.seed(13)


def grid(origin, shape, spacing, type='hex'):

    assert type in ['hex', 'square'], "type must be 'hex' or 'square'"

    # grid parameters
    electrode_number = shape[0] * shape[1]
    electrode_index = np.arange(0, electrode_number)
    electrode_x = np.mod(electrode_index, shape[0]) * spacing
    electrode_y = np.floor(electrode_index / shape[0]) * spacing

    # shift every second row to make a hexagonal grid
    if type == 'hex':
        electrode_y[1::2] += spacing / 2

    # shift the grid to the specified origin
    electrode_x += origin[0] - np.mean(electrode_x)
    electrode_y += origin[1] - np.mean(electrode_y)

    return electrode_x, electrode_y


class FishSignal:
    def __init__(self, samplerate, duration, eodf, nchirps, nrises):
        time = np.arange(0, duration, 1 / samplerate)
        chirp_times = np.random.uniform(0, duration, nchirps)
        rise_times = np.random.uniform(0, duration, nrises)

        # pick random parameters for chirps 
        chirp_size = np.random.uniform(60, 200, nchirps)
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

        self.signal = signal
        self.trace = full_trace
        self.time = time
        self.samplerate = samplerate
        self.eodf = eodf

    def visualize(self):

        spec, freqs, spectime = ps.spectrogram(
            data=self.signal,
            ratetime=self.samplerate,
            freq_resolution=0.5,
            overlap_frac=0.5,
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[1, 4], sharex=True)

        ax1.plot(self.time, self.signal)
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("Time (s)")
        ax1.set_title("EOD signal")

        ax2.imshow(ps.decibel(spec), origin='lower', aspect="auto", extent=[spectime[0], spectime[-1], freqs[0], freqs[-1]])
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("Spectrogram")
        ax2.set_ylim(0, 2000)
        plt.show()




class Recording:

    def __init__(self, fishcount: int, duration: int, grid_shape: Tuple[int, int], electrode_spacing: int):

        self.samplerate = 20000

        # EOD parameters
        mineodf = 500
        maxeodf = 1000
        maxchirprate = 0.2
        maxriserate = 0.1
        nmaxchirps = int(np.floor(duration*maxchirprate))
        nmaxrises = int(np.floor(duration*maxriserate))

        # save the electrode positions
        n_electrodes = grid_shape[0] * grid_shape[1]
        self.ex, self.ey = grid(origin=[0, 0], shape=grid_shape, spacing=electrode_spacing, type='hex') 
        
        self.traces = []
        self.x = []
        self.y = []

        for i in range(fishcount):

            tmax = duration # duration in seconds
            origin = [0, 0] # starting point
            boundaries = [(-5, 5), (-5, 5)] # boundaries of the arena
            mov = FishMovement(tmax, self.samplerate, origin, boundaries) # movement object

            x, y = mov.x, mov.y # get the position data

            # save the position data
            self.x.append(x)
            self.y.append(y)

            # calculate the distance to every electrode
            distances = np.sqrt((x[:, None] - self.ex[None, :]) ** 2 + (y[:, None] - self.ey[None, :]) ** 2)

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
            sig = FishSignal(self.samplerate, duration, eodf, nchirps, nrises)

            # put the signal on every electrode
            grid_signals = np.tile(sig.signal, (n_electrodes, 1)).T

            # attenuate the signal with the distance
            attenuated_signals = grid_signals * distances

            # normalize by the number of simulated fish
            attenuated_signals = attenuated_signals / fishcount

            # add the signal to the recording
            if i == 0:
                self.signal = attenuated_signals
            else:
                self.signal += attenuated_signals

            self.traces.append(sig.trace)
            self.time = np.arange(0, duration, 1/self.samplerate)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import thunderfish.powerspectrum as ps

    rec = Recording(fishcount=1, duration=600, grid_shape=(8, 8), electrode_spacing=0.5)

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
    
    ax[0,1].scatter(rec.ex, rec.ey, color="red", alpha=1, lw=1, zorder=10)

    # plot positions of all fish
    for i in range(len(rec.x)):
        ax[0,1].plot(rec.x[i], rec.y[i], alpha=1, lw=1)

    ax[1,0].set_ylim(400, 1100)
    plt.savefig('exampleplot.png')
    plt.show()
