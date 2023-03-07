
import matplotlib.pyplot as plt
import numpy as np
import fakefish as ff
import thunderfish.powerspectrum as ps

np.random.seed(1)

samplerate = 40000
duration = 120.0
time = np.arange(0, duration, 1/samplerate)
nchirps = 10
chirp_times = np.random.uniform(0, duration, nchirps)
nrises = 2
rise_times = np.random.uniform(0, duration, nrises)

# class fish:

#     def __init__(self, eodf, samplerate, duration, phase0, noise_std):
#         self.eodf = eodf
#         self.samplerate = samplerate
#         self.duration = duration


# generate frequency trace with chirps
chirp_trace, chirp_amp = ff.chirps(eodf=0.0,
                                   samplerate=samplerate,
                                   duration=duration,
                                   chirp_times=chirp_times,
                                   chirp_size=100.0,
                                   chirp_width=0.01,
                                   chirp_kurtosis=1.0,
                                   chirp_contrast=0.2,
                                   )

# generate frequency trace with rises
rise_trace = ff.rises(eodf=0.0,
                      samplerate=samplerate,
                      duration=duration,
                      rise_times=rise_times,
                      rise_size=40.0,
                      rise_tau=1.0,
                      decay_tau=10.0
                      )

# combine traces to one
# full_trace = rise_trace + chirp_trace + 500.0
full_trace = chirp_trace + rise_trace + 500.0

# make the EOD from the frequency trace
fish = ff.wavefish_eods(
    fish='Alepto',
    frequency=full_trace,
    samplerate=samplerate,
    duration=duration,
    phase0=0.0,
    noise_std=0.05
)

fish = fish * chirp_amp

# comptute a spectrogram of the resutling EOD
spec, freqs, spectime = ps.spectrogram(
    data=fish,
    ratetime=samplerate,
    freq_resolution=0.5,
    overlap_frac=0.5,
)

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(time, chirp_trace)
ax[0].plot(time, rise_trace)
ax[1].plot(time, full_trace)
ax[2].plot(time, fish)
ax[3].imshow(X=ps.decibel(spec),
             aspect='auto',
             origin='lower',
             extent=(spectime.min(), spectime.max(), freqs.min(), freqs.max())
             )
ax[3].set_ylim(450, 700)
plt.show()
