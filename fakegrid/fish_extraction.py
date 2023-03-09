import numpy as np
from IPython import embed
from fakefish import rises, wavefish_eods
import matplotlib.pyplot as plt

from ssqueezepy import ssq_cwt, issq_cwt
from ssqueezepy.visuals import imshow


def make_even(x: np.ndarray):
    """Make every value in array even"""
    return np.round(x/2)*2

def trace2index(trace, band, cwt_freqs):
    center = np.asarray([np.argmin(np.abs(cwt_freqs - x)) for x in trace])
    upper_bound = np.asarray([np.argmin(np.abs(cwt_freqs - (x + band/2))) for x in trace])
    lower_bound = np.asarray([np.argmin(np.abs(cwt_freqs - (x - band/2))) for x in trace])
    band = make_even(upper_bound - lower_bound).astype(int) 
    return center, band


samplerate = 10000
duration = 30
rise_time = [5]
rise_size = 100.0
rise_tau = 1.0
decay_tau = 8.0

rise_trace = rises(
        600.0, 
        samplerate, 
        duration, 
        rise_time, 
        rise_size, 
        rise_tau, 
        decay_tau
        )

eod1 = wavefish_eods(
        fish='Alepto', 
        frequency=rise_trace, 
        samplerate=samplerate,
        duration=duration, 
        phase0=0.0,
        noise_std=0.05
        )

eodf2 = 620.0 
eod2 = wavefish_eods(
        fish='Alepto',
        frequency=eodf2,
        samplerate=samplerate,
        duration=duration,
        phase0=0.0,
        noise_std=0.05
        )

signal = eod1 + eod2
time = np.arange(0, duration, 1/samplerate)

# compute synchrosqueezed continuous wavelet transform
kw = dict(wavelet=('morlet', {'mu': 230}), nv=150 )
Tx, Wx, ssq_freqs, *_ = ssq_cwt(x=signal, t=time, **kw)

# create frequency boundary to invert the synchrosqueezed transform
band = np.ones_like(ssq_freqs) * 100.0
center_freqs, band_freqs = trace2index(rise_trace, band, ssq_freqs)

embed()

# invert synchrosqueezed transform
xrec = issq_cwt(Tx, kw['wavelet'], center_freqs, band_freqs)[0]

fig, ax = plt.subplots(5,1, figsize=(10,5), constrained_layout=True)

# constrain plotted region to frequency range of interest
freq_range = (np.min(rise_trace)-10, np.max(rise_trace)+10)
index = np.arange(len(ssq_freqs))[np.logical_and(ssq_freqs>freq_range[0], ssq_freqs<freq_range[1])]
freq_range = (ssq_freqs[index[-1]], ssq_freqs[index[0]])
time_range = (time[0], time[-1])

ax[0].plot(time, rise_trace)
ax[0].axhline(eodf2)
ax[1].plot(time, signal)
ax[2].imshow(np.abs(Wx), origin='upper', aspect='auto', cmap='viridis')
ax[3].imshow(np.abs(Tx), origin='upper', aspect='auto', cmap='viridis')
ax[3].plot(center_freqs-band_freqs/2, c='white', lw=1, ls='--')
ax[3].plot(center_freqs+band_freqs/2, c='white', lw=1, ls='--')
ax[4].plot(time, xrec)

ax[2].set_ylim(index[-1], index[0])
ax[3].set_ylim(index[-1], index[0])

ax[0].set_title('Both frequency traces of two fish')
ax[1].set_title('Resulting simulated signal')
ax[2].set_title('Continous wavelet transform (CWT)')
ax[3].set_title('Synchrosqueezed CWT (SSQ-CWT)')
ax[4].set_title('Inverted SSQ-CWT')

ax[0].set_xlim(time_range)
ax[1].set_xlim(time_range)

ax[-1].set_xlabel('Time (s)')

plt.show()

