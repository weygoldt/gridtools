import numpy as np
from numpy.fft import rfft
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from IPython import embed
from plotstyle import PlotStyle

from ssqueezepy import ssq_cwt, issq_cwt, extract_ridges
from ssqueezepy.toolkit import cos_f, mad_rms
from ssqueezepy.visuals import imshow

ps = PlotStyle()

def power_spectrum(data, time):

    time_step = time[1] - time[0]
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(data.size, time_step)
    idx = np.argsort(freqs)

    return abs(freqs[idx]), abs(ps[idx])


def fix_yaxis(ax, freqs):

    def fmt(freqs):
        return ("%.d" if all(float(h).is_integer() for h in freqs) else
                "%.2f")

    idxs = np.linspace(0, len(freqs) - 1, 8).astype('int32')
    yt = [fmt(freqs) % h for h in np.asarray(freqs)[idxs]]
    ax.set_yticks(idxs)
    ax.set_yticklabels(yt)

def echirp(N):

    t = np.linspace(0, 10, N, False)
    return np.cos(2 * np.pi * np.exp(t / 3)), t

N = 5048
noise_var = 4  # noise variance; compare error against = 12

x, ts = echirp(N)
x *= (1 + .3 * cos_f([1], N))  # amplitude modulation
xo = x.copy()
np.random.seed(4)
x += np.sqrt(noise_var) * np.random.randn(len(x))
pxo, fxo = power_spectrum(xo, ts)
px, fx =  power_spectrum(x, ts)

fig, ax = plt.subplots(2,1)
ax[0].plot(x, label='noisy chirp', c=ps.gblue1)  
ax[0].plot(xo, label='original chirp', c=ps.gblue3)
ax[1].plot(px, fx, label='_', c=ps.gblue1)
ax[1].plot(pxo, fxo, label='_', c=ps.gblue3)
ax[0].set_title('Noisy chirp')
ax[0].set_xlabel('time')
ax[0].set_ylabel('amplitude')
ax[1].set_title('Noisy power-spectrum')
ax[1].set_xlabel('frequency')
ax[1].set_ylabel('power')
fig.legend()
plt.show()

# compute Fourier transform
kw = dict(wavelet=('morlet', {'mu': 4.5}), nv=64, scales='log')
_Tx, _Wx, _ssq_freqs, _cwt_freqs, *_ = ssq_cwt(xo, t=ts, **kw)
Tx, Wx, ssq_freqs, cwt_freqs, *_ = ssq_cwt(x, t=ts, **kw)

# extract ridge by maximum along frequency dimension
ridge = np.argmax(np.abs(_Tx), axis=0)
ridge = savgol_filter(ridge, 100, 6) # smooth ridge

fig, ax = plt.subplots(2,2, constrained_layout=True, figsize=(8,6))

freqs = [_cwt_freqs, _ssq_freqs, cwt_freqs, ssq_freqs]
scalograms = [abs(_Wx), abs(_Tx), abs(Wx), abs(Tx)]
ylabels = ['scale', 'frequency', 'scale', 'frequency']
titles = ['CWT', 'SSQ-CWT', 'noise + CWT', 'noise + SSQ-CWT']
pkw = dict(cmap='inferno', aspect='auto', origin='upper')
ridge_pad = 30

for i, axi in enumerate(ax.reshape(-1)):

    # better color coding for synchrosqueezed transform
    if ylabels[i] == 'frequency':
        prange = np.max(scalograms[i]) - np.min(scalograms[i])
        vmin, vmax = np.min(scalograms[i]), np.max(scalograms[i]) - prange * 0.8
    else:
        vmin, vmax = np.min(scalograms[i]), np.max(scalograms[i])

    axi.imshow(scalograms[i], **pkw, vmin=vmin, vmax=vmax)
    axi.plot(ridge - ridge_pad , lw=1, ls='--', color='white')
    axi.plot(ridge + ridge_pad , lw=1, ls='--', color='white')
    fix_yaxis(axi, freqs[i])
    axi.set_ylabel(ylabels[i])
    axi.set_title(titles[i])
    axi.set_xlabel('time')

plt.show() 

# invert noisy synchrosqueezed transform by ridge
pad = 10
ix = issq_cwt(Tx, kw['wavelet'], cc=ridge, cw=np.ones_like(ridge)*pad)[0]

# compute power spectrum of whole signal
pxo, fxo = power_spectrum(xo, ts)
pxi, fxi = power_spectrum(ix, ts)

fig, ax = plt.subplots(2,1)
ax[0].plot(xo, label='original', c=ps.gblue3)
ax[0].plot(ix, label='recoverd', c=ps.gblue1)
ax[1].plot(pxo, fxo, label='_', c=ps.gblue3)
ax[1].plot(pxi, fxi, label='_', c=ps.gblue1)
ax[0].set_title('Recovered signal')
ax[0].set_xlabel('time')
ax[0].set_ylabel('amplitude')
ax[1].set_title('Recovered power-spectrum')
ax[1].set_xlabel('frequency')
ax[1].set_ylabel('power')
fig.legend()
plt.show()

print("signal   MAD/RMS: %.6f" % mad_rms(xo, ix))
