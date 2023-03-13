import numpy as np
from numpy.fft import rfft
import matplotlib.pyplot as plt
from IPython import embed

from ssqueezepy import ssq_stft, issq_stft, extract_ridges
from ssqueezepy.toolkit import cos_f, mad_rms



def echirp(N):
    t = np.linspace(0, 10, N, False)
    return np.cos(2 * np.pi * np.exp(t / 3)), t

N = 2048
noise_var = 4  # noise variance; compare error against = 12

x, ts = echirp(N)
x *= (1 + .3 * cos_f([1], N))  # amplitude modulation
xo = x.copy()
np.random.seed(4)
x += np.sqrt(noise_var) * np.random.randn(len(x))
axfo = np.abs(rfft(xo))
axf = np.abs(rfft(x))

fig, ax = plt.subplots(2,1)
ax[0].plot(x)  
ax[0].plot(xo)
ax[1].plot(axf)
ax[1].plot(axfo)
plt.show()

# compute Fourier transform
_Tx, _Wx, _ssq_freqs, _stft_freqs, *_ = ssq_stft(xo, t=ts)
Tx, Wx, ssq_freqs, stft_freqs, *_ = ssq_stft(x, t=ts, padtype="wrap")

# extract ridge by maximum along frequency dimension
ridge = np.argmax(np.abs(_Tx), axis=0)

fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
kw = dict(cmap='inferno', aspect='auto', origin='lower')
ax[0,0].imshow(abs(_Wx), extent=(ts[0], ts[-1], _stft_freqs[0], _stft_freqs[-1]), **kw)
ax[0,1].imshow(abs(_Tx), extent=(ts[0], ts[-1], _ssq_freqs[0], _ssq_freqs[-1]), **kw)
ax[1,0].imshow(abs(Wx), extent=(ts[0], ts[-1], stft_freqs[0], stft_freqs[-1]), **kw)
ax[1,1].imshow(abs(Tx), extent=(ts[0], ts[-1], ssq_freqs[0], ssq_freqs[-1]), **kw)

pad = 5 
kw = dict(lw=1, ls='--', color='white', alpha=.5)
ax[0,0].plot(ts, _stft_freqs[ridge]+pad, **kw)
ax[0,0].plot(ts, _stft_freqs[ridge]-pad, **kw)
ax[0,1].plot(ts, _ssq_freqs[ridge]+pad, **kw)
ax[0,1].plot(ts, _ssq_freqs[ridge]-pad, **kw)
ax[1,0].plot(ts, stft_freqs[ridge]+pad, **kw)
ax[1,0].plot(ts, stft_freqs[ridge]-pad, **kw)
ax[1,1].plot(ts, ssq_freqs[ridge]+pad, **kw)
ax[1,1].plot(ts, ssq_freqs[ridge]-pad, **kw)
[axi.set_xlim(ts[0], ts[-1]) for axi in ax.reshape(-1)]
[axi.set_ylim(ssq_freqs[0], ssq_freqs[-1]) for axi in ax.reshape(-1)]

fig.supxlabel('Time')
fig.supylabel('Frequency')
plt.show() 

# invert noisy synchrosqueezed transform by ridge
pad = 2
ix = issq_stft(Tx, cc=ridge, cw=np.ones_like(ridge)*pad)[0]

# compute Fourier transform of whole signal
axo   = np.abs(rfft(xo))
axi = np.abs(rfft(ix))

fig, ax = plt.subplots(2,1)
ax[0].plot(xo)
ax[0].plot(ix/np.max(ix), lw=1, ls='--', color='black')
ax[1].plot(axo/np.max(axo))
ax[1].plot(axi, lw=1, ls='--', color='black')
plt.show()

print("signal   MAD/RMS: %.6f" % mad_rms(xo, ix))
