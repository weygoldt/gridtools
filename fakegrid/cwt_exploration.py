import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt
from ssqueezepy.visuals import plot, imshow
from simulations import Recording

np.random.seed(2)

def make_signal(frange, npairs, pairdiff):
    fs = 20000
    duration = 5
    t = np.arange(0, duration, 1/fs)
    for i in range(npairs):
        f1 = np.random.randint(frange[0], frange[1])
        f2 = f1 + pairdiff
        x1 = np.sin(2*np.pi*f1*t) + np.random.normal(0, 0.1, len(t))
        x2 = np.sin(2*np.pi*f2*t) + np.random.normal(0, 0.1, len(t))
        if i == 0:
            x = x1 + x2
        else:
            x = x + x1 + x2
    return x, t


def Scalogram(x,t, kw):
	Tx, Wx, scales, *_ = ssq_cwt(x, **kw)
	imshow(Wx, yticks=scales, abs=1, title="abs(CWT) | Morlet wavelet", ylabel="scales", xlabel="samples")
	imshow(Tx, yticks=scales, abs=1, title="abs(SSQ-CWT) | Morlet wavelet", ylabel="scales", xlabel="samples")


# x, t = make_signal([500, 1000], 3, 20)

rec = Recording(fishcount=3, duration=20, grid=(1, 1), electrode_spacing=0.5, step_size=0.001)
x = rec.signal[:, 0]
t = rec.time

# Plot signal in time domain
fig, ax = plt.subplots()
ax.plot(t, x)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Signal in time domain')
plt.show()

kw = dict(wavelet=('morlet', {'mu': 323}), nv= 86, scales='log-piecewise')
kw = dict(wavelet=('morlet'))
Scalogram(x,t, kw)

