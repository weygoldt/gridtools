from simulations import Recording
import matplotlib.pyplot as plt
import numpy as np
from ssqueezepy import ssq_cwt, extract_ridges
from ssqueezepy.visuals import plot, imshow

def viz(x, Tf, ridge_idxs, yticks=None, ssq=False, transform='cwt', show_x=True):
    if show_x:
        plot(x, title="x(t)", show=1,
             xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]")

    ylabel = ("Frequency scales [1/Hz]" if (transform == 'cwt' and not ssq) else
              "Frequencies [Hz]")
    title = "abs({}{}) w/ ridge_idxs".format("SSQ_" if ssq else "",
                                             transform.upper())

    ikw = dict(abs=1, cmap='turbo', yticks=yticks, title=title)
    pkw = dict(linestyle='--', color='k', xlabel="Time [samples]", ylabel=ylabel,
               xlims=(0, Tf.shape[1]))

    imshow(Tf, **ikw, show=0)
    plot(ridge_idxs, **pkw, show=1)


rec = Recording(2, 120, (1,1), 0, 0.001)
signal = rec.signal[:,0]
time = rec.time
del rec

Tx, Wx, ssq_freqs, scales = ssq_cwt(x=signal, t=time, wavelet='morlet', padtype='wrap')
ridge_idxs = extract_ridges(Wx, scales, penalty=2.0, n_ridges=2, bw=25)

imshow(Wx, abs=1, cmap='turbo', yticks=scales, show=1)
plot(ridge_idxs, linestyle='--', color='k', show=1)
