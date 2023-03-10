import numpy as np
from IPython import embed
from fakefish import rises, wavefish_eods
import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
"""Shows methods to use for CWT scales selection; also see their docstrings."""
if __name__ != '__main__':
    raise Exception("ran example file as non-main")

import numpy as np
from ssqueezepy import ssq_cwt, Wavelet
from ssqueezepy.visuals import imshow, plot
from ssqueezepy.utils import cwt_scalebounds, make_scales, p2up
from ssqueezepy.utils import logscale_transition_idx

#%%# Helper visual method ####################################################
def viz(wavelet, scales, scaletype, show_last, nv):
    plot(scales, show=1, title="scales | scaletype=%s, nv=%s" % (scaletype, nv))
    if scaletype == 'log-piecewise':
        extra = ", logscale_transition_idx=%s" % logscale_transition_idx(scales)
    else:
        extra = ""
    print("n_scales={}, max(scales)={:.1f}{}".format(
        len(scales), scales.max(), extra))

    psih = wavelet(scale=scales)
    last_psihs = psih[-show_last:]

    # find xmax of plot
    least_large = last_psihs[0]
    mx_idx = np.argmax(least_large)
    last_nonzero_idx = np.where(least_large[mx_idx:] < least_large.max()*.1)[0][0]
    last_nonzero_idx += mx_idx + 2

    plot(last_psihs.T[:last_nonzero_idx], color='tab:blue', show=1,
         title="Last %s largest scales" % show_last)

#%%# EDIT HERE ###############################################################
# signal length
# your signal here
samplerate = 20000
duration = 10
rise_time = [2]
rise_size = 100.0
rise_tau = 1.0
decay_tau = 3.0

rise_trace = rises(
        600.0, 
        samplerate, 
        duration, 
        rise_time, 
        rise_size, 
        rise_tau, 
        decay_tau
        )

time = np.arange(0, duration, 1/samplerate)

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

x = eod1 + eod2

N = len(time)

# choose wavelet
wavelet = 'morlet'
# choose padding scheme for CWT (doesn't affect scales selection)
padtype = 'reflect'

# one of: 'log', 'log-piecewise', 'linear'
# 'log-piecewise' lowers low-frequency redundancy; see
# https://github.com/OverLordGoldDragon/ssqueezepy/issues/29#issuecomment-778526900
scaletype = 'log-piecewise'
# one of: 'minimal', 'maximal', 'naive' (not recommended)
preset = 'minimal'
# number of voices (wavelets per octave); more = more scales
nv = 64
# downsampling factor for higher scales (used only if `scaletype='log-piecewise'`)
downsample = 4
# show this many of lowest-frequency wavelets
show_last = 20

#%%## Make scales ############################################################
# `cwt` uses `p2up`'d N internally
M = p2up(N)[0]
wavelet = Wavelet(wavelet, N=M)

min_scale, max_scale = cwt_scalebounds(wavelet, N=len(x), preset=preset)
scales = make_scales(N, min_scale, max_scale, nv=nv, scaletype=scaletype,
                     wavelet=wavelet, downsample=downsample)

#%%# Visualize scales ########################################################
viz(wavelet, scales, scaletype, show_last, nv)
wavelet.viz('filterbank', scales=scales)

#%%# Show applied ############################################################
Tx, Wx, ssq_freqs, scales, *_ = ssq_cwt(x, wavelet, scales=scales,
                                        padtype=padtype)
imshow(Wx, abs=1, title="abs(CWT)")
imshow(Tx, abs=1, title="abs(SSQ_CWT)")

