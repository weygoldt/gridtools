import matplotlib.pyplot as plt
import nixio as nio
import numpy as np
from scipy.ndimage import minimum_filter1d

from gridtools.utils.datahandling import lowpass_filter

datapath = "/mnt/backups/@data/output/2016_colombia.nix"
file = nio.File.open(datapath, nio.FileMode.ReadOnly)
rec = file.blocks[0]

freqs = rec.data_arrays["frequencies"][:]
idx = rec.data_arrays["indices"][:]
ident = rec.data_arrays["identities"][:]
temp = rec.data_arrays["temperature"][:]
times = rec.data_arrays["times"][:]
ids = rec.data_arrays["ids"][:]


for fish in ids:
    # get frequency, time and temp
    f = freqs[ident == fish]
    t = times[idx[ident == fish]]
    tc = temp[idx[ident == fish]]

    # filter signal to remove all signals
    f_medfilt = minimum_filter1d(f, 801)
    f_lpfilt = lowpass_filter(f_medfilt, 3, 0.0005, 2)

    # get max and min temperature of fish
    indices = np.arange(len(tc))
    mint = indices[tc == tc.min()]
    maxt = indices[tc == tc.max()]

    # compute q10
    def temperature_coefficient(tmin, tmax, fmin, fmax):
        return (fmax / fmin) ** (10 / (tmax - tmin))

    q10 = temperature_coefficient(tc[mint], tc[maxt], f[mint], f[maxt])

    print(q10)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, f)
    ax[0].plot(t, f_medfilt)
    ax[0].plot(t, f_lpfilt)
    ax[1].plot(t, tc)
    plt.show()
