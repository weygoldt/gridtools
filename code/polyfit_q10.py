import matplotlib.pyplot as plt
import numpy as np
from gridtools.utils.datahandling import lowpass_filter, removeOutliers

datapath = "/mnt/backups/@data/output/2016-04-20-18_49/"
freqs = np.load(datapath + "frequencies.npy", allow_pickle=True)
idx = np.load(datapath + "indices.npy", allow_pickle=True)
ident = np.load(datapath + "identities.npy", allow_pickle=True)
temp = np.load(datapath + "temperature.npy", allow_pickle=True)
times = np.load(datapath + "times.npy", allow_pickle=True)
ids = np.unique(ident)

for fish in ids:

    f = freqs[ident == fish]
    t = times[idx[ident == fish]]
    tc = temp[idx[ident == fish]]

    f_filt = lowpass_filter(f, 3, 0.0001)
    # make fit line
    tempfit = np.polyfit(f_filt, tc, 1)
    fit = np.poly1d(tempfit)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(t, f)
    ax[0].plot(t, f_filt)
    ax[1].plot(t, tc)
    ax[2].plot(tc, f_filt)
    ax[2].plot(tc, fit(tc))
    plt.show()
