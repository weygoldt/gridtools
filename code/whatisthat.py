import matplotlib.pyplot as plt
import numpy as np
from gridtools.utils.datahandling import lowpass_filter, removeOutliers

datapath = "/mnt/backups/@data/output/2016-04-20-18_49/"
freqs = np.load(datapath + "fill_times.npy", allow_pickle=True)