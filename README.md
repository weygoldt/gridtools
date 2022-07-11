# GridTools

**Disclaimer**: The build file does not include all dependencies, docstrings are missing and some functions might be buggy! 

`gridtools` provides easy access to tracked frequencies of electrode recordings as provided by the [wavetracker](https://github.com/tillraab/wavetracker.git).
Functions include preprocessing, position estimation and ID extraction.

The main classes provided by `gridtools` is `gridtools.GridTracks`. It loads all tracked fundamental frequenies of a single tracked recording and supports common operations such as:

- `gridtools.GridTracks.fill_powers`: Recompute missing powers after manual frequency tracking using the `EODsorter.py` gui from the [wavetracker](https://github.com/tillraab/wavetracker.git). 
- `gridtools.GridTracks.remove_nans`: Remove all NANs from the dataset.
- `gridtools.GridTracks.remove_short`: Remove short tracks from the dataset.
- `gridtools.GridTracks.remove_poor`: Remove poorly tracked tracks from the dataset.
- `gridtools.GridTracks.load_logger`: Load temperature and light data from a hobologger file.
- `gridtools.GridTracks.q10_norm`: Normalize frequency tracks to a reference temperature based on the Q10 value.
- `gridtools.GridTracks.sex_ids`: Estimate sex of fish based on a frequency threshold.
- `gridtools.GridTracks.positions`: Estimate fish positions based on the weighted powers of $n$ electrodes.
- `gridtools.GridTracks.interpolate`: Interpolate frequency, power and position estimates independently.
- `gridtools.GridTracks.smooth_positions`: Smooth positions using a velocity threshold, median filter and Savitzky Golay filter.

... and a few more. The provided module `datacleaner.py` uses this functionality in the correct order with parameters supplied in a configuration file to iterate over all recordings and returns datasets ready for analysis.

The other classes build on `GridTracks` instances:

- `gridtools.Monad`: Extracts a single individual and associated data.
- `gridtools.Dyad`: Extracts two individuals and associated data. Computes frequency and spatial distances.
- `gridtools.Collective`: Extracts $n$ individuals as `Monad`  instances if they overlap in time.

A submodule callable by `gridtools.utils` includes additional analysis functions, such as:

- `gridtools.utils.kde1d`: Kernel density estimation of a 1-D array.
- `gridtools.utils.kde2d`: Kernel density estimation of x and y positions.
- `gridtools.utils.gaussianhist2d`: Gaussian-smoothed 2-D histogram of x and y coordinates.
- `gridtools.utils.velocity2d`: Computes fish velocity at each coordinate.

... and some more.

## Demonstration
```python
import gridtools as gt
import matplotlib.pyplot as plt
import numpy as np

datapath = "../output/2016-04-09-22_25/"
grid = gt.GridTracks(datapath)

# look at all frequency tracks on a spectrogram
fig, ax = plt.subplots()
grid.plot_spec(ax)
plt.show()

# look at all position estimates
fig, ax = plt.subplots()
grid.plot_pos(ax)
plt.show()

# extract data of a single interesting individual from the previous plot
fish = gt.Monad(grid, 26787)

# plot position and frequency of individual
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
iois = np.arange(30000, 32000)  # indices of interest
ax[0].plot(fish.times[iois], fish.fund[iois])
ax[1].plot(fish.xpos_smth[iois], fish.ypos_smth[iois])
ax[1].set_xlim(0, 350)
ax[1].set_ylim(0, 350)
plt.show()

# extract data of two individuals
ids = [26789, 26788]
dyad = gt.Dyad(grid, ids)

# extract some interesting time points
start = gt.utils.find_closest(dyad.times, 26500)
stop = gt.utils.find_closest(dyad.times, 27500)
iois = np.arange(start, stop)

# plot fundamental frequencies and smoothed positions
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(dyad.times[iois], dyad.fund_id1[iois])
ax[0].plot(dyad.times[iois], dyad.fund_id2[iois])
ax[1].plot(dyad.xpos_smth_id1[iois], dyad.ypos_smth_id1[iois])
ax[1].plot(dyad.xpos_smth_id2[iois], dyad.ypos_smth_id2[iois])
plt.show()

# extract n interacting individuals
ids = ids = [26789, 26788, 26368]
col = gt.Collective(grid, ids)

# extract some interesting time points
start = gt.utils.find_closest(col.fish[0].times, 26500)
stop = gt.utils.find_closest(col.fish[0].times, 27500)
iois = np.arange(start, stop)

# plot
fig, ax = plt.subplots()
col.plot_pos(ax)
plt.show()

# look at some specific point in time
fig, ax = plt.subplots()
for fish in col.fish:
    ax.plot(fish.xpos_smth[iois], fish.ypos_smth[iois])
plt.show()
```