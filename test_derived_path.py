from gridtools.datasets.loaders import load
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

datapath = Path("/mnt/data2/2024_tube_competition/raw/01_06")
data = load(datapath, search_intermediate=True)

data.track.pprint()

for fish_id in data.track.ids[~np.isnan(data.track.ids)]:
    f = data.track.freqs[data.track.idents == fish_id]
    t = data.track.times[data.track.indices[data.track.idents == fish_id]]
    print(np.min(t))

    plt.plot(t, f)
plt.show()
