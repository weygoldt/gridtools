import datetime
import datetime as dt
from dataclasses import dataclass

import h5py
import nixio
import numpy as np


@dataclass
class fish:
    ident: int
    eod: [float, ...]
    time: [float, ...]
    start_time: str
    stop_time: str


ident = 1
eod = np.random.rand(500).tolist()
time = np.linspace(0, 100, 500).tolist()
start_time = "2022-11-12 08:00"
stop_time = "2022-11-12 09:00"

f = fish(ident, eod, time, start_time, stop_time)

d = nixio.File.open("testfile.nix", nixio.FileMode.Overwrite)
block = d.create_block("fish", "grid fish")
data = block.create_data_array("eod", "tracked eod", data=f.eod)
data.label = "Frequency"
data.unit = "Hz"
data.append_sampled_dimension(0.3, label="time", unit="s")

d.close()
