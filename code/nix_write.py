import datetime
import datetime as dt
from dataclasses import dataclass

import h5py
import nixio
import numpy as np


@dataclass
class fish:
    id: int
    eod: list[float, ...]
    pos: list[tuple[float, float], ...]
    start_time: str
    stop_time: str

    def __repr__(self) -> str:
        return f"Fish {ident} at {start_time}"


ident = 1
eod = np.random.rand(500).tolist()
pos = [np.random.rand(500).tolist(), np.random.rand(500).tolist()]

start_time = "2022-11-12 08:00"
stop_time = "2022-11-12 09:00"

f = fish(ident, eod, pos, start_time, stop_time)
print(f)

# open new nix file
d = nixio.File.open("testfile.nix", nixio.FileMode.Overwrite, compression=nixio.Compression.DeflateNormal)

# create a block
block = d.create_block("fish", "grid fish")

# create a data array in that block
data = block.create_data_array("eod", "tracked eod", data=f.eod)

# label data array in block
data.label = "Frequency"

# add a unit to the data array in the block
data.unit = "Hz"
data.append_sampled_dimension(0.3, label="time", unit="s")

# add more info
data = block.create_data_array("pos", "x, y position estimates", data=f.pos)
data.label = "Position"
data.unit = "cm"
data.append_sampled_dimension(0.3, label="time", unit="s")



d.close()
