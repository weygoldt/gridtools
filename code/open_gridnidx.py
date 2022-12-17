import gridtools as gt
import matplotlib.pyplot as plt
import nixio as nio
import numpy as np


class NixGridRecording:
    def __init__(
        self, recording_number: int, filepath: str, filemode: str = "ReadOnly"
    ) -> None:

        # check if filemode is usable
        assert filemode in [
            "ReadOnly",
            "ReadWrite",
        ], f"Filemode can be ReadOnly or ReadWrite, you supplied {filemode}!"

        # set file mode
        if filemode == "ReadOnly":
            filemode = nio.FileMode.ReadOnly
        else:
            filemode = nio.FileMode.ReadWrite

        # open file
        file = nio.File.open(filepath, filemode)

        # select recording
        recording = file.blocks[recording_number]

        # read data arrays
        self.times = recording.data_arrays["times"][:]
        self.frequencies = recording.data_arrays["frequencies"][:]
        self.identities = recording.data_arrays["identities"][:]
        self.indices = recording.data_arrays["indices"][:]
        self.xpositions = recording.data_arrays["xpositions"][:]
        self.ypositions = recording.data_arrays["ypositions"][:]
        self.temperature = recording.data_arrays["temperature"][:]
        self.light = recording.data_arrays["light"][:]
        self.ids = recording.data_arrays["ids"][:]


grid = NixGridRecording(0, "/mnt/backups/@data/output/2016_colombia.nix", "ReadOnly")


for track_id in grid.ids:
    plt.plot(
        grid.xpositions[grid.identities == track_id],
        grid.ypositions[grid.identities == track_id],
    )

plt.show()
