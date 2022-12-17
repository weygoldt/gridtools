import datetime

import nixio as nio


class NixGridRecording:
    """
    Loads a single recording from a .nix file containing multiple recordings.
    """

    def __init__(self, block: nio.Block) -> None:

        # read data arrays
        dt_format = "%Y-%m-%d %H:%M:%S"
        self.starttime = datetime.datetime.strptime(block.name, dt_format)
        self.times = block.data_arrays["times"][:]
        self.frequencies = block.data_arrays["frequencies"][:]
        self.identities = block.data_arrays["identities"][:]
        self.indices = block.data_arrays["indices"][:]
        self.xpositions = block.data_arrays["xpositions"][:]
        self.ypositions = block.data_arrays["ypositions"][:]
        self.temperature = block.data_arrays["temperature"][:]
        self.light = block.data_arrays["light"][:]
        self.ids = block.data_arrays["ids"][:]

    def __repr__(self) -> str:
        return "NixGridRecording({})".format(
            self.starttime.strftime("%Y-%m-%d %H:%M:%S")
        )

    def __str__(self) -> str:
        return "Single recording at {}".format(
            self.starttime.strftime("%Y-%m-%d %H:%M:%S")
        )


class NixGrid:
    """
    Loads data arrays of all recordings from a .nix grid recording file. Utilized
    to create pointers between recordings that group frequency tracks into the same fish.
    """

    def __init__(self, filepath: str, filemode: str = "ReadOnly") -> None:

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
        self._filepath = filepath
        file = nio.File.open(filepath, filemode)

        # load all recordings in grid
        self.recordings = []
        for block in file.blocks:
            self.recordings.append(NixGridRecording(block))

    def __repr__(self) -> str:
        return "NixGrid({})".format(self._filepath)

    def __str__(self) -> str:
        return "Grid recording set at {}".format(self._filepath)
