import random

import nixio as nio
import numpy as np
from scipy.signal import savgol_filter


def make_eod():
    # build baseline
    length = 1000
    eodf = random.randint(500, length)  # make eodf
    base = np.zeros(1000) + eodf  # make baseline eodf
    noise = np.random.normal(0, 1, length)  # make some noise
    baseline = noise + base

    # add n rises at random time points
    rise_number = 10
    rise_height = 100  # in hz
    rise_lenght = np.arange(40)  # time from peak back to baseline
    rise_positions = [
        random.randint(10, length - 10 - len(rise_lenght))
        for i in range(rise_number)
    ]
    rise = rise_height * np.exp(-rise_lenght / 5)

    # add rises
    for i in range(rise_number):
        baseline[
            rise_positions[i] : rise_positions[i] + len(rise_lenght)
        ] += rise

    return baseline


def make_position():
    # make random numbers
    xpos = savgol_filter(np.random.randint(0, 350, 1000), 60, 6)
    ypos = savgol_filter(np.random.randint(0, 350, 1000), 60, 6)

    return xpos, ypos


class Fish:
    def __init__(self, lenght: int = 1000):
        self.eod = make_eod()
        self.xpos, self.ypos = make_position()


class Grid:
    def __init__(self) -> None:
        self.identities = []
        self.frequencies = []
        self.xpositions = []
        self.ypositions = []

        num_fish = 10

        for i in range(num_fish):
            self.identities.extend(np.ones(1000) * i)
            eod = make_eod()
            self.frequencies.extend(eod.tolist())
            x, y = make_position()
            self.xpositions.extend(x.tolist())
            self.ypositions.extend(y.tolist())

        self.ids = np.unique(self.identities)
        self.identities = np.asarray(self.identities)
        self.frequencies = np.asarray(self.frequencies)
        self.xpositions = np.asarray(self.xpositions)
        self.ypositions = np.asarray(self.ypositions)


grid1 = Grid()
grid2 = Grid()

# save the grids
file = nio.File.open("testgrid.nix", nio.FileMode.Overwrite)


# make first recordig instance
def grid2block(file, grid1, number):
    block = file.create_block(f"{number}", f"grid recording {number}")
    block.create_data_array("eod", "tracked eod", data=grid1.frequencies)
    block.create_data_array("xpos", "x coordinates", data=grid1.xpositions)
    block.create_data_array("ypos", "y coordinates", data=grid1.ypositions)
    block.create_data_array(
        "ident", "identities of data arrrays", data=grid1.identities
    )


grid2block(file, grid1, 1)
grid2block(file, grid2, 2)
file.close()
