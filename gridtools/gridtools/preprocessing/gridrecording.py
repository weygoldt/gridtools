import datetime
from dataclasses import dataclass

import nixio as nio
import numpy as np

from ..logger import makeLogger

logger = makeLogger(__name__)

@dataclass
class Fish:
    id: int
    eod: np.ndarray
    x: np.ndarray 
    y: np.ndarray
    starttime: datetime.datetime
    stoptime: datetime.datetime

    def __repr__(self) -> str:
        return "Fish {} starting at {} stopping at {}".format(self.id, self.starttime, self.stoptime)
        
    def __str__(self) -> str:
        return "Fish {}".format(self.id)
    
    def __len__(self) -> int:
        return len(self.eod)

class NumpyGrid:
    def __init__(self, recpath: str) -> None:
        
        # try to load files from disk
        try:
            self.times = np.load(f"{recpath}times.npy", allow_pickle=True)
            self.idx_v = np.load(f"{recpath}idx_v.npy", allow_pickle=True)
            self.fund_v = np.load(f"{recpath}fund_v.npy", allow_pickle=True)
            self.sign_v = np.load(f"{recpath}sign_v.npy", allow_pickle=True)
            self.ident_v = np.load(f"{recpath}ident_v.npy", allow_pickle=True)
            self.xpos = np.load(f"{recpath}xpos.npy", allow_pickle=True)
            self.ypos = np.load(f"{recpath}ypos.npy", allow_pickle=True)
            self.temp = np.load(f"{recpath}temp.npy", allow_pickle=True)
            self.light = np.load(f"{recpath}light.npy", allow_pickle=True)
            self.sex = np.load(f"{recpath}sex.npy", allow_pickle=True)
            self.ids = np.asarray(np.unique(self.ident_v), dtype=int)

        except FileNotFoundError as error:
            logger.error(error)
            raise error

        # convert whole data arrays to list of fish objects
        self.fish = []
        for track_id in self.ids:
            fund = self.fund_v[self.ident_v == track_id]
            xpos = self.xpos[self.xpos == track_id]
            ypos = self.ypos[self.ypos == track_id]
            starttime = 
            stoptime = 

            fish = Fish(track_id, fund, xpos, ypos, starttime, stoptime)
            self.fish.append

    def toNix(self, outpath: str) -> None:
        pass

class NixGrid:
    def __init__(self, recpath: str) -> None:
        pass

    def toNumpy(self, outpath: str) -> None:         
        pass