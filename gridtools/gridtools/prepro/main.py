import numpy as np

from .gridcleaner import GridCleaner


path = "/mnt/backups/@data/2016-04-20-18_49/"

g = GridCleaner(datapath=path)

print("")
print("Wavetracker output files:")
print(f"Shape of self.times: {np.shape(g.times)}")
print(f"Shape of self.idx_v: {np.shape(g.idx_v)}")
print(f"Length of unique values in idx_v: {np.shape(np.unique(g.idx_v))}")
print(f"Shape of self.fund_v: {np.shape(g.fund_v)}")
print(f"Shape of self.sign_v: {np.shape(g.sign_v)}")
print(f"Shape of self.ident_v: {np.shape(g.ident_v)}")
print(f"Shape of self.ids: {np.shape(g.ids)}")

print("")
print("Grid metadata from dataloader")
print(f"grid_rate: {g.grid_rate}")
print(f"grid_spacings: {g.grid_spacings}")
print(f"grid_grid: {g.grid_grid}")

print(g.starttime)
print(g.stoptime)
