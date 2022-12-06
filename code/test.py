import matplotlib.pyplot as plt
from gridtools.prepro import GridCleaner

datapath = "/mnt/backups/@data/2016-04-20-18_49/"

grid = GridCleaner(datapath)

print(f"Recorded from {grid.starttime} to {grid.stoptime}.")

# run prepro functions
grid.fillPowers()
# grid.loadLogger()
# 
# grid.purgeUnassigned()
# grid.purgeShort(thresh=3600)
# grid.purgeBad(0.7)
# 
# grid.triangPositions(electrode_number=6)
# grid.interpolateAll()
# grid.sexFish()

# grid.integrityCheck()
# grid.saveData()
