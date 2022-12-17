import nixio as nio

file = nio.File.open("testgrid.nix", nio.FileMode.ReadWrite)

rec1 = file.blocks[0]

import os

files = os.listdir(".")
nixfiles = [file for file in files if ".nix" in file]
print(nixfiles)
