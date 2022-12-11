import nixio

data = nixio.File.open("testfile.nix", nixio.FileMode.ReadWrite)
fish = data.blocks[0]
fish_eod = fish.data_arrays[0]
fish_time = fish_eod.dimensions[0].axis(len(fish_eod))
