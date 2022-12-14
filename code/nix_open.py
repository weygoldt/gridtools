import nixio


class FishLoader:
    def __init__(self, block) -> None:
        self.eod = block.data_arrays[0]
        self.pos = block.data_arrays[1]

    def __repr__(self) -> str:
        return f"Fish"


class GridLoader:
    def __init__(self, filepath, filemode: str = "ReadOnly") -> None:

        if filemode == "ReadOnly":
            mode = nixio.FileMode.ReadOnly
        elif filemode == "ReadWrite":
            mode = nixio.FileMode.ReadWrite

        data = nixio.File.open(filepath, mode)

        self.fish = []
        for block in data.blocks:
            self.fish.append(FishLoader(block))


grid = GridLoader("testfile.nix", filemode="ReadWrite")
