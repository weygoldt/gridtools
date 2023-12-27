"""# Datasets.

Classes and functions to load, work with and save data associated with
electrode grid recordings of wave-type weakly electric fish.

The main functionalities include the following:
- gridtools.datasets.load: Load a dataset from a given path.
- gridtools.datasets.save: Save a dataset to disk.
- gridtools.datasets.subset: Make a subset of a dataset.

## Architecture and design principles

The architecture of the `datasets` module follows these design principles:
- **Composition over inheritance**: The `Dataset` class is a composition of
different subclasses, making it easily extensible to other data types in the
future.
- **Data models**: The `Dataset` class is not just a dataclass but a data
model: Upon instantiation, the data is checked for consistency and errors are
raised if the data is inconsistent.

The Dataset class is a composition of:
- `GridData`: The raw recording from the electrode grid.
- `WavetrackerData`: Tracking arrays produced by - and derived from - the
[`wavetracker`](https://github.com/tillraab/wavetracker.git).
- `CommunicationData`: Chirp and rise times, identifiers and optionally,
extracted parameters such as height, width, etc.

## Usage

Loading a dataset is as easy as calling the `load` function with the path to
the dataset as an argument. The function returns a `Dataset` object containing
the loaded data.
```python
from gridtools.datasets import load
ds = load(pathlib.Path("path/to/dataset"))
```

To create a subset of a dataset, use the `subset` function. The function takes
the dataset to subset, the start and stop time of the subset, and the mode
("time" or "index") as arguments. The function returns a new `Dataset` object
containing the subsetted data.
```python
from gridtools.datasets import load, subset
ds = load(pathlib.Path("path/to/dataset"))
subset = subset(ds, 0.1, 0.5) # default mode is "time"
```

To save this subset to disk, use the `save` function. The function takes the
subsetted dataset and the path to the directory where the dataset should be
saved as arguments.
```python
from gridtools.datasets import load, subset, save
ds = load(pathlib.Path("path/to/dataset"))
subset = subset(ds, 0.1, 0.5)
save(subset, pathlib.Path("path/to/save"))
```
"""

from .loaders import load
from .savers import save
from .subsetters import subset

__all__ = ["load", "save", "subset"]
