<p align="center">
  <img src="assets/logo.svg" width="120" alt="gridtools logo">
</p>

# gridtools

Read and simulate multi-electrode grid recordings of wave-type weakly
electric fish.

- **I/O** (`gridtools.io`): fishgrid recordings and wav files (memory-mapped,
  in volts), electrode geometry from `fishgrid.cfg` or `electrodes.csv`,
  wavetracker frequency tracks, audian label files and chirpdetector v1
  detections.
- **Simulator** (`gridtools sim`): complete synthetic grid recordings of
  chirping *Apteronotus leptorhynchus* with exact ground truth: EODs with
  harmonics and drift, real chirp templates, a dipole field from moving fish
  to the electrodes, noise, hum, clicks and electrode faults, or real
  background recordings with their own noise and fish (hybrid scenes). Every
  chirp gets the box it occupies in a spectrogram (its footprint) and a
  visibility, and its parameters are defined on the emitter's noise-free
  per-EOD-period frequency.
- **Chirp template library** (`gridtools.templates`): real chirps extracted
  from recordings, cleaned, quality-flagged, sampled with equalisation over
  size and duration, and split by fish into train, validation and test.
- **Background library** (`gridtools.background`): one-minute segments of
  real recordings without chirps, screened by chirp detectors, measured
  (noise floor, real fish, hum, clicks) and used at a controlled SNR.
- **Calibration** (`gridtools.calibrate`): the measurements of real
  recordings that the simulator's defaults come from.

gridtools makes the training and test data of
[chirpdetector](https://github.com/weygoldt/chirpdetector).

## Installation

```sh
uv pip install "gridtools @ git+https://github.com/weygoldt/gridtools"
uv pip install "gridtools[plot] @ git+https://github.com/weygoldt/gridtools"   # with the annotated spectrograms and reports
```

Python 3.13 or later. The PyPI package named `gridtools` is unrelated:
install from GitHub.

## Simulate a scene

```sh
gridtools sim run default --seeds 0:10 --out scenes/default
gridtools sim run my.toml --seeds 0:100 --out scenes/my --workers 8
gridtools sim plot scenes/my/000003
gridtools sim compare scenes/my --calibration calibration.json
```

`default` is the shipped config
(`src/gridtools/configs/scenes/default.toml`, every key with its default);
copy it to make your own. Each seed gives one scene directory
(`OUT/<seed:06d>/`): `recording.wav` (float32, one channel per electrode,
in volts), `fishgrid.cfg`, ground-truth tracks in wavetracker's format,
`chirps.csv`/`.parquet` with every chirp's timing, size, footprint box and
visibility, `clicks.csv`, `paths.npz`, `chirp_traces.npz`, `scene.json` and
two annotated spectrograms.

The simulator needs a chirp template library, and hybrid scenes a background
library. Both are built from the chirp competition recordings and are not
public yet; the shipped config points to their location on the development
machine (`[chirps] library`, `[background] library`). Hosting them is an open
item.

## Documentation

| page | contents |
|---|---|
| [Installation](docs/installation.md) | uv, pip, extras |
| [Simulating scenes](docs/simulation.md) | configs, layouts, hybrid scenes, templates and splits, seeds, the Python API |
| [Scene outputs](docs/outputs.md) | directory layout, `chirps.csv` columns, tracks, `scene.json`, plots |
| [Ground truth](docs/ground-truth.md) | the per-period instantaneous frequency, footprints, visibility |
| [Template library](docs/templates.md) | building it, the long-chirp harvest, QC flags, sampling, fish-disjoint splits |
| [Background library](docs/background.md) | windows, chirp screens, the recording gate, building, review |
| [Reading recordings](docs/io.md) | recordings, geometry, tracks, labels |
| [Calibration](docs/calibration.md) | what the simulator's defaults are fitted to |
| [Development](docs/development.md) | tests, checks, the docs site |

The pages build as a site with `uvx --from mkdocs==1.6.1 --with
mkdocs-material==9.8.1 mkdocs build --strict`.

## Versions

This is version 2. The code used for the chirp competition paper (2024) is
tag `paper-chirp-competition-v1`.

## License

MIT
