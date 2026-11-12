<p align="center">
  <img src="assets/logo.svg" width="120" alt="gridtools logo">
</p>

# gridtools

Tools for multi-electrode grid recordings of wave-type weakly electric fish:
reading recordings, electrode geometry and frequency tracks, and simulating
grid recordings with moving, chirping fish for training and testing
detectors.

> **Status:** version 2 is being rebuilt on the `v2` branch. The code used for
> the chirp competition paper (2024) is available as tag
> `paper-chirp-competition-v1`.

## Installation

```sh
uv add "gridtools @ git+https://github.com/weygoldt/gridtools@v2"
```

The name `gridtools` on PyPI belongs to an unrelated package, so install from
GitHub.

## Simulated scenes

`gridtools sim` writes complete synthetic grid recordings of chirping
*Apteronotus leptorhynchus* with exact ground truth (training data for chirp
detection and emitter assignment):

```sh
uv run gridtools sim run default --seeds 0:10 --out /mnt/data1/chirp_v2/scenes/default
uv run gridtools sim plot /mnt/data1/chirp_v2/scenes/default/000000
```

`default` is the shipped config `src/gridtools/configs/scenes/default.toml`
(every key with its default; copy it to make your own). Its defaults are
calibrated on the 33 tube-competition trials (`gridtools.calibrate`):
recordings in volts, white noise floors, hum, faults and clipping as
measured, EOD amplitudes and a dipole field fitted to real `sign_v`
profiles, baseline EODf, drift, movement regimes and chirp timing. Scenes are
deliberately harder than the real trials (close frequencies, crossings,
near-simultaneous chirps, higher chirp rates, noisier recordings), and vary
the electrode count (1-16), spacing, reference and, optionally, the sampling
rate; every setting is in physical units.

Hybrid scenes (`[background]`, by default 60 % of the scenes) put the
simulated fish on a real 60-s background segment from the background library
(`/mnt/data1/chirp_v2/background/v1`, `gridtools.background`): the
segment's grid (all its channels, or a subset with their positions), its
noise, hum, clicks and non-chirping real fish, which join the tracks with
identities from 1000 on. Synthetic noise, hum and clicks are then off; the
background is scaled to a drawn SNR of the simulated fish (hard SNRs make
the simulated fish quieter rather than amplify the background). Hybrid
scenes are at most 60 s long; set `background.prob = 0` for longer ones.

Each scene directory holds `recording.wav` (float32 V, one channel per
electrode), `fishgrid.cfg` (plus `electrodes.csv` for layouts that are not a
full rectangle), ground-truth tracks in wavetracker layout, `chirps.csv`/
`.parquet` with each chirp's physical extent and its footprint box and
visibility, `clicks.csv`, `paths.npz`, `chirp_traces.npz`, `scene.json` and
two annotated spectrograms. `recording = sum of render_fish + render_background`
(for a hybrid scene, the scaled background segment, reloaded from the library
named in `scene.json`). In Python:

```python
from gridtools.sim.config import SceneConfig
from gridtools.sim.footprint import HannSpectrogram, chirp_footprint
from gridtools.sim.scene import render_fish, simulate_scene, write_scene

scene = simulate_scene(SceneConfig.from_toml("my.toml"), seed=3)
path = write_scene(scene, "out/000003")
# one fish's exact contribution, with and without chirp 12
with_chirp = render_fish(path, 0, 10.0, 10.5)
without = render_fish(path, 0, 10.0, 10.5, chirps=[12])
# chirp 12's box in a spectrogram of your choice
box = chirp_footprint(path, 12, HannSpectrogram(window_s=0.0256, hop_s=0.0016)).box
```

`gridtools sim compare SCENES_DIR` measures scenes with the calibration's
estimators and tabulates them against the real recordings.

## Development

```sh
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

## License

MIT
