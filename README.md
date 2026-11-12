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
(every key with its default; copy it to make your own). Each scene directory
holds `recording.wav` (float32, one channel per electrode), `fishgrid.cfg`,
ground-truth tracks in wavetracker layout, `chirps.csv`/`.parquet`,
`clicks.csv`, `paths.npz`, `chirp_traces.npz`, `scene.json` and two annotated
spectrograms. In Python:

```python
from gridtools.sim.config import SceneConfig
from gridtools.sim.scene import render_fish, simulate_scene, write_scene

scene = simulate_scene(SceneConfig.from_toml("my.toml"), seed=3)
path = write_scene(scene, "out/000003")
# one fish's exact contribution, with and without chirp 12
with_chirp = render_fish(path, 0, 10.0, 10.5)
without = render_fish(path, 0, 10.0, 10.5, chirps=[12])
```

## Development

```sh
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

## License

MIT
