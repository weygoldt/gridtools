# gridtools documentation

Read and simulate multi-electrode grid recordings of wave-type weakly
electric fish.

```sh
uv pip install "gridtools[plot] @ git+https://github.com/weygoldt/gridtools"
gridtools sim run default --seeds 0:10 --out scenes/default
```

## What is in it

| module | what |
|---|---|
| `gridtools.io` | recordings (fishgrid raw, wav), electrode geometry, wavetracker tracks, audian labels, chirpdetector v1 detections |
| `gridtools.sim` | the scene simulator and `gridtools sim` |
| `gridtools.templates` | the real-chirp template library: build, harvest long chirps, merge, QC, sampling, fish-disjoint splits |
| `gridtools.background` | the library of real background segments for hybrid scenes |
| `gridtools.calibrate` | measurements of real recordings behind the simulator's defaults |

## How a scene is made

```text
scene config (TOML) + seed
   |
   v
population: 1-5 fish, baseline EODf (female/male modes), drift, close pairs and crossings
movement:   correlated random walk with rest states, in a tank around the grid
chirps:     real templates (library, optionally one fish-disjoint split),
            bursts and responses, amplitude dip from the frequency trace
EOD:        phase-continuous harmonic series locked to the instantaneous frequency
field:      point dipole along the body axis, falloff exponent ~1.5-2
grid:       real layouts (3x3, 3x5) or 1-16 drawn electrodes, spacing 0.15-0.5 m
   |
   +--> synthetic noise, hum, faults, clicks        (synthetic scenes)
   +--> a real background segment at a drawn SNR    (hybrid scenes)
   |
   v
recording.wav + tracks + chirps.csv (with footprint boxes) + scene.json + plots
```

Every random draw comes from one seed sequence per scene, with one child
stream per component, so changing one part of the config does not reshuffle
the others. Every fish's contribution can be re-rendered exactly for any
span, with or without any chirp; that defines the chirp boxes and the
ground truth of chirpdetector's parameter decoder.

## Pages

- [Installation](installation.md): uv, pip, extras
- [Simulating scenes](simulation.md): `gridtools sim`, configs, electrode layouts, hybrid scenes, chirps and template splits, the Python API
- [Scene outputs](outputs.md): directory layout, `chirps.csv` columns, tracks, clicks, `scene.json`, plots
- [Ground truth](ground-truth.md): the per-EOD-period instantaneous frequency, re-rendering, footprint boxes, visibility
- [Template library](templates.md): versions, building, the long-chirp harvest, QC flags, sampling, fish-disjoint splits
- [Background library](background.md): library v2 (windows, screens, gate), building, review, layout, levels and SNR
- [Reading recordings](io.md): recordings, geometry, tracks, labels
- [Calibration](calibration.md): the measurements behind the simulator's defaults
- [Development](development.md): tests and checks

Back to the [project README](../README.md).
