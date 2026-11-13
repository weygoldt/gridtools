# Simulating scenes

```sh
gridtools sim run CONFIG --seeds 0:10 --out OUT [--workers N] [--no-plots]
gridtools sim plot OUT/000003 [--out DIR]
gridtools sim compare OUT [--calibration calibration.json]
```

`CONFIG` is a TOML file or the name of a shipped config (`default`).
`--seeds` takes `0:10` (stop excluded), `0:10:2`, `3` or `1,4,7`; every seed
gives one scene directory `OUT/<seed:06d>/`, and `OUT/config.json` records
the resolved config. Without `--out`, scenes go to
`/mnt/data1/chirp_v2/scenes/<name>` (a path of the development machine).
`--no-plots` skips the two annotated spectrograms per scene.

A short synthetic scene, as used to check these docs:

```toml
# tiny.toml: 20 s, two or three fish on a 3 x 3 grid, no real background
name = "tiny"
duration_s = 20.0

[grid]
layouts = [[3, 3]]
real_layout_prob = 1.0

[population]
n_fish = [2, 3]

[chirps]
rate_hz = [0.2, 0.5]

[background]
prob = 0.0
```

```text
$ gridtools sim run tiny.toml --seeds 0:2 --out scenes
INFO gridtools.sim.cli: seed 000000: 3 fish, 9 electrodes, 25 chirps, synthetic background; simulate 3.9 s, write 0.0 s, plot 1.2 s; peak RSS 537 MB
INFO gridtools.sim.cli: seed 000001: 3 fish, 9 electrodes, 27 chirps, synthetic background; simulate 3.1 s, write 0.0 s, plot 0.9 s; peak RSS 564 MB
INFO gridtools.sim.cli: 2 scenes in scenes (9.2 s wall)
```

## Configs

A config describes the *distributions* a scene is drawn from; a seed picks
one scene. Ranges are `[low, high]`, drawn uniformly per scene or per fish
(log-uniformly where noted). Units are in the key names, recordings are in
volts at the ADC like the real raw data, and every time and frequency
setting is physical, so `samplerate_hz` (or `samplerate_choices_hz`, one
rate drawn per scene) changes nothing else. Unknown keys are an error. A
config only needs the keys it changes; the shipped `default.toml` lists
all of them with their defaults (documentation of every field: the
dataclasses in `gridtools.sim.config`):

| section | what it draws |
|---|---|
| top level | `name`, `duration_s`, `samplerate_hz`, `samplerate_choices_hz` |
| `[grid]` | electrode layout, spacing, rotation, jitter, in-tank reference |
| `[population]` | number of fish, baseline EODf (female and male modes, broad component), close pairs and crossings, ramps, individual and shared drift, harmonic waveform |
| `[chirps]` | template library, families, exclusions, equalisation, fish-disjoint split, rate, bursts, responses, simultaneous chirps, amplitude dip |
| `[movement]` | resting (light) or active (dark) regime per scene or fish, random-walk parameters |
| `[field]` | EOD amplitude distribution, dipole falloff exponent and regularisation |
| `[noise]` | white/coloured electrode noise floors, harder-than-real offset, 50-Hz hum and harmonics, gain mismatch, dead and noisy electrodes, clipping |
| `[clicks]` | rate, duration, amplitude, electrodes, pairs, periodic and LED-sync clicks |
| `[tracks]` | frame grid of the ground-truth tracks and `sign_v` resolution |
| `[footprint]` | the reference spectrogram of the chirp boxes, box threshold, visibility threshold |
| `[background]` | hybrid scenes: library, probability, SNR distribution, datasets and groups |
| `[plot]` | the annotated spectrograms |

The defaults are calibrated on the 33 tube-competition trials
([Calibration](calibration.md)): noise floors, hum, faults and clipping as
measured, EOD amplitudes and a dipole field fitted to real per-electrode
power profiles, baseline EODf, drift, movement regimes and chirp timing.
Scenes are deliberately harder than the real trials: close frequencies
(with probability `close_pair_prob`, 0.5, a fish is placed 0-20 Hz from an
earlier one, and half of these pairs cross), near-simultaneous chirps,
higher chirp rates and noisier recordings.

## Electrode layouts

With probability `grid.real_layout_prob` a scene uses one of the real
layouts (`[[3, 3], [3, 5], [5, 3]]`); otherwise a drawn layout of 1-16
electrodes (1-4 with probability `small_prob`), rectangles up to 3:1 or
near-square layouts with gaps, jittered and optionally rotated. The spacing
is drawn per scene (0.15-0.5 m: the spacing of the competition grids is not
documented), and with probability `reference_prob` the recording reference
is an electrode in the tank instead of at infinity. Detectors trained on
these scenes cannot rely on absolute positions, spacing or channel order.

Layouts that are not a full rectangle are written with an `electrodes.csv`
(`channel, x_m, y_m`) next to the wav; `gridtools.io.open_recording` reads
it.

## Hybrid scenes

With probability `background.prob`, a scene uses a real 60-s background
segment from the [background library](background.md) instead of synthetic
noise:

- the segment's electrodes (the whole grid, or a contiguous run of a larger
  one) with their positions and channel order replace the drawn layout, and
  the simulated fish are rendered on them;
- synthetic noise, hum and clicks are off; the segment brings its own;
- the segment is scaled to a drawn SNR of the simulated fish
  (`snr_hard_db` [10, 35] dB for 40 % of scenes, else `snr_typical_db`
  [35, 80] dB; real fish at most `max_real_gap_db` louder than the
  simulated ones). Hard SNRs make the simulated fish quieter rather than
  amplify the background;
- the real fish in the segment join the tracks with identities from 1000 on
  (listed in `scene.json` `real_fish`); they never chirp in the ground
  truth.

Hybrid scenes are at most 60 s long (the segment length). `include_groups`
/ `exclude_groups` restrict the source recordings (e.g.
`competition2019/2019-11-28-09_58`), which is how training, validation and
test scenes get disjoint background recordings; `unscreened_weight` lowers
the sampling weight of segments without a chirp screen.

```toml
# hybrid.toml
name = "hybrid"
duration_s = 60.0

[chirps]
template_split = "test"
template_splits = "/libraries/templates/v4/splits_v1.json"

[background]
library = "/libraries/background/v2"
prob = 1.0
```

```text
$ gridtools sim run hybrid.toml --seeds 0:1 --out hybrid_scenes
INFO gridtools.sim.cli: seed 000000: 4 fish, 6 electrodes, 77 chirps, competition2019_2019-11-28-09_58_015672 (2 real fish, SNR 26 dB) background; ...
```

## Chirps

Chirps are real chirps from the [template library](templates.md)
(`[chirps] library`), relative to the baseline, so a template fits a fish of
any EODf. Templates are drawn with the library's tempered equalisation over
a 10 x 10 grid of excursion x 10 %-width bins (log-spaced, bin weight
count^(1 - 0.6)), choosing an individual first and then one of its
templates, so rare big and long chirps are seen often. The
amplitude envelope is the inverted frequency trace scaled to a dip drawn
from `dip` (EOD amplitude 100-20 % at the peak). Timing: bursts of
spontaneous chirps per fish (`rate_hz`, `burst_size_mean`,
`burst_interval_s`), responses by other fish (`response_prob`,
`response_delay_s`), near-simultaneous chirps (`simultaneous_fraction`), a
minimum interval between one fish's chirps.

`template_split` (`train`, `val` or `test`) with `template_splits` (a
split file of the library) restricts a scene to the templates of one
fish-disjoint split, so test scenes carry chirps of fish never seen in
training.

## The training sets of chirpdetector

The released detector was trained on `train_v4`: 3,000 one-minute hybrid
scenes, `train` seeds 5000-7399, `val` 7400-7699, `test` 7700-7999, each
split with its own config that differs only in `template_split` and the
background recordings (`include_groups`, disjoint across splits), with
`close_pair_prob = 0.3`, `rate_hz = [0.05, 0.25]` and background library v2
at `prob = 1.0`. The configs are in the chirp_competition_upgrade workspace
(`configs/scenes/train_v4_*.toml`).

## In Python

```python
from gridtools.sim.config import SceneConfig
from gridtools.sim.footprint import HannSpectrogram, chirp_footprint
from gridtools.sim.scene import render_fish, simulate_scene, write_scene

scene = simulate_scene(SceneConfig.from_toml("tiny.toml"), seed=3)
path = write_scene(scene, "out/000003")

# one fish's exact contribution (samples x electrodes), with and without chirp 12
with_chirp = render_fish(path, 0, 10.0, 10.5)
without = render_fish(path, 0, 10.0, 10.5, chirps=[12])

# chirp 12's box in a spectrogram of your choice
box = chirp_footprint(path, 12, HannSpectrogram(window_s=0.0256, hop_s=0.0016)).box
```

`recording = sum of render_fish over all fish + render_background` holds
to float32 rounding (unless clipping is configured); `render_background`
reloads a hybrid scene's scaled segment from the library named in
`scene.json`.

## Comparing scenes with real recordings

`gridtools sim compare OUT` measures the scenes with the calibration's own
estimators (noise floor, hum, per-electrode power profiles, strongest
electrode switches, drift, chirp intervals, clicks) and tabulates them
against the real recordings (`OUT/comparison.md`, `.csv`). Scenes are
meant to be harder than real, so this is a plausibility check, not a fit.
