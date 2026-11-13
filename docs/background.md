# Background library

Simulated electrode noise is Gaussian; real recordings are not (hum with
odd harmonics, clicks, non-stationary interference, other fish). Hybrid
scenes therefore put the simulated fish on real one-minute segments of grid
recordings that hold no chirps, scaled to a controlled SNR, with the real
fish in them tracked and counted as non-chirping candidates.

The difficulty is "no chirps": real fish in a segment may chirp, their
chirps are not labelled, and a detector trained on such scenes would be
punished for finding them. Library v2 therefore takes competition segments
only from the fish's inactive period and screens every candidate with two
chirp detectors.

## Library v2

1,342 segments (98.9 GB): 780 from the 2019/2020 tube competitions, 411
from the 2022 competitions, and 151 fish-free segments from two field
deployments (`field_iriri2026`, `field_flona2025`), carried over from v1.

- **Windows.** Competition segments come only from the fish's inactive
  (light) period with 30 min of padding, measured from the recording's end:
  the last 2.5 h of the ~6-h 2019/2020 recordings and the last 0.5 h of the
  ~4-h 2022 recordings, never earlier than 3.5 h after the start (the first
  3 h of every recording are dark).
- **Screens.** A candidate segment is kept only if neither screen has a box
  within the segment +-1 s: a re-run of the v1 production detector
  (frozen code and weights; at score >= 0.5 it reproduces all saved v1
  boxes) at score >= 0.1, and a YOLO26s model trained on hybrid scenes at
  >= 0.2. Spans a screen did not cover count as failing.
- **Recording gate.** A recording whose window holds more than 20 v1 boxes
  per hour (score >= 0.5) is dropped entirely: at v1's recall of 0.76, a
  v1-clean segment of such a recording still carries about 0.1 missed
  chirps. This dropped 20 of 47 screened 2019/2020 recordings and 1 of 17
  from 2022; in many of them fish keep chirping after lights-on.
- **Labelled recordings.** The four hand-labelled recordings (the real test
  set) never enter the library.
- **Selection.** Candidates start every 5 s, at least 60 s inside the
  recording; the window is cut into strata, each contributing the valid
  candidate with the fewest expected missed chirps and the best track
  coverage; empty strata are filled with the remaining valid candidates
  (`fill`), and the 30-min 2022 windows are packed from the earliest
  candidate on (`pack`). Up to 30 segments per recording, never
  overlapping.

## Building it

```sh
python -m gridtools.background carry-over LIB_V1 LIB_V2 --datasets field_flona2025 field_iriri2026
python -m gridtools.background build LIB_V2 --config v2.toml --datasets competition2019 competition2022
python -m gridtools.background review LIB_V2          # one PNG per segment + review.csv
python -m gridtools.background report LIB_V2          # report.md + figures
python -m gridtools.background diagnose LIB_V2 --screen v1 --snippets 2020-03-18-10_34
```

`build` processes every source recording in its own process with its own
random stream (derived from the library seed and the recording name), so the
result does not depend on the order, the subset or the number of workers.
Per recording it chooses segments, reads each once, measures it (noise
floor, real fish and their SNR, hum, clicks, tonal peaks), and writes it
with its tracks; channels without fish signal in any segment are marked
dead. `--recordings-2019` / `--recordings-2022` restrict the recordings,
`--root-2019` etc. give the data locations (required for every dataset in
`--datasets`), `--workers` the processes.

The build config (`BackgroundConfig`, TOML; its keys replace the
command-line defaults) of library v2:

```toml
seed = 0
workers = 6
exclude_labelled = true
fish_free_per_recording = 2
use_saved_detections = false       # the v1 screen replaces the saved v1 detections

[selection]
segment_s = 60.0
step_s = 5.0
margin_s = 1.0                     # margin of hand labels and review vetoes
edge_s = 60.0
segments_per_recording = 30
fill = true

[selection_by_dataset.competition2022]
pack = true

[windows.competition2019]          # last 2.5 h, never before 3.5 h
last_s = 9000.0
min_start_s = 12600.0
min_duration_s = 14400.0

[windows.competition2022]          # last 0.5 h
last_s = 1800.0
min_start_s = 12600.0
min_duration_s = 10800.0

[[screens]]
name = "v1"
root = "screens/v1"
threshold = 0.1
margin_s = 1.0

[[screens]]
name = "v2"
root = "screens/v2_yolo26s"
threshold = 0.2
margin_s = 1.0

[gate]
screen = "v1"
threshold = 0.5
max_rate_per_h = 20.0
```

A two-segment library from one 2019 recording, built with this config
(`segments_per_recording = 2`), took 6 s; `review`, `report` and
`diagnose` ran on it.

### Screens

Chirp detectors run outside gridtools and write a *screen directory*:

```text
SCREEN/screen.json                 detector, weights sha256, store threshold
SCREEN/<dataset>/<recording>.csv   boxes: t_start_s, t_end_s, f_low_hz, f_high_hz, score
SCREEN/<dataset>/<recording>.json  covered_s: [[start, stop], ...] spans the detector processed
```

Every box down to the store threshold is listed, so a screen's threshold can
change without re-running the detector. The two screens of library v2 were
written by `scripts/v1_screen.py` (the frozen v1 detector in its own
environment) and `scripts/v2_screen.py` (a chirpdetector bundle) of the
chirp_competition_upgrade workspace.

### Review

`review` draws every segment (electrode-summed spectrogram, 102-ms window,
10-s rows) into `LIB/review/` with an HTML contact sheet per source, and
merges `review.csv`, where `accept = False` (and a `note`) vetoes a segment.
Existing decisions are never overwritten; the library drops vetoed segments,
and a rebuild treats them like chirp detections.

`diagnose` draws, per competition recording, the screen boxes per 30 min
over the whole recording with the dark phase, the library window and the
gate rate, a gate overview, and 10-s snippets around screen boxes in the
light phase, to tell real chirps from artefacts.

## Layout

```text
LIB/
  library.json            builds, carried-over datasets, sizes
  manifest.parquet/.csv   one row per segment: source, offset, levels, real fish, flags, split group
  recordings.csv          per recording: window, gate rate, candidates, segments, why skipped
  screen_counts.csv       candidates rejected per screen
  review.csv              the author's vetoes
  report.md, figures/     the report
  segments/<segment_id>/
    background.wav        float32 samples at the source rate, in the source's unit
    fishgrid.cfg          the source's configuration (geometry)
    times.npy fund_v.npy ident_v.npy idx_v.npy sign_v.npy   the real fish's tracks (absent if fish-free)
    segment.json          per-electrode floors, per-fish measurements, tonal peaks, hum, clicks
```

## Using it

Scenes use the library through `[background]` of the scene config
([Simulating scenes](simulation.md#hybrid-scenes)). Split training,
validation and test scenes by source recording (`split_group`, e.g.
`competition2019/2019-11-28-09_58`, in `include_groups`). In Python:

```python
import numpy as np
from gridtools.sim.background import BackgroundLibrary

rng = np.random.default_rng(0)

library = BackgroundLibrary.load("LIB")
segment = library.select(screened=True).sample(rng, 60.0, n_electrodes=6)
segment.geometry, segment.samplerate_hz, segment.tracks
```

## Levels and SNR

Spectra are one-sided Welch PSDs per electrode (electrodes are never summed
as signals: fish fields have opposite polarity on different electrodes,
common-mode interference does not). The background level is the median
noise floor in 500-1500 Hz without tonal bins, times a reference bandwidth
of 7.32 Hz (one bin of a 4096-point Hann spectrogram at 20 kHz). A fish's
level is the power of its EOD fundamental on its strongest electrode. SNR is
10 log10(signal / background) in dB; the background is scaled by
sqrt(S / (B 10^(SNR/10))) to reach a target. Real fish are scaled with it,
so their own SNR does not change.

## Library v1

v1 (2026-11-12) took competition segments from whole recordings, screened
only by v1's saved detections at score >= 0.5. It is kept on disk for
reference; the final training scenes use v2.
