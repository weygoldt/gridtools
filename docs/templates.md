# Template library

The simulator's chirps are real chirps: frequency traces relative to the
emitter's baseline, extracted from recordings, so a template fits a fish of
any EODf. Parametric chirp models are too smooth; real templates bring the
variety of shapes (undershoots, plateaus, slow decays, ripples) a detector
has to see.

## Layout

```text
LIBRARY/
  delta_f_hz.npy               (n_templates, n_samples) float32: excursion in Hz, alignment point at n_samples // 2
  amp_rel.npy                  (n_templates, n_samples) float32: measured relative amplitude
  templates.parquet            one row per template: measurements, quality flags, source, group
  near_duplicate_pairs.parquet scored candidate pairs of the duplicate check
  manifest.json                config, source files and hashes, code version, counts
  report.md, figures/          the library's report
  splits_v1.json, splits_v1.md fish-disjoint train/val/test split (v4)
```

Outside its support a template is exactly 0 (frequency) and 1 (amplitude).

```python
from gridtools.templates import load_library

library = load_library("LIBRARY", mmap=True)  # memory-mapped: only drawn rows are read
library.n_templates, library.samplerate_hz
library.metadata  # templates.parquet
```

## Versions

| version | contents |
|---|---|
| v1, v2 | 13,015 chirps of the MSc thesis' chirp-chamber extraction (`extracted_chirps.h5`, 0.4-s snippets at 20 kHz, *A. leptorhynchus*; the source recordings are lost). v2 = v1 with 0.4-s templates and the duplicate window |
| v3 | v2 plus 533 long chirps harvested from the 2019/2020 competition recordings (minus review vetoes); 1.2-s templates |
| v4 | the arrays of v3 with the final selection and sampling rules (below); used for every final training set |

## Building

### 1. The base library

```python
from gridtools.templates import BuildConfig, build_library, write_report

build_library("extracted_chirps.h5", "templates/v2", BuildConfig())
write_report("templates/v2", h5_path="extracted_chirps.h5")  # needs the plot extra
```

The build undoes the extraction's circular shift, finds each chirp's
baseline, support and alignment, tapers it to a smooth baseline, measures it
(excursion, 10 % and 50 % widths, undershoot, amplitude dip, asymmetry,
ripple, drift, steps) and flags duplicates and artefacts. It is
deterministic. `manifest.json` records the config used (v2: the defaults of
`BuildConfig` at the time; four keys were added later).

### 2. Long chirps from grid recordings

The chirp-chamber snippets are 0.4 s long, so no template there exceeds
about 150 ms. Long chirps (300-450 ms: an abrupt rise of 400-550 Hz, a
plateau, a slow decay) are harvested from the competition recordings:

```python
from gridtools.templates.harvest import HarvestConfig, run_harvest
from gridtools.templates.harvest_plots import (
    write_contact_sheets,
    write_rejection_examples,
)

records = run_harvest(
    recording_dirs,
    "templates/long_chirps",
    HarvestConfig(),
    individuals_csv="order_meta.csv",
    workers=6,
)
write_contact_sheets("templates/long_chirps")  # review sheets (plot extra)
write_rejection_examples("templates/long_chirps", "/data/2019_tube_competition")
```

Candidates are long, tall chirpdetector v1 boxes. For each, the emitter is
the fish whose baseline band loses the most power during the chirp (v1's
assignment is only a hint); all electrodes are band-passed and combined
with weights matched to the emitter's field pattern, with the other fish's
patterns (fundamental and second harmonic) projected out; the instantaneous
frequency is a power-weighted phase-advance estimate with a window that
widens where the amplitude drops (large chirps fall to 5-20 % of the
baseline amplitude on their plateau). Candidates with another fish or its
harmonic inside the chirp's band, failed extractions and duplicates are
rejected, with the reason in `candidates.csv`. Bad templates are vetoed in
`review.csv` (`accept = False`) after looking at the contact sheets.

### 3. Merge

```python
from gridtools.templates.merge import build_merged_library
from gridtools.templates.report_merged import write_v4_report

build_merged_library(
    "templates/v2",
    "templates/long_chirps/harvest",
    "templates/v4",
    review_csv="templates/long_chirps/review.csv",
)
write_v4_report("templates/v4")
```

Base templates keep their ids; the length becomes the smallest multiple of
0.1 s that holds the widest harvested support (1.2 s), padding the base
templates exactly.

### 4. Fish-disjoint splits

```python
from gridtools.templates import (
    load_library,
    assign_template_splits,
    write_splits,
    splits_report,
    load_splits,
)

library = load_library("templates/v4", mmap=True)
splits = assign_template_splits(
    library, seed=0, fractions=(0.8, 0.1, 0.1), pins={"competition/g6f5": "train"}
)
write_splits(splits, "templates/v4/splits_v1.json", "templates/v4")
load_splits("templates/v4/splits_v1.json").counts()
# {'train': {'units': 625, 'groups': 628, 'templates': 11230},
#  'val': {'units': 78, 'groups': 79, 'templates': 1124},
#  'test': {'units': 78, 'groups': 79, 'templates': 1194}}
```

The unit of the split is the fish where the library knows it (harvested
templates of a known individual, across all its recordings, e.g.
`competition/g6f5`), else its group (a chirp-chamber recording group, or
`<recording>/track<id>`). Units are assigned 80/10/10, stratified so that
validation and test hold small, big and long chirps; `g6f5`, the fish with
most harvested long chirps, is pinned to training. Run with these
arguments, the call above reproduces `splits_v1.json` exactly. Scenes then
select a split with `[chirps] template_split` and `template_splits`.

## Quality flags and the default selection

Every template carries flags; the default selection (the `exclude` list of
the scene config, the library's `DEFAULT_EXCLUDE`) drops every flagged
template: `invalid`, `has_nan`, `duplicate` (exact or near duplicates),
`truncated`, `step_flag` (step artefacts), `too_short` (shorter than 5 ms at
10 % height: not real chirps), `brief_cycles` (FWHM of at most 2 EOD
cycles), `ripple_flag` (beat ripple from another fish), `multi_peak`,
`line_in_span` (another fish's line in the chirp's band), `baseline_drift`,
`low_snr`, `decay_interrupted`. Of v4's 13,548 templates, 12,692 pass.

## Sampling

`TemplateLibrary.sample` and `TemplateLibrary.cycler` (used by the
simulator) draw templates of the selected families (`small`, `big`) with
equalisation over a 10 x 10 grid of excursion x 10 %-width bins, so that
rare big and long chirps are drawn often:

- log-spaced bins (`equalize_spacing = "log"`);
- tempered weights: a bin is drawn with weight count^(1 - alpha), alpha 0.6
  (`alpha = 1` is flat);
- individual first (`equalize_balance = "individual"`): within a bin, pick a
  fish (or group), then one of its templates, so one prolific fish cannot
  dominate a bin.

Repeats of a template are fine: its context (noise, other fish, position in
the spectrogram) varies with every scene. The amplitude envelope of a
placed chirp comes from its frequency trace (`amplitude_from_frequency`,
dip drawn in [0, 0.8]); measured amplitude traces (`amp_rel`,
`amp_usable`) are an alternative (`amplitude_mode = "measured"`).
