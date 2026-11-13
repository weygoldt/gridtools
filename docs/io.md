# Reading recordings

`gridtools.io` reads grid recordings, their electrode geometry, frequency
tracks and labels. Everything is read-only and memory-mapped: opening a
26-GB recording is instant, and reading a few seconds costs only those
seconds.

```python
from gridtools.io import load_tracks, open_recording, read_labels, read_v1_detections

rec = open_recording("/data/2019_tube_competition/2019-12-09-10_00")
print(rec.samplerate_hz, rec.n_channels, rec.duration_s)  # 20000.0 9 21540.27
rec.geometry  # GridGeometry from fishgrid.cfg
samples = rec.read_time(1140.0, 1141.0)  # (20000, 9) float32, volts
samples = rec.read(0, 20000, channels=[0, 1])  # by sample index

tracks = load_tracks("/data/2019_tube_competition/2019-12-09-10_00")
tracks.frequencies_at(1140.0, max_gap_s=10.0)  # {5: 738.8, 9: 776.8}
tracks.track(5).frequency_at([1140.0, 1141.0], max_gap_s=10.0)

labels = read_labels(".../traces-grid1-editable-labels.csv")  # audian boxes
v1 = read_v1_detections(".../chirpdetector_bboxes.csv")  # chirpdetector v1 output
```

## Recordings

Two layouts:

- **fishgrid directories**: `traces-grid1.raw` (interleaved float32, one
  column per electrode) next to `fishgrid.cfg`. Samples are volts at the
  ADC: on the tube competition recordings the quantisation step is 0.625 µV
  and the range ±20.48 mV, although the configuration and thunderlab label
  the data mV. Electrode potentials are raw / 100 (amplifier gain). Some
  files end in a partial frame (an interrupted write); it is dropped with a
  warning.
- **wav files** (PCM or float, one channel per electrode), e.g. excerpts or
  simulated scenes. Integer PCM is scaled to [-1, 1). Electrode positions
  come from an `electrodes.csv` (`channel, x_m, y_m`) next to the wav, else
  from `fishgrid.cfg`.

## Electrode geometry

`read_fishgrid_config` parses `fishgrid.cfg` (indented `key : value` lines
with units, `50.0cm`, `20.000kHz`); `GridGeometry` holds the positions.
Only the first grid is supported. The mapping from channel index to
electrode position is not documented in the configuration files;
`GridGeometry.regular` assumes row-major order, which could be neither
confirmed nor rejected on the tube competition recordings, and their
electrode spacing is disputed (the `fishgrid.cfg` value of 0.5 m is likely
wrong). Prefer features that do not depend on positions or channel order.

## Tracks

A wavetracker result directory holds five arrays:

| file | shape | content |
|---|---|---|
| `times.npy` | (T,) | time of every spectrogram frame, s |
| `fund_v.npy` | (N,) | fundamental frequency of every detection, Hz |
| `ident_v.npy` | (N,) | track identity of every detection; NaN = unassigned |
| `idx_v.npy` | (N,) | index into `times` of every detection |
| `sign_v.npy` | (N, E) | power of every detection on each electrode, dB |

A track exists only between its first and last detection, and only where
its detections are at most `max_gap_s` apart: `Track.frequency_at` returns
NaN elsewhere instead of extrapolating. `sign_v` may have fewer columns than
the recording has channels (all 2019 tube competition recordings: channel 9
carries no fish signal and `sign_v` holds channels 1-8). `save_tracks`
writes the same layout.

## Labels

audian label files (`read_labels`, `write_labels`) store boxes in time (s)
and frequency (Hz) with a `category`; the hand labels of the chirp
competition data use `labeled_region` (a span in which every chirp was
labelled) and `chirp`. Extra columns (e.g. an emitter) are kept.
`read_v1_detections` maps chirpdetector v1's `chirpdetector_bboxes.csv`
onto the same box columns.
