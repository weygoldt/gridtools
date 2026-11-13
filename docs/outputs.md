# Scene outputs

```text
OUT/
  config.json                 the resolved config of the run
  000000/                     one directory per seed
    recording.wav             float32 multichannel wav (WAVE_FORMAT_EXTENSIBLE), one channel per electrode
    fishgrid.cfg              sampling rate and grid geometry
    electrodes.csv            channel, x_m, y_m (layouts that are not a full rectangle)
    times.npy fund_v.npy ident_v.npy idx_v.npy sign_v.npy   ground-truth tracks, wavetracker format
    chirps.csv, chirps.parquet  one row per chirp
    clicks.csv                one row per click
    paths.npz                 positions, headings, velocities, rest states of every fish
    chirp_traces.npz          the placed chirp traces exactly as rendered
    scene.json                config, seed, random streams, versions, fish, background, render model
    spectrogram_overview.png  annotated spectrogram of the whole scene
    spectrogram_zooms.png     zooms on hard chirps
```

A scene directory is a recording directory for `gridtools.io` and
chirpdetector: `open_recording(scene)` reads the wav with its geometry and
`load_tracks(scene)` the tracks.

## recording.wav

Unit: volts at the ADC (electrode potential times the amplifier gain),
like the raw competition recordings: EOD amplitudes are drawn in these
units, and a field gain of 1 is the potential 10 cm in front of a fish on
its body axis. Not normalised, and not clipped unless
`noise.clip_limit` is set (the default clips at the competition grids'
ADC range, 20.48 mV). Channels are in row-major grid order, or the
background segment's order in hybrid scenes.

## Tracks

Ground-truth baseline EODf of every fish (with drift and ramps, without
chirp excursions) on frames every `tracks.step_s` (0.1229 s, the production
tracks' grid), in wavetracker's layout: `times.npy` (frames), `fund_v.npy`
(Hz), `ident_v.npy` (fish id; real background fish from 1000 on),
`idx_v.npy` (frame index), `sign_v.npy` (power of each fish's fundamental
on each electrode, dB, as wavetracker computes it).

## chirps.csv

One row per simulated chirp (real background fish never chirp). Times in
seconds from the scene's start, frequencies in Hz.

| column | meaning |
|---|---|
| `chirp_id` | scene-wide id, in order of centre time |
| `emitter_id` | the emitting fish (track ident) |
| `t_onset_s`, `t_offset_s` | first and last crossing of 10 % of the excursion, bracketing the peak |
| `t_center_s` | the template's alignment point (the chirp time) |
| `t_peak_s` | time of the largest excursion |
| `t_support_start_s`, `t_support_stop_s` | where the trace differs from baseline at all |
| `duration_s` | `t_offset_s - t_onset_s` |
| `baseline_hz` | the emitter's baseline at `t_center_s` |
| `excursion_hz` | maximum of the frequency excursion (on the rendered trace) |
| `undershoot_hz` | how far the frequency drops below baseline (>= 0) |
| `f_low_hz`, `f_high_hz` | `baseline_hz - undershoot_hz`, `baseline_hz + excursion_hz` (the trajectory's extent, not the box) |
| `dip` | amplitude dip at the peak (0 = none, 0.8 = amplitude 20 %) |
| `amplitude_mode` | `frequency` (envelope from the frequency trace) or `measured` |
| `template_id`, `family` | template of the library, `small` or `big` |
| `is_response`, `response_to` | generated as a response, and to which chirp (-1: none) |
| `overlaps_other_chirp`, `dt_nearest_other_chirp_s` | another fish's chirp within 0.25 s, and the distance to the nearest one |
| `min_df_hz`, `nearest_fish_id` | distance to the nearest other simulated fish's baseline, and that fish |
| `truncated` | the chirp's support crosses the scene's start or end |
| `emitter_x_m`, `emitter_y_m` | emitter position at the chirp |
| `emitter_power_db_max` | emitter's fundamental power on its strongest electrode (dB, as `sign_v`) |
| `box_t0_s`, `box_t1_s`, `box_f0_hz`, `box_f1_hz` | the footprint box ([Ground truth](ground-truth.md#footprint-boxes)) |
| `snr_db`, `snr_baseline_db`, `snr_excursion_db` | footprint SNR overall, in the emitter's baseline band, outside every fish's band |
| `visible_fraction`, `visible` | fraction of footprint cells above the background; visibility label |
| `harmonic_box_t0_s`, `harmonic_box_t1_s`, `harmonic_box_f0_hz`, `harmonic_box_f1_hz` | footprint of the second harmonic in the band (NaN if outside) |

The footprint columns are present when `[footprint] enabled` (default).

## clicks.csv

`click_id`, `t_start_s`, `t_end_s`, `duration_s`, `kind` (`spike` or
`burst`), `origin` (e.g. `random`, `pair`), `amplitude`,
`amplitude_rel_noise_sd`, `electrodes` and `weights` (which electrodes, how
strongly). Hybrid scenes
have no synthetic clicks (the background's own are measured in the
background library).

## scene.json

`duration_s` and `samplerate_hz` at the top level; the resolved config;
`seed` and the random streams; versions and git SHA of gridtools; the
template library's manifest hash; the grid (electrode positions,
reference); levels and channels; every fish's parameters; the chirp
schedule; clicks; `background` (for hybrid scenes: library, segment id,
dataset, recording, `split_group`, offset, channels, scale, target SNR);
`real_fish`; the footprint settings; the render model (enough, with
`paths.npz` and `chirp_traces.npz`, to re-render any fish without the
template library); timings.

## Annotated spectrograms

`spectrogram_overview.png`: the whole scene, electrode-summed power in dB,
each fish's ground-truth track in its colour, every chirp as a vertical
segment from its baseline to baseline + excursion at its centre time, the
emitter's EODf at the chirp as a dot, clicks ticked along the top.
`spectrogram_zooms.png`: zooms at higher time resolution on
near-simultaneous chirps of different fish and chirps close in frequency to
another fish. `gridtools sim plot SCENE` redraws both (needs the `plot`
extra).
