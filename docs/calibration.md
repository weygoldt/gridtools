# Calibration

The defaults of the shipped scene config come from measurements of the 33
analysed tube-competition trials (2019/2020), made with
`gridtools.calibrate`:

```sh
python -m gridtools.calibrate RECORDINGS_ROOT OUT --workers 8 --seed 0 [--dates 2019-12-09-10_00 ...] [--no-cache]
```

`RECORDINGS_ROOT` holds the trial directories (`YYYY-MM-DD-HH_MM`, raw data,
wavetracker tracks, optional `chirpdetector_bboxes.csv` and
`temperatures.csv`) and a `trial_summary.csv` with the winner and loser
track ids of each trial. `OUT` gets `calibration.json` (every measured
quantity, the field fit, and proposed simulator parameters with
uncertainty, method and the current default), `report.md` with figures (needs
the `plot` extra) and a per-trial cache. Each trial gets its own random
stream, so results do not depend on the number of workers.

What is measured, per trial and pooled:

- **frequencies**: baseline EODf of the two fish (a running low quantile of
  the track, which ignores chirps and rises), ΔF, and drift as a
  semivariogram of a shared relative drift and an individual drift;
- **field**: the falloff exponent and regularisation of the dipole field,
  fitted to the real per-electrode power profiles (`sign_v`) by
  simulation, using only statistics that do not depend on electrode
  positions or channel order (sorted relative profiles), in units of the
  electrode spacing;
- **noise**: per-electrode noise floors, 50-Hz hum and its harmonics,
  fish fundamental amplitudes, clicks, from Welch spectra of seeded raw
  segments;
- **movement**: switches and dwell times of the strongest electrode, as a
  proxy that needs no positions;
- **chirps**: timing statistics of the v1 detections (intervals, bursts,
  responses). These describe v1-detected, v1-assigned chirps, not all
  chirps.

Units: "recording units" are the raw ADC units, volts at the ADC input
(electrode potential x 100).

`gridtools sim compare SCENES --calibration OUT/calibration.json` runs the
same estimators on simulated scenes and tabulates them against the real
values.

The movement model's defaults are plausible values for *A. leptorhynchus*
in a tank but are not fitted to tracked positions.
