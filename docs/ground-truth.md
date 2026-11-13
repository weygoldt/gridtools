# Ground truth

A simulated scene knows exactly what every fish did. This page defines what
"the truth" of a chirp is: its frequency trajectory, its box in a
spectrogram, its visibility, and its parameters.

## The instantaneous frequency

Each fish's EOD is a harmonic series locked to one phase,

```text
theta(t) = theta_0 + 2 pi * integral_0^t f(t') dt'
e(t)     = a(t) * sum_k A_k cos(k theta(t) + phi_k)
```

with the instantaneous frequency

```text
f(t) = (baseline + individual drift(t) + ramps(t)) * (1 + shared drift(t)) + sum of chirp excursions(t)
```

integrated with the trapezoid rule at the scene's sampling rate. Because
frequency enters only through the phase, any excursion leaves the waveform
continuous. The ground-truth tracks hold the chirp-free part of f(t)
(`base_frequency`), and `chirps.csv` holds the extent of each chirp's
rendered trace.

## Per-period frequency: the chirp parameters

Chirp parameters are defined on the noise-free frequency the simulator
integrated, one value per EOD period, never on a spectrogram (this is the
ground truth of chirpdetector's parameter decoder,
`chirpdetector.params.truth`):

- the cycle boundaries t_m are where the unwrapped phase crosses 2πm, with
  only this chirp in place (its fish's other chirps left out, so that a
  neighbour in a burst cannot lend it its peak or undershoot);
- period k lasts T_k; 1/T_k is the time average of f(t) over the period;
- its excursion is 1/T_k minus the period mean of the chirp-free baseline;
- **height** is the largest excursion; **FWHM** and the **10 % width** are
  the times between the crossings of 50 % and 10 % of the height that
  bracket the peak (interpolated between period midpoints); **undershoot**
  is the deepest excursion below zero after the peak.

The render model of a written scene gives everything needed:

```python
import numpy as np
import pandas as pd
from gridtools.sim.scene import load_scene_model

scene = "scenes/000000"
model = load_scene_model(scene)  # scene.json, paths.npz, chirp_traces.npz
chirps = pd.read_csv(f"{scene}/chirps.csv")
chirp = chirps.iloc[3]
fish, fs = int(chirp["emitter_id"]), model.samplerate_hz
start = int((chirp["t_support_start_s"] - 0.05) * fs)
stop = int((chirp["t_support_stop_s"] + 0.05) * fs)
others = [
    c.chirp_id for c in model.fish[fish].chirps if c.chirp_id != chirp["chirp_id"]
]
cycles = model.phase(fish, start, stop, chirps=others, phase="natural") / (2 * np.pi)
m = np.arange(np.ceil(cycles[0]), np.floor(cycles[-1]) + 1)
bounds = (start + np.interp(m, cycles, np.arange(cycles.size))) / fs  # 2*pi*m crossings
period_hz = 1 / np.diff(bounds)
mid = (bounds[:-1] + bounds[1:]) / 2
excursion = period_hz - model.base_frequency_at(fish, mid)
print(f"height {excursion.max():.1f} Hz")  # 276.8 Hz; chirps.csv excursion_hz 277.2
```

(This sketch takes the baseline at the period midpoint; chirpdetector's
`params.truth` averages it over the period and interpolates the widths.)
`chirps` in `phase`, `frequency` and `render_fish` lists the chirps to
*leave out* (`"all"` and `"none"` are the other choices).

## Rendering with and without a chirp

Any fish can be re-rendered for any span, bit-identical to the same
samples of the full rendering (`render_fish(scene, fish, t0, t1,
chirps=...)`, `SceneModel.render`). Leaving a chirp out changes the phase
after it by the chirp's phase advance (about 0.7 cycles for a 64-Hz, 10-ms
chirp), so there are two definitions (`phase=`):

- `"aligned"` (default): the excursion and dip are removed and the
  fractional part of the phase advance is spread over the chirp's support,
  so after the support the two renderings are identical. "With minus
  without" is exactly zero outside the support, in any representation. This
  defines the footprints.
- `"natural"`: the fish that never chirped; its phase lags the full
  rendering after the chirp. This defines the per-period truth.

## Footprint boxes

A chirp's box is where it changes the spectrogram:

1. render the emitter alone over the chirp's support plus the analysis
   window and a margin, with and without the chirp (`"aligned"`);
2. compute both spectrograms with the front end of choice and take
   D = |P_with - P_without| per cell: the energy along the excursion and the
   power the chirp removes from the baseline band, smeared by the window;
3. keep the cells where D is within 30 dB (`footprint.threshold_db`) of
   its maximum and that are connected to the chirp's trajectory;
4. the box spans these cells, its edges at cell edges.

The scene stores footprints (`box_*` in `chirps.csv`) for the reference
front end in `[footprint]` (Hann window 0.2048 s, hop 5 ms, 500-1500 Hz,
electrodes summed in power). chirpdetector recomputes them with its own
front end where it differs. Any front end can be passed in:

```python
from gridtools.sim.footprint import HannSpectrogram, chirp_footprint

footprint = chirp_footprint(
    "scenes/000000", 0, HannSpectrogram(window_s=0.0256, hop_s=0.0016)
)
footprint.box  # (t0, t1, f0, f1) in s and Hz
```

The second harmonic's footprint is returned as well when it lies in the
band (`harmonic_box_*`); chirpdetector makes it an ignore region.

## Visibility

The footprint's energy is compared with a background spectrogram B over the
same span and electrodes, by default everything except the emitter (other
fish, noise, hum, clicks, a real background): `snr_db` over the whole
footprint, `snr_baseline_db` over its cells in the emitter's baseline band
(the power the chirp takes out of the band), `snr_excursion_db` over its
cells outside every fish's band (where only noise competes). A chirp is
`visible` when either part reaches `visible_snr_db` (0 dB): either cue
reveals a chirp, and a single sum over the whole footprint is dominated by
strong baseline cells, so that in a band crowded with fish a plainly
visible excursion would look invisible. `visible_fraction` is the fraction
of footprint cells where D > B.
