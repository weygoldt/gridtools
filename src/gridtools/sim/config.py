"""Scene configuration: frozen dataclasses, loadable from TOML and JSON.

A :class:`SceneConfig` describes the *distributions* a scene is drawn from;
:func:`gridtools.sim.scene.simulate_scene` turns it plus a seed into one
scene. Every section is a frozen dataclass with documented defaults and
units, validated on construction. Ranges are ``(low, high)`` tuples (in TOML
and JSON: two-element arrays) from which one value is drawn per scene or per
fish, uniformly unless the field says log-uniform.

Loading is strict: unknown keys raise ``ValueError`` (a misspelt key would
otherwise silently fall back to its default). ``to_dict`` / ``from_dict``
round-trip exactly, through JSON as well.

The shipped example ``gridtools/configs/scenes/default.toml`` lists every
key with its default; :func:`default_config_path` returns its location.
"""

import dataclasses
import json
import math
import tomllib
import types
import typing
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from gridtools.sim.eod import DriftParams, HarmonicJitterParams, SharedDriftParams
from gridtools.sim.movement import MovementParams

__all__ = [
    "ChirpConfig",
    "ClickConfig",
    "FieldConfig",
    "GridConfig",
    "NoiseConfig",
    "PlotConfig",
    "PopulationConfig",
    "SceneConfig",
    "TrackConfig",
    "default_config_path",
]

FloatRange = tuple[float, float]
IntRange = tuple[int, int]

#: Template flags excluded by default (``gridtools.templates.DEFAULT_EXCLUDE`` of
#: library v2: demonstrable artefacts plus the atypical-EODf group).
DEFAULT_TEMPLATE_EXCLUDE: tuple[str, ...] = (
    "invalid",
    "has_nan",
    "duplicate",
    "truncated",
    "step_flag",
    "eodf_atypical",
)

DEFAULT_LIBRARY = "/mnt/data1/chirp_v2/templates/v2"


def _check_range(
    value: tuple[float, float],
    name: str,
    *,
    low: float | None = None,
    high: float | None = None,
    positive: bool = False,
) -> None:
    lo, hi = value
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        msg = f"{name} must be a finite range (low, high) with low <= high, got {value}"
        raise ValueError(msg)
    if positive and lo <= 0:
        msg = f"{name} must be > 0, got {value}"
        raise ValueError(msg)
    if low is not None and lo < low:
        msg = f"{name} must be >= {low}, got {value}"
        raise ValueError(msg)
    if high is not None and hi > high:
        msg = f"{name} must be <= {high}, got {value}"
        raise ValueError(msg)


def _check_number(
    value: float,
    name: str,
    *,
    low: float | None = None,
    high: float | None = None,
    strict_low: bool = False,
) -> None:
    if not math.isfinite(value):
        msg = f"{name} must be finite, got {value!r}"
        raise ValueError(msg)
    if low is not None and (value <= low if strict_low else value < low):
        relation = ">" if strict_low else ">="
        msg = f"{name} must be {relation} {low}, got {value!r}"
        raise ValueError(msg)
    if high is not None and value > high:
        msg = f"{name} must be <= {high}, got {value!r}"
        raise ValueError(msg)


@dataclasses.dataclass(frozen=True)
class GridConfig:
    """Electrode grid and the arena around it.

    Electrodes sit on a regular grid with the first electrode at the origin
    and channels in row-major order (as :meth:`gridtools.io.GridGeometry.regular`
    assumes for real recordings).

    Attributes
    ----------
    layouts : tuple of (int, int)
        Candidate layouts as (rows, columns); one is chosen uniformly per
        scene. Default: 3x3, 3x5 and 5x3, the grids of the 2019-2022 tube
        competitions (14 trials on 3x3, 19 on 15 electrodes). Give a single
        layout to fix it.
    spacing_m : float
        Distance between neighbouring electrodes in metres (default 0.5).
    arena_margin_m : float
        Margin of the rectangular arena (tank) beyond the outermost
        electrodes in metres (default 0.25). Fish move inside the arena.
    """

    layouts: tuple[tuple[int, int], ...] = ((3, 3), (3, 5), (5, 3))
    spacing_m: float = 0.5
    arena_margin_m: float = 0.25

    def __post_init__(self) -> None:
        if not self.layouts:
            msg = "grid.layouts must not be empty"
            raise ValueError(msg)
        for rows, cols in self.layouts:
            if rows < 1 or cols < 1:
                msg = f"grid layouts need >= 1 row and column, got {(rows, cols)}"
                raise ValueError(msg)
        _check_number(self.spacing_m, "grid.spacing_m", low=0.0, strict_low=True)
        _check_number(self.arena_margin_m, "grid.arena_margin_m", low=0.0)


@dataclasses.dataclass(frozen=True)
class PopulationConfig:
    """Number of fish, their baseline EOD frequencies and slow frequency changes.

    Baselines are drawn fish by fish. The first is uniform in
    ``baseline_hz``. Each further fish is, with probability
    ``close_pair_prob``, placed ``close_pair_df_hz`` (uniform) above or below
    a randomly chosen earlier fish (a *close pair*); otherwise it is uniform
    in ``baseline_hz``. There is no minimum frequency difference: close and
    even identical baselines are the hard cases for assignment and must be in
    the training data (v1 enforced >= 20 Hz).

    Frequency crossings come from slow ramps added to the drift: a fish of a
    close pair crosses its partner's baseline with probability
    ``crossing_prob`` (ramp of ``|dF| + ramp_overshoot_hz`` towards and past
    the partner); independently, any fish gets a random ramp with
    probability ``ramp_prob``. A ramp is a raised-cosine step lasting
    ``ramp_duration_s`` (uniform), starting at a uniform time.

    Attributes
    ----------
    n_fish : (int, int)
        Inclusive range of the number of fish (default 1-5, uniform).
    baseline_hz : (float, float)
        Baseline EODf range in Hz (default 600-1100, the range of
        *A. leptorhynchus*; to be replaced by the empirical distribution of
        the 33 trials).
    close_pair_prob : float
        Probability that a fish after the first forms a close pair
        (default 0.5).
    close_pair_df_hz : (float, float)
        Range of the baseline difference of a close pair in Hz (default
        0-20).
    crossing_prob : float
        Probability that the fish of a close pair ramps across its partner
        (default 0.5).
    ramp_overshoot_hz : (float, float)
        How far a crossing ramp goes past the partner in Hz (default 2-10).
    ramp_prob : float
        Probability of an additional random ramp per fish (default 0.1).
    ramp_df_hz : (float, float)
        Magnitude of a random ramp in Hz, sign random (default 5-30).
    ramp_duration_s : (float, float)
        Duration of any ramp in seconds (default 10-60).
    drift : DriftParams
        Individual slow drift (see :class:`gridtools.sim.eod.DriftParams`).
    shared_drift : SharedDriftParams
        Relative drift shared by all fish (temperature).
    harmonics : HarmonicJitterParams
        Per-fish variation of the EOD waveform.
    """

    n_fish: IntRange = (1, 5)
    baseline_hz: FloatRange = (600.0, 1100.0)
    close_pair_prob: float = 0.5
    close_pair_df_hz: FloatRange = (0.0, 20.0)
    crossing_prob: float = 0.5
    ramp_overshoot_hz: FloatRange = (2.0, 10.0)
    ramp_prob: float = 0.1
    ramp_df_hz: FloatRange = (5.0, 30.0)
    ramp_duration_s: FloatRange = (10.0, 60.0)
    drift: DriftParams = dataclasses.field(default_factory=DriftParams)
    shared_drift: SharedDriftParams = dataclasses.field(
        default_factory=SharedDriftParams
    )
    harmonics: HarmonicJitterParams = dataclasses.field(
        default_factory=HarmonicJitterParams
    )

    def __post_init__(self) -> None:
        lo, hi = self.n_fish
        if not (1 <= lo <= hi):
            msg = f"population.n_fish must satisfy 1 <= low <= high, got {self.n_fish}"
            raise ValueError(msg)
        _check_range(self.baseline_hz, "population.baseline_hz", positive=True)
        _check_number(
            self.close_pair_prob, "population.close_pair_prob", low=0.0, high=1.0
        )
        _check_range(self.close_pair_df_hz, "population.close_pair_df_hz", low=0.0)
        _check_number(self.crossing_prob, "population.crossing_prob", low=0.0, high=1.0)
        _check_range(self.ramp_overshoot_hz, "population.ramp_overshoot_hz", low=0.0)
        _check_number(self.ramp_prob, "population.ramp_prob", low=0.0, high=1.0)
        _check_range(self.ramp_df_hz, "population.ramp_df_hz", low=0.0)
        _check_range(self.ramp_duration_s, "population.ramp_duration_s", positive=True)


@dataclasses.dataclass(frozen=True)
class ChirpConfig:
    """Chirp templates, timing and amplitude modulation.

    Timing. Each fish has a spontaneous chirp rate (uniform in ``rate_hz``).
    Spontaneous chirps come in bursts: burst onsets are a Poisson process
    with rate ``rate / burst_size_mean``; a burst has a geometric number of
    chirps (mean ``burst_size_mean``), each starting a silent gap
    ``burst_gap_s`` (uniform) after the previous one's support ends.
    Responses: after every spontaneous chirp, with probability
    ``response_prob`` another fish (uniform among the others) chirps
    ``response_delay_s`` later (uniform, centre to centre), so
    near-simultaneous chirps of different fish occur. Responses do not
    trigger further responses.

    Limits (MSc thesis): between the supports of two chirps of one fish
    there are at least ``min_interval_s`` (measured on the drawn templates'
    supports, so long chirps take more room; later chirps that violate it
    are dropped), and a fish has at most ``floor(max_rate_hz * duration)``
    chirps. A response that would violate either limit of the responding
    fish is dropped. Chirp centres lie in ``[0, duration]``; chirps near the
    edges are cut by the scene boundary and flagged ``truncated``.

    Templates are real chirps from the curated library (decision 5),
    baseline-relative, added to the fish's baseline frequency, drawn per
    fish with the library's cycler. With ``equalize`` (decision 10), the
    pool is binned on excursion and width at 10 % in ``equalize_bins``
    equal-width bins; every non-empty bin is equally likely, and templates
    within a bin come from successive permutations.

    Amplitude modulation (``amplitude_mode``):

    - ``"frequency"`` (default, the author's method): the EOD amplitude
      during a chirp is ``1 - dip * clip(df, 0) / max(df)`` with
      ``dip`` uniform in ``dip`` (default 0-0.8: amplitude between 100 % and
      20 %);
    - ``"measured"``: the template's measured amplitude trace; only
      templates with a usable amplitude trace (``amp_usable``) are drawn.

    Attributes
    ----------
    library : str
        Path of the template library (default ``/mnt/data1/chirp_v2/templates/v2``).
    families : tuple of str
        Template families to draw from (``"small"``, ``"big"``; default both).
    exclude : tuple of str
        Library flags whose templates are excluded (default: library v2's
        default exclusion list).
    equalize : bool
        Equalise the (excursion x width) histogram (default True).
    equalize_bins : (int, int)
        Equalisation bins on excursion and on width at 10 % (default 10 x 10).
    rate_hz : (float, float)
        Range of the spontaneous chirp rate per fish in chirps/s
        (default 0.05-0.5).
    max_rate_hz : float
        Upper bound of a fish's mean chirp rate over the scene (default 1).
    min_interval_s : float
        Minimum silent interval between the supports of two chirps of one
        fish in s (default 0.2).
    burst_size_mean : float
        Mean number of chirps per burst, >= 1 (default 2).
    burst_gap_s : (float, float)
        Silent interval between successive chirps of a burst in s (default
        0.2-0.5).
    response_prob : float
        Probability that a spontaneous chirp triggers a response by another
        fish (default 0.3).
    response_delay_s : (float, float)
        Delay of a response after its trigger in s (default 0-0.5).
    amplitude_mode : {"frequency", "measured"}
        Source of the chirp amplitude modulation (default ``"frequency"``).
    dip : (float, float)
        Range of the amplitude dip in frequency mode (default 0-0.8).
    """

    library: str = DEFAULT_LIBRARY
    families: tuple[str, ...] = ("small", "big")
    exclude: tuple[str, ...] = DEFAULT_TEMPLATE_EXCLUDE
    equalize: bool = True
    equalize_bins: tuple[int, int] = (10, 10)
    rate_hz: FloatRange = (0.05, 0.5)
    max_rate_hz: float = 1.0
    min_interval_s: float = 0.2
    burst_size_mean: float = 2.0
    burst_gap_s: FloatRange = (0.2, 0.5)
    response_prob: float = 0.3
    response_delay_s: FloatRange = (0.0, 0.5)
    amplitude_mode: Literal["frequency", "measured"] = "frequency"
    dip: FloatRange = (0.0, 0.8)

    def __post_init__(self) -> None:
        if not self.library:
            msg = "chirps.library must not be empty"
            raise ValueError(msg)
        for family in self.families:
            if family not in ("small", "big"):
                msg = f"chirps.families must be 'small' or 'big', got {family!r}"
                raise ValueError(msg)
        if not self.families:
            msg = "chirps.families must not be empty"
            raise ValueError(msg)
        if min(self.equalize_bins) < 1:
            msg = f"chirps.equalize_bins must be >= 1, got {self.equalize_bins}"
            raise ValueError(msg)
        _check_range(self.rate_hz, "chirps.rate_hz", low=0.0)
        _check_number(self.max_rate_hz, "chirps.max_rate_hz", low=0.0)
        _check_number(self.min_interval_s, "chirps.min_interval_s", low=0.0)
        _check_number(self.burst_size_mean, "chirps.burst_size_mean", low=1.0)
        _check_range(self.burst_gap_s, "chirps.burst_gap_s", low=0.0)
        _check_number(self.response_prob, "chirps.response_prob", low=0.0, high=1.0)
        _check_range(self.response_delay_s, "chirps.response_delay_s", low=0.0)
        if self.amplitude_mode not in ("frequency", "measured"):
            msg = (
                "chirps.amplitude_mode must be 'frequency' or 'measured', "
                f"got {self.amplitude_mode!r}"
            )
            raise ValueError(msg)
        _check_range(self.dip, "chirps.dip", low=0.0, high=1.0)


@dataclasses.dataclass(frozen=True)
class FieldConfig:
    """Dipole field and per-fish EOD amplitude and depth.

    See :mod:`gridtools.sim.field` for the field model.

    Attributes
    ----------
    amplitude : (float, float)
        Range of the EOD amplitude per fish, drawn log-uniformly (default
        0.5-2, a factor of 4). Unit: potential on the body axis at
        ``reference_distance_m`` (nominally mV); this sets the unit of the
        whole recording.
    fish_depth_m : (float, float)
        Range of the fish depth below the surface in m (default 0.05-0.15).
    electrode_depth_m : float
        Electrode depth in m (default 0.15).
    min_distance_m : float
        Regularisation distance of the point dipole in m (default 0.1).
    distance_exponent : float
        Falloff exponent (default 2, a physical dipole).
    reference_distance_m : float
        Axis distance with gain 1 in m (default 0.1).
    """

    amplitude: FloatRange = (0.5, 2.0)
    fish_depth_m: FloatRange = (0.05, 0.15)
    electrode_depth_m: float = 0.15
    min_distance_m: float = 0.1
    distance_exponent: float = 2.0
    reference_distance_m: float = 0.1

    def __post_init__(self) -> None:
        _check_range(self.amplitude, "field.amplitude", positive=True)
        _check_range(self.fish_depth_m, "field.fish_depth_m", low=0.0)
        _check_number(self.electrode_depth_m, "field.electrode_depth_m", low=0.0)
        _check_number(self.min_distance_m, "field.min_distance_m", low=0.0)
        _check_number(
            self.distance_exponent, "field.distance_exponent", low=0.0, strict_low=True
        )
        _check_number(
            self.reference_distance_m,
            "field.reference_distance_m",
            low=0.0,
            strict_low=True,
        )


@dataclasses.dataclass(frozen=True)
class NoiseConfig:
    """Electrode noise, power-line hum, channel gains and faults.

    Levels are relative to the reference amplitude ``a_ref``: the median EOD
    amplitude of the scene's fish times the dipole gain on the body axis at
    0.5 m horizontal distance (at the median fish depth). The coloured noise
    SD is ``a_ref * 10**(-snr_db / 20)`` with ``snr_db`` uniform in
    ``snr_db``, so scenes span clean to noisy. This is a broadband ratio of
    amplitudes; in a spectrogram with 4096-point windows at 20 kHz a fish at
    0.5 m stands roughly ``snr_db + 28`` dB above the noise floor.

    Attributes
    ----------
    snr_db : (float, float)
        Range of the broadband SNR in dB (default -15 to 20).
    exponent : (float, float)
        Range of the spectral exponent of the noise (default 0.5-1.5;
        0 = white, 1 = pink).
    f_min_hz : float
        Corner below which the noise spectrum is flat (default 1 Hz).
    hum_db : (float, float)
        Range of the hum fundamental amplitude in dB re ``a_ref`` (default
        -20 to 20).
    hum_line_freq_hz : float
        Mains frequency (default 50).
    hum_n_harmonics : int
        Hum components including the fundamental (default 5).
    gain_mismatch_log_sd : float
        SD of the log channel gain (default 0.1).
    dead_prob, noisy_prob : float
        Probability of a dead / noisy electrode (defaults 0.05).
    dead_gain : float
        Channel gain of a dead electrode (default 1e-3).
    noisy_factor : float
        Noise SD factor of a noisy electrode (default 5).
    clip_limit : float or None
        Symmetric clipping level in recording units; None (default)
        disables clipping. Clipping breaks the linearity
        ``recording = sum(render_fish) + background``.
    """

    snr_db: FloatRange = (-15.0, 20.0)
    exponent: FloatRange = (0.5, 1.5)
    f_min_hz: float = 1.0
    hum_db: FloatRange = (-20.0, 20.0)
    hum_line_freq_hz: float = 50.0
    hum_n_harmonics: int = 5
    gain_mismatch_log_sd: float = 0.1
    dead_prob: float = 0.05
    noisy_prob: float = 0.05
    dead_gain: float = 1e-3
    noisy_factor: float = 5.0
    clip_limit: float | None = None

    def __post_init__(self) -> None:
        _check_range(self.snr_db, "noise.snr_db")
        _check_range(self.exponent, "noise.exponent", low=0.0)
        _check_number(self.f_min_hz, "noise.f_min_hz", low=0.0, strict_low=True)
        _check_range(self.hum_db, "noise.hum_db")
        _check_number(
            self.hum_line_freq_hz, "noise.hum_line_freq_hz", low=0.0, strict_low=True
        )
        if self.hum_n_harmonics < 1:
            msg = f"noise.hum_n_harmonics must be >= 1, got {self.hum_n_harmonics}"
            raise ValueError(msg)
        _check_number(self.gain_mismatch_log_sd, "noise.gain_mismatch_log_sd", low=0.0)
        _check_number(self.dead_prob, "noise.dead_prob", low=0.0, high=1.0)
        _check_number(self.noisy_prob, "noise.noisy_prob", low=0.0, high=1.0)
        _check_number(self.dead_gain, "noise.dead_gain", low=0.0)
        _check_number(self.noisy_factor, "noise.noisy_factor", low=0.0)
        if self.clip_limit is not None:
            _check_number(self.clip_limit, "noise.clip_limit", low=0.0, strict_low=True)


@dataclasses.dataclass(frozen=True)
class ClickConfig:
    """Rare impulsive artefacts (clicks), added after the channel gains.

    In the real recordings clicks were the main source of false chirp
    detections: broadband events of a fraction of a millisecond to a few
    milliseconds, up to 10^3-10^4 times the noise SD, on one or a few
    electrodes (see audian's click removal). Clicks are a Poisson process
    with a per-scene rate (uniform in ``rate_hz``). Each click has a
    log-uniform duration and peak amplitude (in units of the scene's noise
    SD), a random polarity, and a shape: a Hann-windowed white-noise burst
    or a spike (instant rise, exponential decay). It hits a geometric number
    of electrodes (mean ``electrodes_mean``); electrodes after the first get
    a relative amplitude uniform in ``spread``.

    Attributes
    ----------
    rate_hz : (float, float)
        Range of the click rate in clicks/s (default 0-0.1).
    duration_s : (float, float)
        Range of click durations in s, log-uniform (default 0.1-5 ms).
    amplitude_rel : (float, float)
        Range of the peak amplitude in units of the noise SD, log-uniform
        (default 10-10^4).
    electrodes_mean : float
        Mean number of electrodes per click, >= 1 (default 1.5).
    spread : (float, float)
        Relative amplitude on secondary electrodes (default 0.1-1).
    burst_prob : float
        Probability of the burst shape (else spike; default 0.5).
    """

    rate_hz: FloatRange = (0.0, 0.1)
    duration_s: FloatRange = (1e-4, 5e-3)
    amplitude_rel: FloatRange = (10.0, 1e4)
    electrodes_mean: float = 1.5
    spread: FloatRange = (0.1, 1.0)
    burst_prob: float = 0.5

    def __post_init__(self) -> None:
        _check_range(self.rate_hz, "clicks.rate_hz", low=0.0)
        _check_range(self.duration_s, "clicks.duration_s", positive=True)
        _check_range(self.amplitude_rel, "clicks.amplitude_rel", positive=True)
        _check_number(self.electrodes_mean, "clicks.electrodes_mean", low=1.0)
        _check_range(self.spread, "clicks.spread", low=0.0)
        _check_number(self.burst_prob, "clicks.burst_prob", low=0.0, high=1.0)


@dataclasses.dataclass(frozen=True)
class TrackConfig:
    """Ground-truth tracks in wavetracker layout.

    Attributes
    ----------
    step_s : float
        Frame interval in s (default 0.1229, the frame grid of the 2020
        production tracks, about 8.1 Hz). Frames start at 0 s.
    power_nfft : int
        Window length used to express ``sign_v`` as a power spectral
        density (default 16384; see :func:`gridtools.sim.scene.fish_power_db`).
    """

    step_s: float = 0.1229
    power_nfft: int = 16384

    def __post_init__(self) -> None:
        _check_number(self.step_s, "tracks.step_s", low=0.0, strict_low=True)
        if self.power_nfft < 2:
            msg = f"tracks.power_nfft must be >= 2, got {self.power_nfft}"
            raise ValueError(msg)


@dataclasses.dataclass(frozen=True)
class PlotConfig:
    """Annotated spectrograms written next to each scene.

    Attributes
    ----------
    enabled : bool
        Write the PNGs in :func:`gridtools.sim.scene.write_scene` (default
        True; needs the ``plot`` extra).
    f_min_hz, f_max_hz : float
        Frequency range shown (default 500-1500 Hz).
    nfft : int
        FFT length of the overview spectrogram (default 4096).
    hop : int
        Hop of the overview spectrogram in samples (default 512).
    n_zooms : int
        Number of zoomed panels (default 6).
    zoom_s : float
        Duration of a zoomed panel in s (default 5).
    zoom_nfft : int
        FFT length of the zoomed panels (default 512: 26 ms at 20 kHz, so a
        chirp's excursion is visible; fish closer than ~40 Hz merge, their
        tracks are drawn on top).
    zoom_hop : int
        Hop of the zoomed panels in samples (default 32).
    dpi : int
        Resolution of the PNGs (default 110).
    """

    enabled: bool = True
    f_min_hz: float = 500.0
    f_max_hz: float = 1500.0
    nfft: int = 4096
    hop: int = 512
    n_zooms: int = 6
    zoom_s: float = 5.0
    zoom_nfft: int = 512
    zoom_hop: int = 32
    dpi: int = 110

    def __post_init__(self) -> None:
        _check_number(self.f_min_hz, "plot.f_min_hz", low=0.0)
        if not self.f_max_hz > self.f_min_hz:
            msg = "plot.f_max_hz must be > plot.f_min_hz"
            raise ValueError(msg)
        for name in ("nfft", "hop", "zoom_nfft", "zoom_hop", "dpi"):
            if getattr(self, name) < 1:
                msg = f"plot.{name} must be >= 1"
                raise ValueError(msg)
        if self.n_zooms < 0:
            msg = "plot.n_zooms must be >= 0"
            raise ValueError(msg)
        _check_number(self.zoom_s, "plot.zoom_s", low=0.0, strict_low=True)


@dataclasses.dataclass(frozen=True)
class SceneConfig:
    """Everything a scene is drawn from.

    Attributes
    ----------
    name : str
        Config name; used for the default output directory (default
        ``"default"``).
    duration_s : float
        Scene duration in s (default 120).
    samplerate_hz : float
        Sampling rate in Hz (default 20000, as the real grids).
    grid, population, chirps, movement, field, noise, clicks, tracks, plot
        Sections; see their classes.
    """

    name: str = "default"
    duration_s: float = 120.0
    samplerate_hz: float = 20_000.0
    grid: GridConfig = dataclasses.field(default_factory=GridConfig)
    population: PopulationConfig = dataclasses.field(default_factory=PopulationConfig)
    chirps: ChirpConfig = dataclasses.field(default_factory=ChirpConfig)
    movement: MovementParams = dataclasses.field(default_factory=MovementParams)
    field: FieldConfig = dataclasses.field(default_factory=FieldConfig)
    noise: NoiseConfig = dataclasses.field(default_factory=NoiseConfig)
    clicks: ClickConfig = dataclasses.field(default_factory=ClickConfig)
    tracks: TrackConfig = dataclasses.field(default_factory=TrackConfig)
    plot: PlotConfig = dataclasses.field(default_factory=PlotConfig)

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            msg = f"name must be a non-empty string without '/', got {self.name!r}"
            raise ValueError(msg)
        _check_number(self.duration_s, "duration_s", low=0.0, strict_low=True)
        _check_number(self.samplerate_hz, "samplerate_hz", low=0.0, strict_low=True)
        if self.n_samples < 2:
            msg = "the scene must have at least 2 samples"
            raise ValueError(msg)
        f_top = self.population.baseline_hz[1] * 2.0
        if f_top >= self.samplerate_hz / 2.0:
            msg = (
                f"samplerate_hz {self.samplerate_hz} is too low for baselines up to "
                f"{self.population.baseline_hz[1]} Hz and their harmonics"
            )
            raise ValueError(msg)

    @property
    def n_samples(self) -> int:
        """Number of samples per channel, ``round(duration_s * samplerate_hz)``."""
        return round(self.duration_s * self.samplerate_hz)

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict (tuples stay tuples; JSON writes them as lists)."""
        return dataclasses.asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        """JSON text of :meth:`to_dict` (``kwargs`` go to :func:`json.dumps`)."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneConfig":
        """Build and validate a config from nested dicts; unknown keys raise."""
        return _build(cls, data, "")

    @classmethod
    def from_json(cls, text: str) -> "SceneConfig":
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_toml(cls, path: str | Path) -> "SceneConfig":
        """Load a TOML file; ``name`` defaults to the file stem."""
        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        data.setdefault("name", path.stem)
        return cls.from_dict(data)

    def replace(self, **changes: Any) -> "SceneConfig":
        """Return a copy with top-level fields replaced (and validated)."""
        return dataclasses.replace(self, **changes)


def default_config_path() -> Path:
    """Location of the shipped example config ``configs/scenes/default.toml``."""
    resource = resources.files("gridtools") / "configs" / "scenes" / "default.toml"
    return Path(str(resource))


# -- strict conversion from nested dicts ------------------------------------


def _build[T](cls: type[T], data: Any, where: str) -> T:
    if not isinstance(data, dict):
        msg = f"{where or 'config'} must be a table/object, got {type(data).__name__}"
        raise TypeError(msg)
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - names)
    if unknown:
        msg = (
            f"unknown key(s) {unknown} in {where or 'config'}; allowed: {sorted(names)}"
        )
        raise ValueError(msg)
    kwargs = {
        key: _convert(hints[key], value, f"{where}.{key}" if where else key)
        for key, value in data.items()
    }
    return cls(**kwargs)


def _convert(annotation: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return _build(annotation, value, where)
    if origin in (types.UnionType, typing.Union):
        if value is None and type(None) in args:
            return None
        errors = []
        for option in args:
            if option is type(None):
                continue
            try:
                return _convert(option, value, where)
            except (TypeError, ValueError) as err:
                errors.append(str(err))
        msg = f"{where}: {value!r} matches none of {args}: {errors}"
        raise TypeError(msg)
    if origin is Literal:
        if value not in args:
            msg = f"{where} must be one of {args}, got {value!r}"
            raise ValueError(msg)
        return value
    if origin is tuple:
        if not isinstance(value, list | tuple):
            msg = f"{where} must be an array, got {value!r}"
            raise TypeError(msg)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _convert(args[0], v, f"{where}[{i}]") for i, v in enumerate(value)
            )
        if len(value) != len(args):
            msg = f"{where} must have {len(args)} elements, got {len(value)}"
            raise ValueError(msg)
        return tuple(
            _convert(a, v, f"{where}[{i}]")
            for i, (a, v) in enumerate(zip(args, value, strict=True))
        )
    if annotation is bool:
        if not isinstance(value, bool):
            msg = f"{where} must be true or false, got {value!r}"
            raise TypeError(msg)
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"{where} must be an integer, got {value!r}"
            raise TypeError(msg)
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            msg = f"{where} must be a number, got {value!r}"
            raise TypeError(msg)
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            msg = f"{where} must be a string, got {value!r}"
            raise TypeError(msg)
        return value
    msg = f"{where}: unsupported config type {annotation!r}"
    raise TypeError(msg)
