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
from gridtools.templates.library import DEFAULT_EXCLUDE as _LIBRARY_EXCLUDE

__all__ = [
    "ChirpConfig",
    "ClickConfig",
    "FieldConfig",
    "FootprintConfig",
    "GridConfig",
    "MovementConfig",
    "NoiseConfig",
    "PlotConfig",
    "PopulationConfig",
    "SceneConfig",
    "TrackConfig",
    "default_config_path",
]

FloatRange = tuple[float, float]
IntRange = tuple[int, int]

#: Template flags excluded by default (the library's ``DEFAULT_EXCLUDE``:
#: demonstrable artefacts and events shorter than 5 ms).
DEFAULT_TEMPLATE_EXCLUDE: tuple[str, ...] = tuple(_LIBRARY_EXCLUDE)

DEFAULT_LIBRARY = "/mnt/data1/chirp_v2/templates/v3"


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
    """Electrodes and the arena around them.

    Per scene, with probability ``real_layout_prob`` one of the real
    ``layouts`` is used (a regular grid, first electrode at the origin,
    channels in row-major order as :meth:`gridtools.io.GridGeometry.regular`
    assumes). Otherwise the electrode count is drawn from ``n_electrodes``
    (inclusive; with probability ``small_prob`` uniformly from the small
    counts up to ``small_max``, else uniformly from the larger ones), so
    single-electrode (chirp-chamber-like) and 2-4-electrode scenes are well
    represented, and arranged as:

    - 1: a single electrode; 2: a pair;
    - 3-4: a line, or a 2 x 2 grid (with one position missing for 3);
    - >= 5: an r x c rectangle if the count factorises with an aspect ratio
      up to ``max_aspect`` (orientation random), else the smallest
      near-square grid with randomly chosen positions missing.

    These drawn layouts get Gaussian position jitter (SD ``jitter_rel``
    spacings) and, with probability ``rotation_prob``, a random rotation
    about the layout's centre. The real layouts stay regular.

    The arena is the electrodes' bounding box plus ``arena_margin_rel``
    spacings, and at least one spacing wide in each direction (single
    electrodes and lines).

    Attributes
    ----------
    layouts : tuple of (int, int)
        Real layouts as (rows, columns) (default 3x3, 3x5, 5x3: the 2019-2020
        tube competitions).
    real_layout_prob : float
        Probability of a real layout (default 0.3).
    n_electrodes : (int, int)
        Inclusive range of drawn electrode counts (default 1-16).
    small_max : int
        Largest "small" count (default 4).
    small_prob : float
        Probability of a small count among drawn layouts (default 0.4).
    max_aspect : float
        Largest aspect ratio of a rectangular layout (default 3).
    jitter_rel : float
        SD of the position jitter of drawn layouts in spacings (default 0.05).
    rotation_prob : float
        Probability of a random rotation of drawn layouts (default 0.5).
    spacing_m : (float, float)
        Range of the electrode spacing in metres, drawn uniformly per scene
        (default 0.15-0.5; field-like grids up to ~1 m). The real spacing is
        unknown (the author: not the 0.5 m in ``fishgrid.cfg``, probably
        different in 2019 and 2020). The field is calibrated in units of the
        spacing, and the drawn value is written to ``fishgrid.cfg`` and
        ``scene.json``.
    arena_margin_rel : float
        Margin of the arena beyond the outermost electrodes, in spacings
        (default 0, the calibrated position prior: fish stay over the grid;
        bootstrap 90 % interval 0-0.25).
    reference_prob : float
        Probability of an in-tank reference electrode (default 0.5; the
        real reference is unknown and may be in the tank). Its potential
        from every fish is subtracted from all channels; otherwise the
        reference is at infinity.
    reference_margin_rel : float
        The in-tank reference lies uniformly in the arena extended by this
        many spacings on every side (default 1).
    """

    layouts: tuple[tuple[int, int], ...] = ((3, 3), (3, 5), (5, 3))
    real_layout_prob: float = 0.3
    n_electrodes: IntRange = (1, 16)
    small_max: int = 4
    small_prob: float = 0.4
    max_aspect: float = 3.0
    jitter_rel: float = 0.05
    rotation_prob: float = 0.5
    spacing_m: FloatRange = (0.15, 0.5)
    arena_margin_rel: float = 0.0
    reference_prob: float = 0.5
    reference_margin_rel: float = 1.0

    def __post_init__(self) -> None:
        if not self.layouts:
            msg = "grid.layouts must not be empty"
            raise ValueError(msg)
        for rows, cols in self.layouts:
            if rows < 1 or cols < 1:
                msg = f"grid layouts need >= 1 row and column, got {(rows, cols)}"
                raise ValueError(msg)
        _check_number(self.real_layout_prob, "grid.real_layout_prob", low=0.0, high=1.0)
        lo, hi = self.n_electrodes
        if not 1 <= lo <= hi:
            msg = f"grid.n_electrodes must satisfy 1 <= low <= high, got {self.n_electrodes}"
            raise ValueError(msg)
        if self.small_max < 1:
            msg = "grid.small_max must be >= 1"
            raise ValueError(msg)
        _check_number(self.small_prob, "grid.small_prob", low=0.0, high=1.0)
        _check_number(self.max_aspect, "grid.max_aspect", low=1.0)
        _check_number(self.jitter_rel, "grid.jitter_rel", low=0.0)
        _check_number(self.rotation_prob, "grid.rotation_prob", low=0.0, high=1.0)
        _check_range(self.spacing_m, "grid.spacing_m", positive=True)
        _check_number(self.arena_margin_rel, "grid.arena_margin_rel", low=0.0)
        _check_number(self.reference_prob, "grid.reference_prob", low=0.0, high=1.0)
        _check_number(self.reference_margin_rel, "grid.reference_margin_rel", low=0.0)


@dataclasses.dataclass(frozen=True)
class PopulationConfig:
    """Number of fish, their baseline EOD frequencies and slow frequency changes.

    Baselines are drawn fish by fish from a mixture (calibration v1): with
    probability ``baseline_uniform_prob`` uniform in ``baseline_uniform_hz``
    (coverage of other data sets), otherwise from the female or male normal
    distribution of the tube competitions (``male_prob``), clipped to
    ``baseline_uniform_hz``. Each fish after the first is instead, with
    probability ``close_pair_prob``, placed ``close_pair_df_hz`` (uniform)
    above or below a randomly chosen earlier fish (a *close pair*). Close
    pairs and crossings are deliberately far more common than in the real
    trials (2.4 % of the time below 10 Hz, 2 crossings in 184 h): they are the
    hard cases the models must learn; evaluate on the real distribution.

    Frequency crossings come from slow ramps added to the drift: a fish of a
    close pair crosses its partner's baseline with probability
    ``crossing_prob`` (ramp of ``|dF| + ramp_overshoot_hz`` towards and past
    the partner); independently, any fish gets a random ramp with
    probability ``ramp_prob``. A ramp is a raised-cosine step lasting
    ``ramp_duration_s`` (uniform), starting at a uniform time.

    Drift (calibration v1): each fish's individual drift has SD
    ``drift.sd_hz`` and time-scale ``drift.timescale_s`` times log-normal
    factors with log SDs ``drift_sd_log_sd`` and ``drift_timescale_log_sd``
    (the measured spread across fish: SD 0.50-1.61 Hz, tau 7.7-33 s,
    5-95 %). The shared drift is a linear trend (relative rate per hour
    drawn from N(``shared_trend_per_h``), measured -0.94 %/h, 5-95 % -1.4 to
    -0.47 %/h, the water cooling during a trial) centred on the scene's
    midpoint, plus a small stationary part ``shared_drift`` (not
    calibrated: the common wander beyond the trend is small and slow).
    Within 120 s both are negligible.

    Attributes
    ----------
    n_fish : (int, int)
        Inclusive range of the number of fish (default 1-5, uniform).
    baseline_uniform_hz : (float, float)
        Range of the uniform component and clipping range of all baselines
        in Hz (default 550-1100).
    baseline_uniform_prob : float
        Weight of the uniform component (default 0.3).
    female_hz, male_hz : (float, float)
        Mean and SD of the female and male baselines in Hz (default
        664 +- 35 and 803 +- 54, at 23-26 degC).
    male_prob : float
        Probability that a non-uniform fish is male (default 0.5).
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
        Median individual drift (default SD 1.0 Hz, tau 11.3 s, control
        rate 1 Hz).
    drift_sd_log_sd, drift_timescale_log_sd : float
        Per-fish log-normal spread of the drift SD and time-scale (defaults
        0.35 and 0.4).
    shared_trend_per_h : (float, float)
        Mean and SD of the relative shared trend per hour (default
        -0.0094 +- 0.0028).
    shared_drift : SharedDriftParams
        Stationary shared drift (default SD 5e-4, tau 600 s).
    harmonics : HarmonicJitterParams
        Per-fish variation of the EOD waveform.
    """

    n_fish: IntRange = (1, 5)
    baseline_uniform_hz: FloatRange = (550.0, 1100.0)
    baseline_uniform_prob: float = 0.3
    female_hz: FloatRange = (664.3, 34.9)
    male_hz: FloatRange = (803.2, 54.2)
    male_prob: float = 0.5
    close_pair_prob: float = 0.5
    close_pair_df_hz: FloatRange = (0.0, 20.0)
    crossing_prob: float = 0.5
    ramp_overshoot_hz: FloatRange = (2.0, 10.0)
    ramp_prob: float = 0.1
    ramp_df_hz: FloatRange = (5.0, 30.0)
    ramp_duration_s: FloatRange = (10.0, 60.0)
    drift: DriftParams = dataclasses.field(
        default_factory=lambda: DriftParams(
            sd_hz=1.0, timescale_s=11.3, control_rate_hz=1.0
        )
    )
    drift_sd_log_sd: float = 0.35
    drift_timescale_log_sd: float = 0.4
    shared_trend_per_h: FloatRange = (-0.0094, 0.0028)
    shared_drift: SharedDriftParams = dataclasses.field(
        default_factory=lambda: SharedDriftParams(
            sd_rel=5e-4, timescale_s=600.0, control_rate_hz=0.1
        )
    )
    harmonics: HarmonicJitterParams = dataclasses.field(
        default_factory=HarmonicJitterParams
    )

    def __post_init__(self) -> None:
        lo, hi = self.n_fish
        if not (1 <= lo <= hi):
            msg = f"population.n_fish must satisfy 1 <= low <= high, got {self.n_fish}"
            raise ValueError(msg)
        _check_range(
            self.baseline_uniform_hz, "population.baseline_uniform_hz", positive=True
        )
        _check_number(
            self.baseline_uniform_prob,
            "population.baseline_uniform_prob",
            low=0.0,
            high=1.0,
        )
        for name in ("female_hz", "male_hz"):
            mean, sd = getattr(self, name)
            _check_number(mean, f"population.{name}[0]", low=0.0, strict_low=True)
            _check_number(sd, f"population.{name}[1]", low=0.0)
        _check_number(self.male_prob, "population.male_prob", low=0.0, high=1.0)
        _check_number(
            self.close_pair_prob, "population.close_pair_prob", low=0.0, high=1.0
        )
        _check_range(self.close_pair_df_hz, "population.close_pair_df_hz", low=0.0)
        _check_number(self.crossing_prob, "population.crossing_prob", low=0.0, high=1.0)
        _check_range(self.ramp_overshoot_hz, "population.ramp_overshoot_hz", low=0.0)
        _check_number(self.ramp_prob, "population.ramp_prob", low=0.0, high=1.0)
        _check_range(self.ramp_df_hz, "population.ramp_df_hz", low=0.0)
        _check_range(self.ramp_duration_s, "population.ramp_duration_s", positive=True)
        _check_number(self.drift_sd_log_sd, "population.drift_sd_log_sd", low=0.0)
        _check_number(
            self.drift_timescale_log_sd, "population.drift_timescale_log_sd", low=0.0
        )
        _check_number(self.shared_trend_per_h[0], "population.shared_trend_per_h[0]")
        _check_number(
            self.shared_trend_per_h[1], "population.shared_trend_per_h[1]", low=0.0
        )

    @property
    def max_baseline_hz(self) -> float:
        """Highest possible baseline before ramps and drift."""
        return self.baseline_uniform_hz[1]


@dataclasses.dataclass(frozen=True)
class ChirpConfig:
    """Chirp templates, timing and amplitude modulation.

    Timing. Each fish has a spontaneous chirp rate (uniform in ``rate_hz``).
    The default rates are far above the real ones (tube competitions, v1
    detections: losers 61/h, winners 6/h median) for training efficiency.
    Spontaneous chirps come in bursts: burst onsets are a Poisson process
    with rate ``rate / burst_size_mean``; a burst has a geometric number of
    chirps (mean ``burst_size_mean``) whose centre-to-centre intervals are
    log-normal (``burst_interval_s`` = median and log SD), which with the
    defaults reproduces the measured within-fish intervals below 2 s
    (8 % / 34 % / 54 % below 0.5 / 1 / 2 s overall).

    Responses: after every spontaneous chirp, with probability
    ``response_prob`` another fish (uniform among the others) chirps
    ``response_delay_s`` later (uniform, centre to centre; the measured
    excess of the other fish's next chirp is at 0.45-2.2 s). A fraction
    ``simultaneous_fraction`` of the responses is instead placed within
    ``+-simultaneous_window_s`` of the trigger: near-simultaneous chirps of
    different fish are under-detected by v1 (0.27 % observed vs 1.4 %
    expected by chance), so the detector must see them. Responses do not
    trigger further responses.

    Limits: chirps of one fish are at least ``min_interval_s`` apart centre
    to centre (the thesis' reading) and their supports never overlap; later
    chirps that violate this are dropped. ``max_rate_hz`` optionally caps a
    fish's mean rate (None: no cap; real within-fish intervals are 34 %
    below 1 s). Chirp centres lie in ``[0, duration]``; chirps near the edges
    are cut by the scene boundary and flagged ``truncated``.

    Templates are real chirps from the curated library (decision 5),
    baseline-relative, added to the fish's baseline frequency, drawn per
    fish with the library's cycler. With ``equalize`` (decision 10), the
    pool is binned on excursion and width at 10 % in ``equalize_bins`` bins.
    The library's sampling options (``equalize_spacing``, ``equalize_alpha``,
    ``equalize_balance``) are passed through when set; unset (the default)
    they take the library's own defaults (v3: log-spaced bins, alpha 0.6, a
    bin with ``c`` templates drawn with probability proportional to
    ``c ** (1 - alpha)``), so switching libraries is a path change.
    ``scene.json`` records the values in effect. Templates may be up to 1.2 s
    long (library v3); only each chirp's support matters.

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
        Path of the template library (default ``/mnt/data1/chirp_v2/templates/v3``).
    families : tuple of str
        Template families to draw from (``"small"``, ``"big"``; default both).
    exclude : tuple of str
        Library flags whose templates are excluded (default: the library's
        default exclusion list).
    equalize : bool
        Equalise the (excursion x width) histogram (default True).
    equalize_bins : (int, int)
        Equalisation bins on excursion and on width at 10 % (default 10 x 10).
    equalize_spacing : str or None
        Bin spacing, ``"log"`` or ``"linear"`` (default None: library
        default).
    equalize_alpha : float or None
        Tempering exponent in [0, 1] (default None: library default).
    equalize_balance : str or None
        Within-bin balancing of library v4 (e.g. ``"individual"``; default
        None: library default; not passed to libraries without it).
    rate_hz : (float, float)
        Range of the spontaneous chirp rate per fish in chirps/s
        (default 0.05-0.5, i.e. 180-1800/h).
    max_rate_hz : float or None
        Optional upper bound of a fish's mean chirp rate (default None).
    min_interval_s : float
        Minimum centre-to-centre interval between chirps of one fish in s
        (default 0.2); supports may not overlap either.
    burst_size_mean : float
        Mean number of chirps per burst, >= 1 (default 2.2).
    burst_interval_s : (float, float)
        Median (s) and log SD of the centre-to-centre interval within a burst
        (default 0.75 s, 0.45).
    response_prob : float
        Probability that a spontaneous chirp triggers a response by another
        fish (default 0.3).
    response_delay_s : (float, float)
        Delay of a response after its trigger in s (default 0.45-2.2).
    simultaneous_fraction : float
        Fraction of responses placed near-simultaneously (default 0.1).
    simultaneous_window_s : float
        Half-width of the near-simultaneous window in s (default 0.25).
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
    equalize_spacing: str | None = None
    equalize_alpha: float | None = None
    equalize_balance: str | None = None
    rate_hz: FloatRange = (0.05, 0.5)
    max_rate_hz: float | None = None
    min_interval_s: float = 0.2
    burst_size_mean: float = 2.2
    burst_interval_s: FloatRange = (0.75, 0.45)
    response_prob: float = 0.3
    response_delay_s: FloatRange = (0.45, 2.2)
    simultaneous_fraction: float = 0.1
    simultaneous_window_s: float = 0.25
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
        if self.equalize_spacing not in (None, "log", "linear"):
            msg = (
                "chirps.equalize_spacing must be 'log' or 'linear', "
                f"got {self.equalize_spacing!r}"
            )
            raise ValueError(msg)
        if self.equalize_alpha is not None:
            _check_number(
                self.equalize_alpha, "chirps.equalize_alpha", low=0.0, high=1.0
            )
        _check_range(self.rate_hz, "chirps.rate_hz", low=0.0)
        if self.max_rate_hz is not None:
            _check_number(self.max_rate_hz, "chirps.max_rate_hz", low=0.0)
        _check_number(self.min_interval_s, "chirps.min_interval_s", low=0.0)
        _check_number(self.burst_size_mean, "chirps.burst_size_mean", low=1.0)
        _check_number(
            self.burst_interval_s[0],
            "chirps.burst_interval_s[0]",
            low=0.0,
            strict_low=True,
        )
        _check_number(self.burst_interval_s[1], "chirps.burst_interval_s[1]", low=0.0)
        _check_number(self.response_prob, "chirps.response_prob", low=0.0, high=1.0)
        _check_range(self.response_delay_s, "chirps.response_delay_s", low=0.0)
        _check_number(
            self.simultaneous_fraction,
            "chirps.simultaneous_fraction",
            low=0.0,
            high=1.0,
        )
        _check_number(
            self.simultaneous_window_s, "chirps.simultaneous_window_s", low=0.0
        )
        if self.amplitude_mode not in ("frequency", "measured"):
            msg = (
                "chirps.amplitude_mode must be 'frequency' or 'measured', "
                f"got {self.amplitude_mode!r}"
            )
            raise ValueError(msg)
        _check_range(self.dip, "chirps.dip", low=0.0, high=1.0)


@dataclasses.dataclass(frozen=True)
class MovementConfig:
    """Two movement regimes, mixed per scene or per fish (calibration v1).

    The real fish switch between an active regime (dark phase: ~18
    strongest-electrode switches per minute, no long dwells) and a resting
    regime (light phase: ~1 switch per minute, 85 % of the time in dwells
    of a minute or more). The closest simulated variants are the default
    :class:`~gridtools.sim.movement.MovementParams` (dark) and resting with
    little rotation (light: mean rest 600 s, move 10 s, rest turn factor
    0.05). Speeds and turning are not identified by the data.

    Attributes
    ----------
    dark : MovementParams
        Active regime (default: ``MovementParams()``).
    light : MovementParams
        Resting regime (default: mean rest 600 s, move 10 s, rest turn
        factor 0.05).
    light_prob : float
        Probability of the resting regime (default 0.5).
    per : {"scene", "fish"}
        Draw the regime once per scene (default; lighting is shared) or per
        fish.

    Note: a ``[movement.light]`` table in TOML is a complete
    ``MovementParams``; keys left out take the ``MovementParams`` defaults,
    not those of the light regime above.
    """

    dark: MovementParams = dataclasses.field(default_factory=MovementParams)
    light: MovementParams = dataclasses.field(
        default_factory=lambda: MovementParams(
            mean_rest_s=600.0, mean_move_s=10.0, rest_turn_factor=0.05
        )
    )
    light_prob: float = 0.5
    per: Literal["scene", "fish"] = "scene"

    def __post_init__(self) -> None:
        _check_number(self.light_prob, "movement.light_prob", low=0.0, high=1.0)
        if self.per not in ("scene", "fish"):
            msg = f"movement.per must be 'scene' or 'fish', got {self.per!r}"
            raise ValueError(msg)


@dataclasses.dataclass(frozen=True)
class FieldConfig:
    """Dipole field and per-fish EOD amplitude (calibration v1).

    The falloff exponent ``n`` and the regulariser ``a`` (which also absorbs
    the fish-electrode depth offset; the depth offset itself is 0 here) are
    drawn per scene along the ridge of the fit to the real per-electrode
    power profiles: ``n`` uniform in ``exponent``, ``a / spacing``
    interpolated linearly in the ridge points (``ridge_exponents``,
    ``ridge_regularisers_rel``: best regulariser per exponent at margin 0;
    the best fit is n = 1.5, a = 0.05 spacings). Steeper falloff with a
    larger regulariser gives similar profiles over the grid distances.

    The EOD amplitude (recording units: volts at the ADC, i.e. electrode
    potential x 100) is log-normal with median ``amplitude_median_v`` and
    log SD ``amplitude_log_sd``, clipped at ``amplitude_clip_sd`` SDs. The
    median reproduces the median real fundamental at the strongest
    electrode (4.2e-4 V) for the best-fit field on the 0.5-m grid; for any
    other (n, a, spacing) the amplitude is rescaled by the ratio of the
    median strongest-electrode gains (uniform positions and headings over
    the arena), so the calibrated, measured quantity stays fixed.

    Attributes
    ----------
    amplitude_median_v : float
        Median EOD amplitude in V (default 2.16e-3).
    amplitude_log_sd : float
        Log SD of the EOD amplitude across fish (default 0.748).
    amplitude_clip_sd : float
        Clip the log-normal at this many SDs (default 2.5).
    exponent : (float, float)
        Range of the falloff exponent (default 1.5-2).
    ridge_exponents, ridge_regularisers_rel : tuple of float
        Ridge points (defaults (1.5, 1.75, 2.0) and (0.05, 0.15, 0.3)).
    calibration_exponent, calibration_regulariser_rel : float
        Field the amplitude was calibrated for (defaults 1.5, 0.05).
    calibration_spacing_m : float
        Spacing the amplitude was calibrated for (default 0.5).
    reference_distance_m : float
        Axis distance with gain 1 (default 0.1).
    """

    amplitude_median_v: float = 2.16e-3
    amplitude_log_sd: float = 0.748
    amplitude_clip_sd: float = 2.5
    exponent: FloatRange = (1.5, 2.0)
    ridge_exponents: tuple[float, ...] = (1.5, 1.75, 2.0)
    ridge_regularisers_rel: tuple[float, ...] = (0.05, 0.15, 0.3)
    calibration_exponent: float = 1.5
    calibration_regulariser_rel: float = 0.05
    calibration_spacing_m: float = 0.5
    reference_distance_m: float = 0.1

    def __post_init__(self) -> None:
        _check_number(
            self.amplitude_median_v,
            "field.amplitude_median_v",
            low=0.0,
            strict_low=True,
        )
        _check_number(self.amplitude_log_sd, "field.amplitude_log_sd", low=0.0)
        _check_number(
            self.amplitude_clip_sd, "field.amplitude_clip_sd", low=0.0, strict_low=True
        )
        _check_range(self.exponent, "field.exponent", positive=True)
        if len(self.ridge_exponents) != len(self.ridge_regularisers_rel) or not (
            self.ridge_exponents
        ):
            msg = "field.ridge_exponents and ridge_regularisers_rel must match"
            raise ValueError(msg)
        if list(self.ridge_exponents) != sorted(self.ridge_exponents):
            msg = "field.ridge_exponents must be increasing"
            raise ValueError(msg)
        for value in self.ridge_regularisers_rel:
            _check_number(value, "field.ridge_regularisers_rel", low=0.0)
        _check_number(
            self.calibration_exponent,
            "field.calibration_exponent",
            low=0.0,
            strict_low=True,
        )
        _check_number(
            self.calibration_regulariser_rel,
            "field.calibration_regulariser_rel",
            low=0.0,
        )
        _check_number(
            self.calibration_spacing_m,
            "field.calibration_spacing_m",
            low=0.0,
            strict_low=True,
        )
        _check_number(
            self.reference_distance_m,
            "field.reference_distance_m",
            low=0.0,
            strict_low=True,
        )

    def regulariser_rel(self, exponent: float) -> float:
        """Regulariser (in spacings) on the ridge for a falloff exponent."""
        import numpy as np

        return float(
            np.interp(exponent, self.ridge_exponents, self.ridge_regularisers_rel)
        )


@dataclasses.dataclass(frozen=True)
class NoiseConfig:
    """Electrode noise, power-line hum, channel gains, faults and clipping.

    Physical units (volts at the ADC), calibrated on the 33 tube-competition
    trials. The synthetic noise is one component of the background; real
    background segments will become the main noise source later, so it can
    be switched off (``colored``, ``hum``).

    Noise floor: white (measured spectral exponent 0.02), one-sided PSD
    ``N0`` drawn per scene from ``floor_modes`` (a mixture of normals in dB re
    1 V^2/Hz with weights; defaults: the 2019 trials, -121.4 +- 1.3 dB, weight
    0.42, and the 2020 trials, -131.9 +- 1.6 dB, weight 0.58), noise SD
    ``sqrt(N0 * fs / 2)``. Per electrode the floor varies by
    ``electrode_floor_sd_db`` (measured 2.3 dB).

    Harder than real (robustness): with probability ``harder_prob`` the floor
    is raised by ``harder_db`` (uniform; default 0-20 dB), otherwise it is
    realistic.

    Hum: amplitude of the 50-Hz line on the electrode with most hum,
    log-uniform in ``hum_amplitude_v`` (measured 2.6e-6 to 2.6e-4 V, 5-95 %;
    2019 ~2.4e-4 V, 2020 ~4.8e-6 V); ``hum_n_harmonics`` (uniform integer
    range; measured median 17), geometric decay ``hum_harmonic_decay``
    (0.79), per-electrode coupling log SD ``hum_coupling_log_sd`` (1.15). Real
    hum is odd-harmonic dominated, which a single decay does not express.

    Attributes
    ----------
    colored : bool
        Add the synthetic electrode noise (default True).
    exponent : (float, float)
        Range of the spectral exponent (default 0-0: white).
    f_min_hz : float
        Corner below which the spectrum is flat (default 1 Hz).
    floor_modes : tuple of (float, float, float)
        (mean dB re 1 V^2/Hz, SD dB, weight) of the floor mixture.
    electrode_floor_sd_db : float
        SD of the floor across electrodes in dB (default 2.32).
    harder_prob : float
        Probability of a harder-than-real scene (default 0.5).
    harder_db : (float, float)
        Range of the extra floor in dB for those scenes (default 0-20).
    hum : bool
        Add power-line hum (default True).
    hum_amplitude_v : (float, float)
        Range of the 50-Hz amplitude on the strongest electrode, log-uniform
        (default 2.65e-6 to 2.64e-4 V).
    hum_line_freq_hz : float
        Mains frequency (default 50).
    hum_n_harmonics : (int, int)
        Inclusive range of hum components (default 8-32).
    hum_harmonic_decay : float
        Amplitude ratio between successive harmonics (default 0.788).
    hum_coupling_log_sd : float
        Log-normal spread of the per-electrode coupling (default 1.15).
    gain_mismatch_log_sd : float
        SD of the log channel gain (default 0.1; not identifiable, bounded
        by the floor spread).
    dead_prob, noisy_prob : float
        Probability of a dead / noisy electrode (defaults 0.044, 0.023).
    dead_gain : float
        Channel gain of a dead electrode (default 1e-3).
    noisy_factor : float
        Noise SD factor of a noisy electrode (default 5.04).
    clip_limit : float or None
        Symmetric clipping in V (default 0.02048, the ADC range). None
        disables clipping. Clipping breaks the linearity
        ``recording = sum(render_fish) + background`` where it acts (real
        recordings stay below 6 mV; clicks may reach it).
    """

    colored: bool = True
    exponent: FloatRange = (0.0, 0.0)
    f_min_hz: float = 1.0
    floor_modes: tuple[tuple[float, float, float], ...] = (
        (-121.4, 1.3, 0.42),
        (-131.9, 1.6, 0.58),
    )
    electrode_floor_sd_db: float = 2.32
    harder_prob: float = 0.5
    harder_db: FloatRange = (0.0, 20.0)
    hum: bool = True
    hum_amplitude_v: FloatRange = (2.65e-6, 2.64e-4)
    hum_line_freq_hz: float = 50.0
    hum_n_harmonics: IntRange = (8, 32)
    hum_harmonic_decay: float = 0.788
    hum_coupling_log_sd: float = 1.15
    gain_mismatch_log_sd: float = 0.1
    dead_prob: float = 0.044
    noisy_prob: float = 0.023
    dead_gain: float = 1e-3
    noisy_factor: float = 5.04
    clip_limit: float | None = 0.02048

    def __post_init__(self) -> None:
        _check_range(self.exponent, "noise.exponent", low=0.0)
        _check_number(self.f_min_hz, "noise.f_min_hz", low=0.0, strict_low=True)
        if not self.floor_modes:
            msg = "noise.floor_modes must not be empty"
            raise ValueError(msg)
        for mean, sd, weight in self.floor_modes:
            _check_number(mean, "noise.floor_modes mean")
            _check_number(sd, "noise.floor_modes sd", low=0.0)
            _check_number(weight, "noise.floor_modes weight", low=0.0)
        if sum(w for _, _, w in self.floor_modes) <= 0:
            msg = "noise.floor_modes weights must sum to > 0"
            raise ValueError(msg)
        _check_number(
            self.electrode_floor_sd_db, "noise.electrode_floor_sd_db", low=0.0
        )
        _check_number(self.harder_prob, "noise.harder_prob", low=0.0, high=1.0)
        _check_range(self.harder_db, "noise.harder_db")
        _check_range(self.hum_amplitude_v, "noise.hum_amplitude_v", positive=True)
        _check_number(
            self.hum_line_freq_hz, "noise.hum_line_freq_hz", low=0.0, strict_low=True
        )
        lo, hi = self.hum_n_harmonics
        if not 1 <= lo <= hi:
            msg = f"noise.hum_n_harmonics must satisfy 1 <= low <= high, got {self.hum_n_harmonics}"
            raise ValueError(msg)
        _check_number(self.hum_harmonic_decay, "noise.hum_harmonic_decay", low=0.0)
        _check_number(self.hum_coupling_log_sd, "noise.hum_coupling_log_sd", low=0.0)
        _check_number(self.gain_mismatch_log_sd, "noise.gain_mismatch_log_sd", low=0.0)
        _check_number(self.dead_prob, "noise.dead_prob", low=0.0, high=1.0)
        _check_number(self.noisy_prob, "noise.noisy_prob", low=0.0, high=1.0)
        _check_number(self.dead_gain, "noise.dead_gain", low=0.0)
        _check_number(self.noisy_factor, "noise.noisy_factor", low=0.0)
        if self.clip_limit is not None:
            _check_number(self.clip_limit, "noise.clip_limit", low=0.0, strict_low=True)


@dataclasses.dataclass(frozen=True)
class ClickConfig:
    """Impulsive artefacts (clicks), added after the channel gains.

    Calibration v1: high-band transients occur at 792/h (median; 18-44,320/h,
    5-95 % across trials), and they look like recording artefacts: 9 % of
    the intervals between clicks are 30 ms (pairs), and in some trials clicks
    recur every 10.24 s or come as onset/offset pairs 0.1 s apart about once
    per second (the 100-ms LED sync pulses). Per scene the rate is
    log-normal (median ``rate_per_h_median``, log10 SD ``rate_log10_sd``,
    capped at ``rate_per_h_max``); each click gets a partner
    ``pair_gap_s`` later with probability ``pair_fraction``. Two periodic
    patterns are optional per scene: clicks every ``periodic_interval_s``
    (probability ``periodic_prob``) and LED onset/offset pairs every
    ``led_interval_s`` (probability ``led_prob``).

    Each click has a log-uniform duration and peak amplitude (in units of the
    scene's noise SD), a random polarity, and a shape: a Hann-windowed
    white-noise burst or a spike (instant rise, exponential decay). It hits
    a geometric number of electrodes (mean ``electrodes_mean``); electrodes
    after the first get a relative amplitude uniform in ``spread``.

    Attributes
    ----------
    rate_per_h_median : float
        Median click rate per hour (default 792).
    rate_log10_sd : float
        SD of log10 of the rate across scenes (default 1.0).
    rate_per_h_max : float
        Cap of the rate per hour (default 50000).
    pair_fraction : float
        Probability that a click comes with a partner (default 0.1).
    pair_gap_s : float
        Gap to the partner in s (default 0.03).
    periodic_prob : float
        Probability of a click every ``periodic_interval_s`` (default 0.1).
    periodic_interval_s : float
        Period of the periodic clicks in s (default 10.24).
    led_prob : float
        Probability of LED onset/offset click pairs (default 0.1).
    led_interval_s : float
        Interval between LED pulses in s (default 1.0).
    led_pulse_s : float
        Gap between onset and offset clicks in s (default 0.1).
    duration_s : (float, float)
        Range of click durations in s, log-uniform (default 0.1-5 ms).
    amplitude_rel : (float, float)
        Range of the peak amplitude in units of the noise SD, log-uniform
        (default 10-200: with more energetic clicks the measured noise floor
        of click-rich scenes rises several dB above the real floors).
    electrodes_mean : float
        Mean number of electrodes per click, >= 1 (default 1.5).
    spread : (float, float)
        Relative amplitude on secondary electrodes (default 0.1-1).
    burst_prob : float
        Probability of the burst shape (else spike; default 0.5).
    """

    rate_per_h_median: float = 792.0
    rate_log10_sd: float = 1.0
    rate_per_h_max: float = 50_000.0
    pair_fraction: float = 0.1
    pair_gap_s: float = 0.03
    periodic_prob: float = 0.1
    periodic_interval_s: float = 10.24
    led_prob: float = 0.1
    led_interval_s: float = 1.0
    led_pulse_s: float = 0.1
    duration_s: FloatRange = (1e-4, 5e-3)
    amplitude_rel: FloatRange = (10.0, 200.0)
    electrodes_mean: float = 1.5
    spread: FloatRange = (0.1, 1.0)
    burst_prob: float = 0.5

    def __post_init__(self) -> None:
        _check_number(self.rate_per_h_median, "clicks.rate_per_h_median", low=0.0)
        _check_number(self.rate_log10_sd, "clicks.rate_log10_sd", low=0.0)
        _check_number(self.rate_per_h_max, "clicks.rate_per_h_max", low=0.0)
        for name in ("pair_fraction", "periodic_prob", "led_prob", "burst_prob"):
            _check_number(getattr(self, name), f"clicks.{name}", low=0.0, high=1.0)
        for name in (
            "pair_gap_s",
            "periodic_interval_s",
            "led_interval_s",
            "led_pulse_s",
        ):
            _check_number(
                getattr(self, name), f"clicks.{name}", low=0.0, strict_low=True
            )
        _check_range(self.duration_s, "clicks.duration_s", positive=True)
        _check_range(self.amplitude_rel, "clicks.amplitude_rel", positive=True)
        _check_number(self.electrodes_mean, "clicks.electrodes_mean", low=1.0)
        _check_range(self.spread, "clicks.spread", low=0.0)


@dataclasses.dataclass(frozen=True)
class TrackConfig:
    """Ground-truth tracks in wavetracker layout.

    ``sign_v`` is computed from the simulated recording exactly as the
    production tracks were (calibration v1, verified to 3e-14 dB on the real
    data): 10 log10 of the one-sided Hann power spectral density
    (matplotlib ``specgram`` scaling, mean removed per frame) of each
    channel, ``round(samplerate_hz / power_resolution_hz)`` samples centred
    at the frame time, at the bin
    nearest the fish's frequency. Frames whose window extends beyond the
    recording are zero-padded (wavetracker has no frames there).

    Attributes
    ----------
    step_s : float
        Frame interval in s (default 0.1229, the production frame grid,
        about 8.1 Hz). Frames start at 0 s.
    power_resolution_hz : float
        Frequency resolution of ``sign_v`` (default 20 kHz / 32768 =
        0.61 Hz, the 2019 and May/June 2020 tracks; 20 kHz / 16384 in March
        2020). The FFT length is ``round(samplerate_hz / power_resolution_hz)``.
    """

    step_s: float = 0.1229
    power_resolution_hz: float = 20_000.0 / 32_768

    def __post_init__(self) -> None:
        _check_number(self.step_s, "tracks.step_s", low=0.0, strict_low=True)
        _check_number(
            self.power_resolution_hz,
            "tracks.power_resolution_hz",
            low=0.0,
            strict_low=True,
        )

    def power_nfft(self, samplerate_hz: float) -> int:
        """FFT length of ``sign_v`` at a sampling rate."""
        return max(2, round(samplerate_hz / self.power_resolution_hz))


@dataclasses.dataclass(frozen=True)
class FootprintConfig:
    """Ground-truth chirp boxes from spectrogram footprints (decision 7).

    The reference spectrogram matches chirpdetector's default front end:
    Hann window of ``window_s``, hop ``hop_s``, electrode-summed power, band
    ``f_min_hz``-``f_max_hz``; in samples ``round(window_s * fs)`` and
    ``round(hop_s * fs)``, so every sampling rate gets the same time and
    frequency resolution. See :mod:`gridtools.sim.footprint`.

    Attributes
    ----------
    enabled : bool
        Compute footprints in :func:`~gridtools.sim.scene.simulate_scene`
        (default True).
    window_s : float
        Window length in s (default 0.2048: 4096 samples at 20 kHz, 4.9 Hz
        resolution).
    hop_s : float
        Hop in s (default 0.005: 100 samples at 20 kHz).
    f_min_hz, f_max_hz : float
        Band of the reference spectrogram (default 500-1500 Hz).
    threshold_db : float
        Footprint cells reach within this many dB of the strongest
        difference (default 30: the box holds >= 99.5 % of the difference
        energy; at 20 dB 1-2.5 % and at 25 dB up to 1.5 % fall outside with
        a 0.2-s window).
    visible_snr_db : float
        ``visible`` when the footprint SNR reaches this (default 0 dB;
        to be tuned in training).
    margin_s : float
        Extra time around support and window (default 0.05 s).
    background : {"others", "recording"}
        Background of the visibility: everything but the emitter (default)
        or the recording minus only this chirp (see
        :func:`gridtools.sim.footprint.chirp_footprint`).
    """

    enabled: bool = True
    window_s: float = 0.2048
    hop_s: float = 0.005
    f_min_hz: float = 500.0
    f_max_hz: float = 1500.0
    threshold_db: float = 30.0
    visible_snr_db: float = 0.0
    margin_s: float = 0.05
    background: Literal["others", "recording"] = "others"

    def __post_init__(self) -> None:
        if self.background not in ("others", "recording"):
            msg = f"footprint.background must be 'others' or 'recording', got {self.background!r}"
            raise ValueError(msg)
        _check_number(self.window_s, "footprint.window_s", low=0.0, strict_low=True)
        _check_number(self.hop_s, "footprint.hop_s", low=0.0, strict_low=True)
        if not self.f_max_hz > self.f_min_hz >= 0:
            msg = "footprint band must satisfy 0 <= f_min_hz < f_max_hz"
            raise ValueError(msg)
        _check_number(
            self.threshold_db, "footprint.threshold_db", low=0.0, strict_low=True
        )
        _check_number(self.visible_snr_db, "footprint.visible_snr_db")
        _check_number(self.margin_s, "footprint.margin_s", low=0.0)


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
    window_s : float
        Window of the overview spectrogram in s (default 0.2048).
    hop_s : float
        Hop of the overview spectrogram in s (default 0.0256).
    n_zooms : int
        Number of zoomed panels (default 6).
    zoom_s : float
        Duration of a zoomed panel in s (default 5).
    zoom_window_s : float
        Window of the zoomed panels in s (default 0.0256: a chirp's
        excursion is visible; fish closer than ~40 Hz merge, their tracks
        are drawn on top).
    zoom_hop_s : float
        Hop of the zoomed panels in s (default 0.0016).
    dpi : int
        Resolution of the PNGs (default 110).
    """

    enabled: bool = True
    f_min_hz: float = 500.0
    f_max_hz: float = 1500.0
    window_s: float = 0.2048
    hop_s: float = 0.0256
    n_zooms: int = 6
    zoom_s: float = 5.0
    zoom_window_s: float = 0.0256
    zoom_hop_s: float = 0.0016
    dpi: int = 110

    def __post_init__(self) -> None:
        _check_number(self.f_min_hz, "plot.f_min_hz", low=0.0)
        if not self.f_max_hz > self.f_min_hz:
            msg = "plot.f_max_hz must be > plot.f_min_hz"
            raise ValueError(msg)
        for name in ("window_s", "hop_s", "zoom_window_s", "zoom_hop_s"):
            _check_number(getattr(self, name), f"plot.{name}", low=0.0, strict_low=True)
        if self.dpi < 1:
            msg = "plot.dpi must be >= 1"
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
        Sampling rate in Hz (default 20000, as the real grids). Every
        setting is in physical units, so any rate gives the same scene.
    samplerate_choices_hz : tuple of float
        Augmentation: if not empty, the rate is drawn per scene from these
        (default empty: always ``samplerate_hz``). ``scene.json`` records the
        resolved config with the drawn rate.
    grid, population, chirps, movement, field, noise, clicks, tracks, plot, footprint
        Sections; see their classes.
    """

    name: str = "default"
    duration_s: float = 120.0
    samplerate_hz: float = 20_000.0
    samplerate_choices_hz: tuple[float, ...] = ()
    grid: GridConfig = dataclasses.field(default_factory=GridConfig)
    population: PopulationConfig = dataclasses.field(default_factory=PopulationConfig)
    chirps: ChirpConfig = dataclasses.field(default_factory=ChirpConfig)
    movement: MovementConfig = dataclasses.field(default_factory=MovementConfig)
    field: FieldConfig = dataclasses.field(default_factory=FieldConfig)
    noise: NoiseConfig = dataclasses.field(default_factory=NoiseConfig)
    clicks: ClickConfig = dataclasses.field(default_factory=ClickConfig)
    tracks: TrackConfig = dataclasses.field(default_factory=TrackConfig)
    plot: PlotConfig = dataclasses.field(default_factory=PlotConfig)
    footprint: FootprintConfig = dataclasses.field(default_factory=FootprintConfig)

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            msg = f"name must be a non-empty string without '/', got {self.name!r}"
            raise ValueError(msg)
        _check_number(self.duration_s, "duration_s", low=0.0, strict_low=True)
        _check_number(self.samplerate_hz, "samplerate_hz", low=0.0, strict_low=True)
        if self.n_samples < 2:
            msg = "the scene must have at least 2 samples"
            raise ValueError(msg)
        f_top = self.population.max_baseline_hz * 2.0
        for rate in (self.samplerate_hz, *self.samplerate_choices_hz):
            _check_number(rate, "samplerate", low=0.0, strict_low=True)
            if f_top >= rate / 2.0:
                msg = (
                    f"samplerate {rate} Hz is too low for baselines up to "
                    f"{self.population.max_baseline_hz} Hz and their harmonics"
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
