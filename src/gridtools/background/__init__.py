"""Library of real background segments for hybrid simulations.

Simulated electrode noise is Gaussian; real recordings are not (hum with
odd harmonics, clicks, non-stationary interference, other fish). As in the
hybrid data set of the author's MSc thesis, scenes can add stretches of real
recordings that contain few or no chirps, scaled to a controlled SNR, with
the real fish in them tracked and counted as non-chirping candidates.

- :func:`build_background_library` selects, measures and writes segments
  from :class:`SourceRecording` objects
  (:func:`competition_2019_sources`, :func:`competition_2022_sources`) and
  from field deployments scanned for fish-free stretches
  (:mod:`gridtools.background.field`), optionally only inside a time window
  of each recording, screened by external chirp detectors and with a
  recording gate (:mod:`gridtools.background.screen`);
  :func:`carry_over` takes whole datasets from another library unchanged;
- :class:`BackgroundLibrary` loads the manifest and draws segments;
- :mod:`gridtools.background.levels` defines background and signal levels
  and :func:`scale_for_snr`;
- :mod:`gridtools.background.review` plots every segment and keeps the
  author's veto table (``review.csv``);
- :mod:`gridtools.background.report` documents a built library and
  :mod:`gridtools.background.diagnostics` draws the screens' activity per
  recording (both need the ``plot`` extra); ``python -m
  gridtools.background`` runs all of it.

Scenes use the API re-exported by :mod:`gridtools.sim.background`.
"""

from gridtools.background.build import (
    BackgroundConfig,
    build_background_library,
    carry_over,
)
from gridtools.background.levels import (
    LEVEL_BAND_HZ,
    REFERENCE_BANDWIDTH_HZ,
    BackgroundLevel,
    SnrDistribution,
    background_level,
    fish_signal_level,
    fish_signal_powers,
    scale_for_snr,
    snr_db,
)
from gridtools.background.library import (
    REAL_FISH_IDENT_OFFSET,
    BackgroundLibrary,
    BackgroundSegment,
    add_background,
    clip_tracks,
    merge_tracks,
    read_manifest,
)
from gridtools.background.screen import (
    RecordingGate,
    Screen,
    ScreenTable,
    TimeWindow,
    read_screen,
)
from gridtools.background.select import SelectionConfig
from gridtools.background.sources import (
    SourceRecording,
    competition_2019_sources,
    competition_2022_sources,
    open_source_recording,
)

__all__ = [
    "LEVEL_BAND_HZ",
    "REAL_FISH_IDENT_OFFSET",
    "REFERENCE_BANDWIDTH_HZ",
    "BackgroundConfig",
    "BackgroundLevel",
    "BackgroundLibrary",
    "BackgroundSegment",
    "RecordingGate",
    "Screen",
    "ScreenTable",
    "SelectionConfig",
    "SnrDistribution",
    "SourceRecording",
    "TimeWindow",
    "add_background",
    "background_level",
    "build_background_library",
    "carry_over",
    "clip_tracks",
    "competition_2019_sources",
    "competition_2022_sources",
    "fish_signal_level",
    "fish_signal_powers",
    "merge_tracks",
    "open_source_recording",
    "read_manifest",
    "read_screen",
    "scale_for_snr",
    "snr_db",
]
