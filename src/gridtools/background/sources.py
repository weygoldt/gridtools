"""Source recordings of the background library.

A :class:`SourceRecording` names one real recording and the files around
it: tracks (wavetracker layout), chirpdetector v1 detections and hand
labels. :func:`competition_2019_sources` and :func:`competition_2022_sources`
list the tube competition recordings on ``/mnt/data2``:

2019/2020 competitions (``/mnt/data2/2019_tube_competition/<trial>/``)
    50 trials, fishgrid raw (3x3 grids in 2019, 15 electrodes in 2020) at
    20 kHz, tracks next to the raw file (``sign_v`` of the 2019 trials has 8
    columns: channel 9 carries no fish signal and was not tracked), v1
    detections in ``chirpdetector_bboxes.csv`` (empty for 2020-03-12, which
    is 190 s long). Hand labels in ``traces-grid1-editable-labels.csv`` of
    2019-12-09 and 2020-06-03 (the locked real test set).
2022 competitions (``/mnt/data2/2022_tube_competition/raw/<trial>/``)
    19 trials, fishgrid raw with 11 channels at 20 kHz, 4 h each. Tracks in
    ``interrim/<trial>/`` (2 fish each; ``sign_v`` verified against the raw
    data with 32768-point frames); no chirp detections. The ``fishgrid.cfg``
    files declare a 1 x 11 grid at 0.5 m, but a README in 2022-05-10 says
    the config was changed to 1 x 11 by hand, and the per-electrode power
    profiles of the tracked fish do not follow a line (the second-strongest
    electrode is rarely a neighbour on the line), so the real electrode
    positions are unknown: ``geometry_known=False``. Hand labels in
    2022-05-12 and 2022-06-01.

Some 2020 raw files end with a partial frame (2020-03-19, -20, -24, -25);
:func:`open_source_recording` drops it instead of refusing the file.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gridtools.io import Recording, open_recording, read_fishgrid_config

__all__ = [
    "COMPETITION_2019_ROOT",
    "COMPETITION_2022_ROOT",
    "DARK_PHASE_S",
    "SourceRecording",
    "competition_2019_sources",
    "competition_2022_sources",
    "open_source_recording",
]

COMPETITION_2019_ROOT = Path("/mnt/data2/2019_tube_competition")
COMPETITION_2022_ROOT = Path("/mnt/data2/2022_tube_competition")

#: The competitions start with 3 h of darkness, then the light is switched on.
DARK_PHASE_S = 10_800.0

_LABELS = "traces-grid1-editable-labels.csv"
_DETECTIONS = "chirpdetector_bboxes.csv"
_RAW = "traces-grid1.raw"


@dataclass(frozen=True)
class SourceRecording:
    """One real recording and the files that describe it.

    Attributes
    ----------
    dataset : str
        Short dataset name (``"competition2019"``, ``"competition2022"``).
    name : str
        Recording name, unique within the dataset (the trial directory).
    recording_path : pathlib.Path
        Anything :func:`gridtools.io.open_recording` accepts.
    config_path : pathlib.Path or None
        ``fishgrid.cfg`` copied next to every segment (geometry, rate, input
        range).
    tracks_dir : pathlib.Path or None
        Directory with wavetracker arrays, if tracked.
    detections_path : pathlib.Path or None
        chirpdetector v1 detections; None if the recording was never
        screened for chirps.
    labels_path : pathlib.Path or None
        audian hand labels (the locked real test set), if any.
    geometry_known : bool
        Whether the configuration's layout (rows, columns, channel order) is
        the real one.
    spacing_known : bool
        Whether its electrode spacing is measured. False for the
        competitions: ``fishgrid.cfg`` says 0.5 m, which conflicts with the
        tank size in the paper, so positions are nominal.
    dark_phase_s : float or None
        Duration of the initial dark phase, if known.
    note : str
        Free text for the manifest.
    """

    dataset: str
    name: str
    recording_path: Path
    config_path: Path | None = None
    tracks_dir: Path | None = None
    detections_path: Path | None = None
    labels_path: Path | None = None
    geometry_known: bool = True
    spacing_known: bool = False
    dark_phase_s: float | None = DARK_PHASE_S
    note: str = ""

    @property
    def key(self) -> str:
        """``dataset/name``, the split group of the recording's segments."""
        return f"{self.dataset}/{self.name}"


def _optional(path: Path) -> Path | None:
    return path if path.exists() else None


def _detections(path: Path) -> Path | None:
    # 2020-03-12 has a 1-byte file: v1 never produced a table there
    if not path.exists() or path.stat().st_size < 16:
        return None
    return path


def _select(dirs: Iterable[Path], names: Iterable[str] | None) -> list[Path]:
    dirs = sorted(d for d in dirs if d.is_dir() and (d / _RAW).exists())
    if names is None:
        return dirs
    wanted = set(names)
    found = [d for d in dirs if d.name in wanted]
    missing = wanted - {d.name for d in found}
    if missing:
        msg = f"recordings not found: {sorted(missing)}"
        raise FileNotFoundError(msg)
    return found


def competition_2019_sources(
    root: str | Path = COMPETITION_2019_ROOT,
    recordings: Iterable[str] | None = None,
) -> list[SourceRecording]:
    """The 2019/2020 tube competition trials.

    Parameters
    ----------
    root : path
        Dataset root with one directory per trial.
    recordings : iterable of str, optional
        Trial names to include (default: all with a raw file).

    Returns
    -------
    list of SourceRecording
    """
    root = Path(root)
    sources = []
    for d in _select(root.glob("20*"), recordings):
        tracked = (d / "ident_v.npy").exists()
        sources.append(
            SourceRecording(
                dataset="competition2019",
                name=d.name,
                recording_path=d,
                config_path=d / "fishgrid.cfg",
                tracks_dir=d if tracked else None,
                detections_path=_detections(d / _DETECTIONS),
                labels_path=_optional(d / _LABELS),
                geometry_known=True,
            )
        )
    return sources


def competition_2022_sources(
    root: str | Path = COMPETITION_2022_ROOT,
    recordings: Iterable[str] | None = None,
) -> list[SourceRecording]:
    """The 2022 tube competition trials (no chirp detections; geometry unknown).

    Parameters
    ----------
    root : path
        Dataset root with ``raw/<trial>`` and ``interrim/<trial>``.
    recordings : iterable of str, optional
        Trial names to include (default: all with a raw file).

    Returns
    -------
    list of SourceRecording
    """
    root = Path(root)
    sources = []
    for d in _select((root / "raw").glob("20*"), recordings):
        tracks = root / "interrim" / d.name
        sources.append(
            SourceRecording(
                dataset="competition2022",
                name=d.name,
                recording_path=d,
                config_path=d / "fishgrid.cfg",
                tracks_dir=tracks if (tracks / "ident_v.npy").exists() else None,
                detections_path=None,
                labels_path=_optional(d / _LABELS),
                geometry_known=False,
                note="fishgrid.cfg says 1x11 at 0.5 m (edited by hand); "
                "true electrode positions unknown",
            )
        )
    return sources


def open_source_recording(source: SourceRecording) -> Recording:
    """Open a source recording, dropping a partial trailing frame of a raw file.

    Returns
    -------
    gridtools.io.Recording
        Memory-mapped samples with the configuration's geometry.
    """
    try:
        return open_recording(source.recording_path)
    except ValueError:
        path = Path(source.recording_path)
        raw = path / _RAW if path.is_dir() else path
        if raw.suffix != ".raw":
            raise
        config = read_fishgrid_config(raw.parent)
        n_channels = config.n_electrodes
        n_frames = raw.stat().st_size // (4 * n_channels)
        data = np.memmap(raw, dtype=np.float32, mode="r", shape=(n_frames, n_channels))
        return Recording(raw, config.samplerate_hz, data, 1.0, config.geometry())
