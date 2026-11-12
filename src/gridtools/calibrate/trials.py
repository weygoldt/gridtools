"""Find trials, read their metadata and open their raw data robustly.

The tube competition recordings (``/mnt/data2/2019_tube_competition``) hold
one directory per trial, named ``YYYY-MM-DD-HH_MM``. Besides the raw data
(``traces-grid1.raw``) and wavetracker tracks, a trial may have:

- ``temperatures.csv``: water temperature every 300 s (``time/s; T/C``);
- ``chirpdetector_bboxes.csv``: chirpdetector v1 detections.

A table ``trial_summary.csv`` next to the trial directories gives winner and
loser track identities (as in ``ident_v.npy``), their sex and size.

Raw-data quirks found while calibrating (2026-11):

- ``traces-grid1.raw`` of 2020-03-19, 2020-03-20, 2020-03-24 and 2020-03-25
  ends in a partial frame (32-48 trailing bytes). :func:`open_raw` drops it.
- The raw samples are volts at the ADC input (after the 100x amplifier): the
  quantisation step is 0.625 uV = 40.96 mV / 2**16, i.e. a +/-20.48 mV range
  (``AIMaxVolt: 20.0mV``). Electrode potentials are raw / 100.
- In every 3x3 recording of 2019, channel 9 carries no fish signal (its power
  at the fish frequencies is at the noise floor) and ``sign_v.npy`` has 8
  columns, which equal the powers of raw channels 1-8 (verified by
  recomputing ``sign_v`` from the raw data).
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from gridtools.io import Recording, open_recording, read_fishgrid_config

__all__ = [
    "ADC_STEP_V",
    "ANALYSED_TRIAL_DATES",
    "TrialMeta",
    "open_raw",
    "read_temperatures",
    "read_trial_summary",
    "resolve_trial_dir",
    "sign_v_nfft",
]

logger = logging.getLogger(__name__)

#: Recording dates of the 33 competitions analysed in the chirp paper.
ANALYSED_TRIAL_DATES: tuple[str, ...] = (
    "2019-11-25",
    "2019-11-26",
    "2019-11-27",
    "2019-11-28",
    "2019-11-29",
    "2019-12-02",
    "2019-12-03",
    "2019-12-04",
    "2019-12-06",
    "2019-12-09",
    "2019-12-10",
    "2019-12-11",
    "2019-12-12",
    "2019-12-16",
    "2020-03-13",
    "2020-03-16",
    "2020-03-19",
    "2020-03-20",
    "2020-05-11",
    "2020-05-12",
    "2020-05-13",
    "2020-05-14",
    "2020-05-15",
    "2020-05-18",
    "2020-05-19",
    "2020-05-27",
    "2020-05-29",
    "2020-06-02",
    "2020-06-03",
    "2020-06-04",
    "2020-06-05",
    "2020-06-09",
    "2020-06-11",
)

#: Quantisation step of the raw data in volts (16-bit ADC, +/-20.48 mV).
ADC_STEP_V = 40.96e-3 / 2**16


def resolve_trial_dir(root: str | Path, date: str) -> Path:
    """Return the trial directory of ``root`` whose name starts with ``date``.

    Parameters
    ----------
    root : str or Path
        Directory holding one sub-directory per trial.
    date : str
        Date prefix, e.g. ``"2019-11-25"``.

    Returns
    -------
    Path

    Raises
    ------
    FileNotFoundError
        If no directory, or more than one, matches.
    """
    matches = sorted(p for p in Path(root).glob(f"{date}*") if p.is_dir())
    if len(matches) != 1:
        msg = f"{root}: expected one trial directory for {date}, found {len(matches)}"
        raise FileNotFoundError(msg)
    return matches[0]


@dataclass(frozen=True)
class TrialMeta:
    """Winner and loser of one trial, from ``trial_summary.csv``.

    Attributes
    ----------
    recording : str
        Trial directory name.
    winner_id, loser_id : int
        Track identities (``ident_v``) of winner and loser.
    winner_sex, loser_sex : str
        ``"f"`` or ``"m"`` (sex assessed from EODf at 25 degC and partly by
        gonadal inspection, see Raab et al. 2021).
    winner_size_cm, loser_size_cm : float
        Body length in cm.
    """

    recording: str
    winner_id: int
    loser_id: int
    winner_sex: str
    loser_sex: str
    winner_size_cm: float
    loser_size_cm: float

    def role(self, ident: int) -> str:
        """``"winner"``, ``"loser"`` or ``"other"`` for a track identity."""
        if ident == self.winner_id:
            return "winner"
        if ident == self.loser_id:
            return "loser"
        return "other"

    def sex(self, ident: int) -> str:
        """Sex of a track identity, ``"?"`` if it is neither winner nor loser."""
        if ident == self.winner_id:
            return self.winner_sex
        if ident == self.loser_id:
            return self.loser_sex
        return "?"


def read_trial_summary(path: str | Path) -> dict[str, TrialMeta]:
    """Read ``trial_summary.csv`` into a mapping from recording name to meta.

    Parameters
    ----------
    path : str or Path
        The CSV file, or the directory containing it.

    Returns
    -------
    dict of str to TrialMeta
    """
    path = Path(path)
    if path.is_dir():
        path = path / "trial_summary.csv"
    out: dict[str, TrialMeta] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["recording"]] = TrialMeta(
                recording=row["recording"],
                winner_id=round(float(row["win_ID"])),
                loser_id=round(float(row["lose_ID"])),
                winner_sex=row["sex_win"].strip(),
                loser_sex=row["sex_lose"].strip(),
                winner_size_cm=float(row["size_win"]),
                loser_size_cm=float(row["size_lose"]),
            )
    return out


def read_temperatures(
    trial_dir: str | Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Read ``temperatures.csv`` (``time/s; T/C``), if present.

    Returns
    -------
    (times_s, temperature_c) or None
        None if the file does not exist or holds no rows.
    """
    path = Path(trial_dir) / "temperatures.csv"
    if not path.exists():
        return None
    times: list[float] = []
    temps: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = line.split(";")
        if len(parts) != 2:
            continue
        try:
            t, temp = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        times.append(t)
        temps.append(temp)
    if not times:
        return None
    return np.asarray(times), np.asarray(temps)


def open_raw(trial_dir: str | Path) -> Recording:
    """Open ``traces-grid1.raw``, tolerating a trailing partial frame.

    :func:`gridtools.io.open_recording` rejects a raw file whose size is not
    a whole number of frames. Some recordings end in a partial frame (the
    last write was cut short); here the partial frame is dropped with a
    warning.

    Parameters
    ----------
    trial_dir : str or Path
        Fishgrid directory.

    Returns
    -------
    gridtools.io.Recording
        Memory-mapped, read-only.
    """
    trial_dir = Path(trial_dir)
    try:
        return open_recording(trial_dir)
    except ValueError as err:
        if "whole number" not in str(err):
            raise
    config = read_fishgrid_config(trial_dir)
    path = trial_dir / "traces-grid1.raw"
    frame_bytes = 4 * config.n_electrodes
    n_frames = path.stat().st_size // frame_bytes
    logger.warning(
        "%s: dropping %d trailing bytes (partial frame)",
        path,
        path.stat().st_size - n_frames * frame_bytes,
    )
    data = np.memmap(
        path, dtype=np.float32, mode="r", shape=(n_frames, config.n_electrodes)
    )
    return Recording(path, config.samplerate_hz, data, 1.0, config.geometry())


def sign_v_nfft(frame_times_s: npt.ArrayLike, samplerate_hz: float) -> int:
    """FFT length of the spectrogram behind ``sign_v``, from the frame times.

    The tracks were computed with ``matplotlib.mlab.specgram`` (via
    thunderfish), whose frames are centred at ``nfft / 2 + k * hop``
    samples. The first frame time therefore gives ``nfft``.

    Parameters
    ----------
    frame_times_s : array_like
        ``times.npy`` of the tracks.
    samplerate_hz : float
        Sampling rate of the recording.

    Returns
    -------
    int
        A power of two.

    Raises
    ------
    ValueError
        If the first frame time is not half a power-of-two FFT length.
    """
    times = np.asarray(frame_times_s, dtype=np.float64)
    nfft = 2.0 * times[0] * samplerate_hz
    exponent = round(math.log2(nfft)) if nfft > 0 else 0
    if exponent < 4 or abs(nfft - 2**exponent) > 1.0:
        msg = f"first frame at {times[0]} s is not nfft/2 for a power-of-two nfft"
        raise ValueError(msg)
    return int(2**exponent)
