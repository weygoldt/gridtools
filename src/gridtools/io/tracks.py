"""EOD frequency tracks in wavetracker's on-disk format.

A wavetracker result directory holds five arrays:

``times.npy`` (T,)
    Time of every spectrogram frame in seconds.
``fund_v.npy`` (N,)
    Fundamental frequency of every detection in Hz.
``ident_v.npy`` (N,)
    Track identity of every detection; NaN for unassigned detections.
``idx_v.npy`` (N,)
    Index into ``times`` of every detection.
``sign_v.npy`` (N, E)
    Power of every detection on each electrode in dB.

``sign_v`` may have fewer columns than the recording has channels, so the
electrode axis of the powers is not guaranteed to match the recording's
channels. In all 2019 tube competition recordings raw channel 9 carries no
fish signal and ``sign_v`` holds channels 1-8 only; in 2020-05-11 to 05-14
channel 2 is dead as well (verified 2026-11-12 by ``gridtools.calibrate``).
``sign_v`` is 10 log10 of the one-sided Hann power spectral density at the
bin nearest the fundamental (nfft 32768; 16384 in March 2020).

A track exists only between its first and last detection, and only where
its detections are not too far apart: :meth:`Track.frequency_at` returns NaN
elsewhere instead of extrapolating (v1 matched chirps to tracks that had
ended, audit item D5).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["Track", "Tracks", "load_tracks", "save_tracks"]

_FILES = ("times", "fund_v", "ident_v", "idx_v", "sign_v")


def _frozen(array: npt.ArrayLike, dtype: npt.DTypeLike) -> np.ndarray:
    out = np.array(array, dtype=dtype)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Track:
    """The detections of one fish, sorted by time.

    Parameters
    ----------
    ident
        Track identity.
    times_s
        Detection times, strictly increasing.
    freqs_hz
        Fundamental frequency of each detection.
    powers_db
        Per-electrode power of each detection, shape ``(n, n_electrodes)``,
        or None.
    """

    ident: int
    times_s: npt.NDArray[np.float64]
    freqs_hz: npt.NDArray[np.float64]
    powers_db: npt.NDArray[np.float64] | None = None

    @property
    def start_s(self) -> float:
        """Time of the first detection."""
        return float(self.times_s[0])

    @property
    def stop_s(self) -> float:
        """Time of the last detection."""
        return float(self.times_s[-1])

    def _interpolation(
        self, t_s: npt.ArrayLike, max_gap_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
        right = np.searchsorted(self.times_s, t, side="left")
        right = np.clip(right, 0, len(self.times_s) - 1)
        left = np.clip(right - 1, 0, len(self.times_s) - 1)
        exact = self.times_s[right] == t
        left = np.where(exact, right, left)
        t_left, t_right = self.times_s[left], self.times_s[right]
        inside = (t >= self.times_s[0]) & (t <= self.times_s[-1])
        valid = inside & (t_right - t_left <= max_gap_s)
        span = np.where(t_right > t_left, t_right - t_left, 1.0)
        weight = np.where(exact, 0.0, (t - t_left) / span)
        return left, right, np.clip(weight, 0.0, 1.0), valid

    def frequency_at(
        self, t_s: npt.ArrayLike, max_gap_s: float = 1.0
    ) -> npt.NDArray[np.float64]:
        """Frequency at arbitrary times, linearly interpolated.

        Parameters
        ----------
        t_s
            Times in seconds.
        max_gap_s
            Interpolate only between detections at most this far apart.

        Returns
        -------
        Frequencies with the shape of ``np.atleast_1d(t_s)``; NaN where the
        track does not exist (before its start, after its end, or inside a
        gap longer than ``max_gap_s``).
        """
        left, right, weight, valid = self._interpolation(t_s, max_gap_s)
        f = (1 - weight) * self.freqs_hz[left] + weight * self.freqs_hz[right]
        return np.where(valid, f, np.nan)

    def powers_at(
        self, t_s: npt.ArrayLike, max_gap_s: float = 1.0
    ) -> npt.NDArray[np.float64]:
        """Per-electrode power at arbitrary times (dB, linear interpolation).

        Returns
        -------
        Array of shape ``(len(t), n_electrodes)``, NaN where the track does
        not exist.

        Raises
        ------
        ValueError
            If the track has no powers.
        """
        if self.powers_db is None:
            msg = f"track {self.ident} has no powers"
            raise ValueError(msg)
        left, right, weight, valid = self._interpolation(t_s, max_gap_s)
        w = weight[:, None]
        p = (1 - w) * self.powers_db[left] + w * self.powers_db[right]
        p[~valid] = np.nan
        return p


@dataclass(frozen=True)
class Tracks:
    """All detections of a recording, in wavetracker layout.

    Arrays are read-only. See the module docstring for their meaning.
    """

    times_s: npt.NDArray[np.float64]
    freqs_hz: npt.NDArray[np.float64]
    idents: npt.NDArray[np.float64]
    time_indices: npt.NDArray[np.int64]
    powers_db: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "times_s", _frozen(self.times_s, np.float64))
        object.__setattr__(self, "freqs_hz", _frozen(self.freqs_hz, np.float64))
        object.__setattr__(self, "idents", _frozen(self.idents, np.float64))
        object.__setattr__(self, "time_indices", _frozen(self.time_indices, np.int64))
        if self.powers_db is not None:
            object.__setattr__(self, "powers_db", _frozen(self.powers_db, np.float64))
        self.validate()

    def validate(self) -> None:
        """Check that the arrays are consistent.

        Raises
        ------
        ValueError
            On mismatched lengths, indices outside ``times``, decreasing
            ``times``, or non-finite frequencies of assigned detections.
        """
        n = len(self.freqs_hz)
        for name in ("idents", "time_indices"):
            if len(getattr(self, name)) != n:
                msg = f"{name} has {len(getattr(self, name))} entries, fund_v has {n}"
                raise ValueError(msg)
        if self.powers_db is not None and (
            self.powers_db.ndim != 2 or len(self.powers_db) != n
        ):
            msg = f"powers_db must have shape ({n}, n_electrodes), got {self.powers_db.shape}"
            raise ValueError(msg)
        if n and (
            self.time_indices.min() < 0 or self.time_indices.max() >= len(self.times_s)
        ):
            msg = "time_indices point outside times"
            raise ValueError(msg)
        if np.any(np.diff(self.times_s) < 0):
            msg = "times must not decrease"
            raise ValueError(msg)
        assigned = np.isfinite(self.idents)
        if not np.all(np.isfinite(self.freqs_hz[assigned])):
            msg = "assigned detections must have finite frequencies"
            raise ValueError(msg)

    @property
    def n_electrodes(self) -> int | None:
        """Number of electrode columns in the powers, if present."""
        return None if self.powers_db is None else int(self.powers_db.shape[1])

    @property
    def ids(self) -> npt.NDArray[np.int64]:
        """Sorted identities of all tracks."""
        ids = np.unique(self.idents[np.isfinite(self.idents)])
        return ids.astype(np.int64)

    @cached_property
    def _tracks(self) -> dict[int, Track]:
        assigned = np.flatnonzero(np.isfinite(self.idents))
        order = assigned[
            np.lexsort((self.time_indices[assigned], self.idents[assigned]))
        ]
        ids, starts = np.unique(self.idents[order], return_index=True)
        return {
            int(ident): self._make_track(int(ident), rows)
            for ident, rows in zip(ids, np.split(order, starts[1:]), strict=True)
        }

    def track(self, ident: int) -> Track:
        """The detections of one track, sorted by time.

        Detections of the same track at the same frame are averaged, so the
        track's times are strictly increasing. Tracks are built once, on
        first access.

        Raises
        ------
        KeyError
            If there is no track ``ident``.
        """
        return self._tracks[int(ident)]

    def _make_track(self, ident: int, rows: npt.NDArray[np.intp]) -> Track:
        frames, inverse = np.unique(self.time_indices[rows], return_inverse=True)
        counts = np.bincount(inverse).astype(np.float64)
        freqs = np.bincount(inverse, weights=self.freqs_hz[rows]) / counts
        powers = None
        if self.powers_db is not None:
            powers = np.stack(
                [
                    np.bincount(inverse, weights=self.powers_db[rows, e]) / counts
                    for e in range(self.powers_db.shape[1])
                ],
                axis=1,
            )
        return Track(ident, self.times_s[frames], freqs, powers)

    def __iter__(self) -> Iterator[Track]:
        """Iterate over all tracks in order of identity."""
        yield from self._tracks.values()

    def frequencies_at(self, t_s: float, max_gap_s: float = 1.0) -> dict[int, float]:
        """Frequencies of the tracks that exist at time ``t_s``.

        Returns
        -------
        Mapping from identity to interpolated frequency, for every track
        whose :meth:`Track.frequency_at` is defined at ``t_s``.
        """
        out: dict[int, float] = {}
        for track in self:
            f = track.frequency_at(t_s, max_gap_s)[0]
            if np.isfinite(f):
                out[track.ident] = float(f)
        return out


def load_tracks(directory: str | Path) -> Tracks:
    """Load wavetracker results from a directory.

    Parameters
    ----------
    directory
        Directory with ``times.npy``, ``fund_v.npy``, ``ident_v.npy``,
        ``idx_v.npy`` and, optionally, ``sign_v.npy``.

    Raises
    ------
    FileNotFoundError
        If a required file is missing.
    ValueError
        If the arrays are inconsistent.
    """
    directory = Path(directory)
    arrays = {}
    for name in _FILES:
        path = directory / f"{name}.npy"
        if not path.exists():
            if name == "sign_v":
                continue
            raise FileNotFoundError(path)
        arrays[name] = np.load(path)
    return Tracks(
        times_s=arrays["times"],
        freqs_hz=arrays["fund_v"],
        idents=arrays["ident_v"],
        time_indices=arrays["idx_v"],
        powers_db=arrays.get("sign_v"),
    )


def save_tracks(tracks: Tracks, directory: str | Path) -> None:
    """Write tracks in wavetracker layout (overwrites existing files)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "times.npy", tracks.times_s)
    np.save(directory / "fund_v.npy", tracks.freqs_hz)
    np.save(directory / "ident_v.npy", tracks.idents)
    np.save(directory / "idx_v.npy", tracks.time_indices)
    if tracks.powers_db is not None:
        np.save(directory / "sign_v.npy", tracks.powers_db)
