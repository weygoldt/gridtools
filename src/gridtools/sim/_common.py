"""Shared type aliases and argument checks for the simulator (internal)."""

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


def require_positive(value: float, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is finite and > 0."""
    if not (math.isfinite(value) and value > 0):
        msg = f"{name} must be finite and > 0, got {value!r}"
        raise ValueError(msg)


def require_nonnegative(value: float, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is finite and >= 0."""
    if not (math.isfinite(value) and value >= 0):
        msg = f"{name} must be finite and >= 0, got {value!r}"
        raise ValueError(msg)


def require_probability(value: float, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` lies in [0, 1]."""
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        msg = f"{name} must lie in [0, 1], got {value!r}"
        raise ValueError(msg)


def require_n_samples(value: int, name: str = "n_samples") -> None:
    """Raise ``ValueError`` unless ``value`` is an integer >= 1."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        msg = f"{name} must be an integer >= 1, got {value!r}"
        raise ValueError(msg)


def as_1d_float64(values: npt.ArrayLike, name: str) -> FloatArray:
    """Convert to a finite 1-D float64 array or raise ``ValueError``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        msg = f"{name} must be a non-empty 1-D array, got shape {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} contains non-finite values"
        raise ValueError(msg)
    return array


def as_points_m(values: npt.ArrayLike, name: str) -> FloatArray:
    """Convert to a finite (n, 2) float64 array of x, y coordinates."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        msg = f"{name} must have shape (n, 2) with n >= 1, got {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} contains non-finite values"
        raise ValueError(msg)
    return array
