#!/usr/bin/env python

"""Tests `fakegrid` module."""

import numpy as np

from gridtools.fakegrid import (
    band_limited_noise,
    fftnoise,
    get_random_timestamps,
)


def test_fftnoise():
    # Happy path: Test case 1: Test with a simple power spectrum
    f = np.zeros(1000)
    noise = fftnoise(f)
    assert len(noise) == len(f)
    assert np.allclose(np.mean(noise), 0, atol=1e-6)

    # Happy path: Test case 2: Test with a random power spectrum
    f = np.random.rand(1000)
    noise = fftnoise(f)
    assert len(noise) == len(f)
    assert np.allclose(np.mean(noise), 0, atol=1e-3)

    # Sad path: Test case 1: Test with a non-1D array
    f = np.zeros((1000, 1000))
    try:
        noise = fftnoise(f)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_band_limited_noise():
    min_freq = 0.1
    max_freq = 10
    samplerate = 20000
    num_samples = 20 * samplerate
    std = 1

    # Happy path: Test case 1: Test with a simple power spectrum
    noise = band_limited_noise(min_freq, max_freq, num_samples, samplerate, std)
    assert len(noise) == num_samples
    assert np.allclose(np.mean(noise), 0, atol=1e-6)
    assert np.allclose(np.std(noise), std, atol=1e-6)

    # Sad path: Test nyquist mismatch
    samplerate = 100
    max_freq = 50
    try:
        noise = band_limited_noise(
            min_freq, max_freq, num_samples, samplerate, std
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Sad path: Test min_freq > max_freq
    samplerate = 20000
    max_freq = 50
    min_freq = 100
    try:
        noise = band_limited_noise(
            min_freq, max_freq, num_samples, samplerate, std
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Sad path: Test min_freq < 0
    samplerate = 20000
    max_freq = 50
    min_freq = -1
    try:
        noise = band_limited_noise(
            min_freq, max_freq, num_samples, samplerate, std
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_get_random_timestamps():
    # Happy path: Test normal case
    start_t = 0
    end_t = 100
    n_timestamps = 30
    min_dt = 1
    ts = get_random_timestamps(start_t, end_t, n_timestamps, min_dt)
    assert len(ts) == n_timestamps
    assert np.all(ts >= start_t)
    assert np.all(ts <= end_t)
    assert np.all(np.diff(ts) >= min_dt)
    assert np.all(np.diff(ts) <= end_t - start_t)

    # Sad path: Test start_t > end_t
    start_t = 100
    end_t = 0
    try:
        ts = get_random_timestamps(start_t, end_t, n_timestamps, min_dt)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Sad path: Test n_timestamps < 0
    start_t = 0
    end_t = 100
    n_timestamps = -1
    try:
        ts = get_random_timestamps(start_t, end_t, n_timestamps, min_dt)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Sad path: Test min_dt < 0
    start_t = 0
    end_t = 100
    n_timestamps = 30
    min_dt = -1
    try:
        ts = get_random_timestamps(start_t, end_t, n_timestamps, min_dt)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Sad path: Test min_dt > end_t - start_t
    start_t = 0
    end_t = 100
    n_timestamps = 30
    min_dt = 101
    try:
        ts = get_random_timestamps(start_t, end_t, n_timestamps, min_dt)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    # Sad path: Check if number of timestamps is larger than possible
    start_t = 0
    end_t = 100
    n_timestamps = 1000
    min_dt = 1
    try:
        ts = get_random_timestamps(start_t, end_t, n_timestamps, min_dt)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
