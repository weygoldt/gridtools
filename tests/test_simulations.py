#!/usr/bin/env python

"""Tests `simulations` module."""

import matplotlib.pyplot as plt
import numpy as np

from gridtools.simulations import (
    MovementParams,
    direction_pdf,
    fold_space,
    gaussian,
    interpolate_positions,
    make_chirps,
    make_grid,
    make_positions,
    make_rises,
    make_steps,
    step_pdf,
)


def test_direction_pdf():
    """Test the function that generates a pdf for the directions of
    a moving fish at a point in time."""

    # Happy path: Test case 1: Test with a single direction
    fs = 10000
    dirs, probs = direction_pdf(
        forward_s=1,
        backward_s=1,
        backward_h=0,
        fs=fs,
    )
    expected_len = int(fs * 2 * np.pi) + 1
    expected_min = 0
    expected_max = 2 * np.pi

    assert len(dirs) == expected_len
    assert len(dirs) == len(probs)
    assert np.min(dirs) >= expected_min
    assert np.max(dirs) <= expected_max
    assert np.isclose(np.sum(probs), 1, atol=1e-6)
    assert np.allclose(np.mean(probs), 1 / expected_len, atol=1e-6)

    # Happy path: Test case 2: Test with backward_h > 0
    dirs, probs = direction_pdf(
        forward_s=1,
        backward_s=1,
        backward_h=1,
        fs=fs,
    )
    assert len(dirs) == expected_len
    assert len(dirs) == len(probs)
    assert np.min(dirs) >= expected_min
    assert np.max(dirs) <= expected_max
    assert np.isclose(np.sum(probs), 1, atol=1e-6)
    assert np.allclose(np.mean(probs), 1 / expected_len, atol=1e-6)

    # Sad path: Test case 1: Test with backward_s < 0
    try:
        dirs, probs = direction_pdf(
            forward_s=1,
            backward_s=-1,
            backward_h=0,
            fs=fs,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError")

    # Sad path: Test case 2: Test with forward_s < 0
    try:
        dirs, probs = direction_pdf(
            forward_s=-1,
            backward_s=1,
            backward_h=0,
            fs=fs,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError")

    # Sad path: Test case 3: Test with backward_h < 0
    try:
        dirs, probs = direction_pdf(
            forward_s=1,
            backward_s=1,
            backward_h=-1,
            fs=fs,
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError")


def test_step_pdf():
    """Test generating gamma distributed step lengths of a moving fish."""

    # Happy path: Test case 1: Test with usual parameters
    max_veloc = 1  # m/s
    duration = 6000  # s
    target_fs = 30  # Hz
    g = step_pdf(max_veloc, duration, target_fs)

    assert len(g) == int(duration * target_fs) - 1
    assert np.all(g >= 0)
    assert np.all(g <= max_veloc)

    # Sad path: Test case 1: Test with max_veloc < 0
    try:
        g = step_pdf(-1, duration, target_fs)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError")

    # Sad path: Test case 2: Test with duration < 0
    try:
        g = step_pdf(max_veloc, -1, target_fs)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError")

    # Sad path: Test case 3: Test with target_fs < 0
    try:
        g = step_pdf(max_veloc, duration, -1)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError")


def test_make_steps():
    """Test the generation of trajectories and steps."""

    # Happy path: Test case 1: Test with usual parameters
    mvm = MovementParams()
    traj, steps = make_steps(mvm)
    assert len(traj) == (mvm.duration * mvm.measurement_fs) - 1
    assert len(traj) == len(steps)
    assert np.all(traj >= 0)


def test_make_positions():
    """Test the generation of x and y coordinates from steps and
    trajectories.
    """

    mvm = MovementParams()
    t, s = make_steps(mvm)

    # Happy path: Test case 1: Test with usual parameters
    x, y = make_positions(t, s, origin=(0, 0))
    assert len(x) == len(y)
    assert len(x) == len(t) + 1
    assert x[0] == 0
    assert y[0] == 0

    # Happy path: Test case 2: Test with origin (-5, -5)
    x, y = make_positions(t, s, origin=(-5, -5))
    assert len(x) == len(y)
    assert len(x) == len(t) + 1
    assert x[0] == -5
    assert y[0] == -5

    # Sad path: Test case 1: Test with t and s of different lengths
    try:
        x, y = make_positions(t[:-1], s, origin=(0, 0))
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError")


def test_interpolate_positions():
    """Test the interpolation of the positions to a target frequency."""

    mvm = MovementParams(target_fs=20000)
    t, s = make_steps(mvm)
    x, y = make_positions(t, s, origin=(0, 0))

    # Happy path: Test case 1: Test with usual parameters
    x, y = interpolate_positions(
        x, y, mvm.duration, mvm.measurement_fs, mvm.target_fs
    )
    assert len(x) == len(y)
    assert len(x) == int(mvm.duration * mvm.target_fs)
    assert x[0] == 0
    assert y[0] == 0

    # Sad path: Test case 1: Test with x and y of different lengths
    try:
        x, y = interpolate_positions(
            x[:-1], y, mvm.duration, mvm.measurement_fs, mvm.target_fs
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError")


def test_fold_space():
    """Test folding the virtual 2D space."""

    mvm = MovementParams()
    t, s = make_steps(mvm)
    x, y = make_positions(t, s, origin=(0, 0))
    x, y = interpolate_positions(
        x, y, mvm.duration, mvm.measurement_fs, mvm.target_fs
    )

    # Happy path: Test case 1: Test with usual parameters
    x, y = fold_space(x, y, (-0.5, -0.5, 0.5, 0.5))
    assert len(x) == len(y)
    assert len(x) == int(mvm.duration * mvm.target_fs)
    assert np.all(x >= -0.5)
    assert np.all(x <= 0.5)
    assert np.all(y >= -0.5)
    assert np.all(y <= 0.5)

    # Sad path: Test case 1: Test with x and y of different lengths
    try:
        x, y = fold_space(x[:-1], y, (-0.5, -0.5, 0.5, 0.5))
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError")

    # Sad path: Boundaries are to narrow
    boundaries = (-0.00001, -0.00001, 0.00001, 0.00001)
    try:
        x, y = fold_space(x, y, boundaries)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
