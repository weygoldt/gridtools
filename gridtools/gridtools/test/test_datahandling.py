import numpy as np

from ..toolbox import datahandling


def test_findClosest_inrange() -> None:

    array = np.arange(-100, 100)
    targets = [-50, 0, 50, 50.3, 50.7]
    index = np.arange(len(array))
    solutions = [index[array == x][0] for x in np.round(targets)]

    for target, solution in zip(targets, solutions):
        x = datahandling.findClosest(array, target)
        assert x == solution


def test_findClosest_outofrange() -> None:

    array = np.arange(-100, 100)
    targets = [-110, 101]
    solutions = [0, 199]

    for target, solution in zip(targets, solutions):
        x = datahandling.findClosest(array, target)
        assert x == solution


def test_findOnTime_inrange() -> None:

    time = np.arange(-10, 100, 0.1)
    targets = -10, -2.5234, 0, 99.9
    solutions = [0, 75, 100, 1099]

    for target, solution in zip(targets, solutions):
        x = datahandling.findOnTime(time, target, limit=True)
        assert x == solution


def test_findOnTime_outofrange() -> None:

    time = np.arange(-10, 100, 0.1)
    targets = -12, 101

    for target in targets:
        try:
            x = datahandling.findOnTime(time, target, limit=True)
            assert False
        except Exception:
            assert True
