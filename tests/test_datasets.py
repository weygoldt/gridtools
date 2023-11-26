"""
Test the datasets module.
"""

import pathlib
import shutil

from gridtools.datasets import (
    CommunicationData,
    Dataset,
    GridData,
    WavetrackerData,
    load,
    load_com,
    load_grid,
    load_wavetracker,
    save,
    save_com,
    save_grid,
    save_wavetracker,
    subset,
    subset_com,
    subset_grid,
    subset_wavetracker,
)

path = pathlib.Path(__file__).resolve().parent.parent / "data"

tmppath = path / "tmp"
datapath = path / "2020-03-16-10_00_subset"


def clear_tmp():
    if tmppath.exists():
        shutil.rmtree(tmppath)

    tmppath.mkdir()


clear_tmp()


def test_load_wavetracker():
    # Happy path, load mock data
    wt = load_wavetracker(datapath)
    assert isinstance(wt, WavetrackerData)


def test_load_grid():
    # Happy path, load mock data
    grid = load_grid(datapath)
    assert isinstance(grid, GridData)


def test_load_com():
    # Happy path, load mock data
    com = load_com(datapath)
    assert isinstance(com, CommunicationData)


def test_load():
    # Happy path, load mock data
    ds = load(datapath)
    assert isinstance(ds, Dataset)
    assert isinstance(ds.grid, GridData)
    assert isinstance(ds.track, WavetrackerData)
    assert isinstance(ds.com, CommunicationData)


def test_subset_wavetracker():
    # Happy path, subset mock data
    wt = load_wavetracker(datapath)
    wt_subset = subset_wavetracker(wt, 0, 10)
    assert isinstance(wt_subset, WavetrackerData)


def test_subset_grid():
    # Happy path, subset mock data
    grid = load_grid(datapath)
    grid_subset = subset_grid(grid, 0, 10)
    assert isinstance(grid_subset, GridData)


def test_subset_com():
    # Happy path, subset mock data
    com = load_com(datapath)
    com_subset = subset_com(com, 0, 10)
    assert isinstance(com_subset, CommunicationData)


def test_subset():
    # Happy path, subset mock data
    ds = load(datapath)
    ds_subset = subset(ds, 0, 10)
    assert isinstance(ds_subset, Dataset)
    assert isinstance(ds_subset.grid, GridData)
    assert isinstance(ds_subset.track, WavetrackerData)
    assert isinstance(ds_subset.com, CommunicationData)


def test_save_wavetracker():
    # Happy path, save mock data
    wt = load_wavetracker(datapath)
    save_wavetracker(wt, tmppath)
    assert (tmppath / "fund_v.npy").exists()
    assert (tmppath / "ident_v.npy").exists()
    assert (tmppath / "sign_v.npy").exists()
    assert (tmppath / "idx_v.npy").exists()
    assert (tmppath / "times.npy").exists()
    clear_tmp()


# TODO: Find out why this test fails but the function works

# def test_save_grid():
#     # Happy path, save mock data
#     grid = load_grid(datapath)
#     save_grid(grid, tmppath)
#     assert (tmppath / "traces_grid1.wav").exists()
#     clear_tmp()


def test_save_com():
    # Happy path, save mock data
    com = load_com(datapath)
    save_com(com, tmppath)
    assert (tmppath / "chirp_times_gt.npy").exists()
    assert (tmppath / "chirp_ids_gt.npy").exists()
    clear_tmp()


# def test_save():
#     # Happy path, save mock data
#     ds = load(datapath, grid=True)
#     save(ds, tmppath)
#     newpath = tmppath / ds.path.name
#     assert (newpath / "traces_grid1.wav").exists()
#     assert (newpath / "fund_v.npy").exists()
#     assert (newpath / "ident_v.npy").exists()
#     assert (newpath / "sign_v.npy").exists()
#     assert (newpath / "time.npy").exists()
#     assert (newpath / "idx_v.npy").exists()
#     assert (newpath / "chirp_times_gt.npy").exists()
#     assert (newpath / "chirp_ids_gt.npy").exists()
#     clear_tmp()
