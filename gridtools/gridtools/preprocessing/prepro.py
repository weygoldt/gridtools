"""
.
├── dataroot/            >> REQUIRED <<
│   │
│   ├── datacleaner_conf.yml
│   ├── hobologger.csv
│   │
│   ├── recording1/
│   │       fund_v.npy
│   │       ident_v.npy ...
│   ├── recording2/ more data
│   └── ...
│
└── output/            >> WILL BE CREATED <<
    ├── recording1/
    │       fund_v.npu
    │       temp.npy
    │       xpos.npy ... 
    ├── recording2/
    └── ...
"""
import argparse
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

from ..logger import makeLogger
from ..utils.filehandling import ConfLoader, ListRecordings, makeOutputdir
from .gridcleaner import GridCleaner

logger = makeLogger(__name__)


def plot_grid(grid: GridCleaner) -> None:

    fig, ax = plt.subplots(1, 2, constrained_layout=True)

    for track_id, sex in zip(grid.ids, grid.sex):

        time = grid.times[grid.indices[grid.identities == track_id]]
        fund = grid.frequencies[grid.identities == track_id]

        ax[0].plot(time, fund, alpha=1)
        ax[0].annotate(f"{int(track_id)} {sex}", xy=(time[0], fund[0]))

        xpos = grid.xpositions[grid.identities == track_id]
        ypos = grid.ypositions[grid.identities == track_id]

        ax[1].plot(xpos, ypos, alpha=1)
        ax[1].annotate(f"{int(track_id)} {sex}", xy=(xpos[0], ypos[0]))

        # plot a random subset that is more visible
        # ndata = 10 * 60 * 3  # approx 10 min with 3 Hz sampling
        # index = np.arange(len(time))[ndata:-ndata]  # possible indices to choose from
        # start = np.random.choice(index)
        # stop = start + ndata
        # ax[0].plot(time[start:stop], fund[start:stop])
        # ax[1].plot(xpos[start:stop], ypos[start:stop])

    ax[0].set_title("Frequency tracks")
    ax[1].set_title("Estimated positions")

    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Frequency [Hz]")

    ax[1].set_xlabel("x position [cm]")
    ax[1].set_ylabel("y position [cm]")

    plt.show()


def clean(path: str) -> None:

    try:
        confpath = os.path.join(path, "prepro_conf.yml")
        conf = ConfLoader(confpath)

    except FileNotFoundError as error:
        msg = "No prepro_conf.yml found in specified dir! Change dir or run datacleaner init!"
        logger.error(msg)
        raise error(msg)

    dataroot = conf.data_path
    exclude = conf.exclude_directories

    recs = ListRecordings(dataroot, exclude=exclude)

    if len(conf.include_only_directories) > 0:
        recs.recordings = conf.include_only_directories

    for recording in recs.recordings:

        # create path to recording
        datapath = f"{recs.dataroot}{recording}/"

        # read output path from config
        outpath = f"{conf.output_path}output/"

        # create output directory
        makeOutputdir(f"{conf.output_path}output/")  # make parent

        # load recording data
        grid = GridCleaner(datapath)

        # recompute powers for tracks added manually using wavetracker GUI
        grid.recompute_powers()

        # load hobologger
        grid.load_logger(conf.logger_name)

        # remove unassigned frequencies in dataset
        grid.purge_unassigned()

        # remove short tracks
        grid.purge_short(conf.duration_threshold)

        # remove poorly tracked
        grid.purge_bad(conf.performance_threshold)

        # compute positions
        grid.triangulate_positions(conf.number_electrodes)

        # interpolate
        grid.interpolate_all()

        # compute individual q10 values
        grid.compute_q10()

        # sex ids
        grid.sex_fish(*conf.sexing_parameters.values())

        # smooth positions
        grid.smooth_positions(conf.smoothing_parameters)

        # plot
        if conf.plot:
            plot_grid(grid)

        # save if not dry run
        if not conf.dry_run:
            grid.save_data(outpath, conf.filename)


def main() -> None:

    parser = argparse.ArgumentParser(
        prog="prepro",
        description="Terminal interface to preprocess tracked electrode grid recordings.",
    )

    parser.add_argument(
        "mode",
        type=str,
        nargs="?",
        help="'init' to copy a config file to the specified directory, 'edit' to edit the config file, 'run' to run the datacleaner.",
    )

    parser.add_argument(
        "dir",
        nargs="?",
        type=str,
        default=os.getcwd(),
        help="The directory to copy the config file to.",
    )

    args = parser.parse_args()
    pkgdir = os.path.dirname(sys.modules["gridtools"].__path__[0])
    configfile = os.path.join(pkgdir, "data/prepro_conf.yml")
    destination = os.path.join(args.dir, "prepro_conf.yml")

    # copy config file if "init"
    if args.mode == "init":
        shutil.copy(configfile, destination)
        print("------------------------")
        print("prepro_config.yml copied to directory!")
        print(
            "Please add the nessecary data before running <prepro --mode run --dir .> to start preprocessing."
        )
        print("------------------------")

    # run the datacleaner if "run"
    elif args.mode == "run":
        print("------------------------")
        print("Starting preprocessing. This will take some time.")
        print("------------------------")
        clean(args.dir)


if __name__ == "__main__":
    main()
