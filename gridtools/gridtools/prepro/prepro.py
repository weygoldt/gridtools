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

from ..logger import makeLogger
from ..toolbox.filehandling import ConfLoader, ListRecordings, makeOutputdir
from .gridcleaner import GridCleaner

logger = makeLogger(__name__)

def plot_preview(grid: GridCleaner) -> None:

    fig, ax = plt.subplots(1,2, constrained_layout=True)

    for track_id in grid.ids:
        
        time = grid.times[grid.idx_v[grid.ident_v == track_id]]
        fund = grid.fund_v[grid.ident_v == track_id]
        
        ax[0].plot(time, fund)

        xpos = grid.xpos[grid.ident_v == track_id]
        ypos = grid.ypos[grid.ident_v == track_id]

        ax[1].plot(xpos, ypos, alpha=0.5)
    
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

    dataroot = conf.data
    exclude = conf.exclude

    recs = ListRecordings(dataroot, exclude=exclude)

    if len(conf.include_only) > 0: recs.recordings = conf.include_only

    for recording in recs.recordings:

        # create path to recording
        datapath = f"{recs.dataroot}{recording}/"

        # read output path from config
        outpath = f"{conf.output}{recording}"

        # create output directory
        makeOutputdir(conf.output) # make parent
        makeOutputdir(outpath) # make rec dir

        # load recording data
        grid = GridCleaner(datapath)

        # recompute powers for tracks added manually using wavetracker GUI
        grid.fillPowers()

        # load hobologger
        grid.loadLogger(conf.logger_name)

        # remove unassigned frequencies in dataset
        if conf.purge_unassigned: grid.purgeUnassigned()

        # remove short tracks 
        if conf.purge_short: grid.purgeShort(conf.dur_thresh)

        # remove poorly tracked
        if conf.purge_bad: grid.purgeBad(conf.perf_thresh)

        # compute positions
        grid.triangPositions(conf.num_el)
        
        # interpolate
        grid.interpolateAll()

        # sex ids
        grid.sexFish(*conf.sex_params.values())

        # smooth positions
        if conf.smth_pos: grid.smoothPositions(conf.smth_params)

        if conf.plot: plot_preview(grid)

        if not conf.dry_run: grid.saveData(outpath)

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
    pkgdir = os.path.dirname(sys.modules['gridtools'].__path__[0])
    configfile = os.path.join(pkgdir, "data/prepro_conf.yml")
    destination = os.path.join(args.dir, "prepro_conf.yml")

    # copy config file if "init"
    if args.mode == "init":
        shutil.copy(configfile, destination)
        print("------------------------")
        print("prepro_config.yml copied to directory!")
        print("Please add the nessecary data before running <prepro --mode run --dir .> to start preprocessing.")
        print("------------------------")

    # run the datacleaner if "run"
    elif args.mode == "run":
        clean(args.dir)
        print("------------------------")
        print("Starting preprocessing. This will take some time.")
        print("------------------------")

if __name__ == "__main__":
    main()



