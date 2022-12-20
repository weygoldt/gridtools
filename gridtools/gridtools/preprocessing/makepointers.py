import datetime

import nixio as nio
import numpy as np
from IPython import embed

from gridtools.utils.datahandling import estimateMode

class NixGridRecording:
    """
    Loads a single recording from a .nix file containing multiple recordings.
    """

    def __init__(self, block: nio.Block) -> None:

        # read data arrays
        dt_format = "%Y-%m-%d %H:%M:%S"
        self.starttime = datetime.datetime.strptime(block.name, dt_format)
        self.times = block.data_arrays["times"][:]
        self.frequencies = block.data_arrays["frequencies"][:]
        self.identities = block.data_arrays["identities"][:]
        self.indices = block.data_arrays["indices"][:]
        self.xpositions = block.data_arrays["xpositions"][:]
        self.ypositions = block.data_arrays["ypositions"][:]
        self.temperature = block.data_arrays["temperature"][:]
        self.light = block.data_arrays["light"][:]
        self.ids = block.data_arrays["ids"][:]
        self.sex = block.data_arrays["sex"][:]
        self.q10 = block.data_arrays["q10"][:]
        self.samplingrate = np.mean(np.diff(self.times))
        self.name = block.name

    def __repr__(self) -> str:
        return "NixGridRecording({})".format(
            self.starttime.strftime("%Y-%m-%d %H:%M:%S")
        )

    def __str__(self) -> str:
        return "Single recording at {}".format(
            self.starttime.strftime("%Y-%m-%d %H:%M:%S")
        )


class ConnectFish:
    """
    Loads data arrays of all recordings from a .nix grid recording file. Utilized
    to create pointers between recordings that group frequency tracks into the same fish.
    """

    def __init__(self, filepath: str, filemode: str = "ReadOnly") -> None:

        # check if filemode is usable
        assert filemode in [
            "ReadOnly",
            "ReadWrite",
        ], f"Filemode can be ReadOnly or ReadWrite, you supplied {filemode}!"

        # set file mode
        if filemode == "ReadOnly":
            filemode = nio.FileMode.ReadOnly
        else:
            filemode = nio.FileMode.ReadWrite

        # open file
        self._filepath = filepath
        file = nio.File.open(filepath, filemode)

        # load all recordings in grid
        self.recordings = []
        for block in file.blocks:
            self.recordings.append(NixGridRecording(block))

        self.samplingrate = np.round(
            np.mean([rec.samplingrate for rec in self.recordings]), 4
        )

    def __repr__(self) -> str:
        return "NixGrid({})".format(self._filepath)

    def __str__(self) -> str:
        return "Grid recording set at {}".format(self._filepath)

    def create_pointers(self) -> None:
        def get_pairs(matrix):

            # get indices of pairs sorted by difference in ascending order
            pairs = np.unravel_index(np.argsort(matrix, axis=None), np.shape(matrix))

            # build matrix to store used indices to not use the same twice
            done = np.ones_like(matrix, dtype=bool)

            # go through pairs and append pairs if not used already
            track_rec1 = []
            track_rec2 = []

            for i in range(len(pairs[0])):

                pair_idx = np.asarray([pairs[0][i], pairs[1][i]])

                if done[pair_idx[0], pair_idx[1]] == True:

                    print(pair_idx)
                    print(done)

                    track_rec1.append(pair_idx[0])
                    track_rec2.append(pair_idx[1])

                    done[pair_idx[0], :] = False
                    done[:, pair_idx[1]] = False

                else:
                    print(done)
                    continue

            return track_rec1, track_rec2

        # set thresholds
        dt_between_recordings = int(15 * 60)  # gap threshold between recordings
        dt_to_end = int(120)  # threshold for gap between track end and recording end
        dt_baseline = int(
            np.round(1800 * self.samplingrate)
        )  # window to compute baseline for pointing

        # Get start and end points of all recordings
        pointers = [] # the newly generated pointers go here
        previous_pointers = [] # save previous pointers in this buffer
        track_ids = [] # the old track ids go here
        previous_matches_rec2 = [] # save previous pairs in this buffer
        rec_names = [] # the recording names go here
        for rec1, rec2 in zip(self.recordings[:-1], self.recordings[1:]):

            # get time distance between recordings
            stop1 = datetime.timedelta(0, rec1.times[-1]) + rec1.starttime
            start2 = rec2.starttime
            gap = (start2 - stop1).total_seconds()

            # check if time gap is small enough
            if gap > dt_between_recordings:
                continue

            # get stops of rec1
            ids1 = []
            baseline1 = []

            for track_id in rec1.ids:

                track_id = int(track_id)

                # check if id has data at end of recording
                track_stoptime = rec1.times[rec1.indices[rec1.identities == track_id]][
                    -1
                ]
                abs_stoptime = rec1.times[-1]
                diff_time = abs_stoptime - track_stoptime
                if diff_time < dt_to_end:
                    continue

                # get data at end of recording
                ids1.append(int(track_id))
                baseline1.append(
                    estimateMode(
                        rec1.frequencies[rec1.identities == track_id][-dt_baseline:]
                    )
                )

            # get starts of rec2
            ids2 = []
            baseline2 = []

            for track_id in rec2.ids:

                track_id = int(track_id)

                # check if id has data at end of recording
                track_stoptime = rec2.times[rec2.indices[rec2.identities == track_id]][
                    -1
                ]
                abs_stoptime = rec2.times[-1]
                diff_time = abs_stoptime - track_stoptime
                if diff_time < dt_to_end:
                    continue

                # get data at start of recording
                ids2.append(track_id)
                baseline2.append(
                    estimateMode(
                        rec2.frequencies[rec2.identities == track_id][-dt_baseline:]
                    )
                )

            # convert to numpy arrays
            ids1 = np.asarray(ids1)
            baseline1 = np.asarray(baseline1)
            ids2 = np.asarray(ids2)
            baseline2 = np.asarray(baseline2)

            # compute difference matrix
            diff_matrix = np.asarray([baseline1 - x for x in baseline2])

            # get indices for unique pairs with minimum difference frequency
            idxs1, idxs2 = get_pairs(diff_matrix)

            # get ids for these indices
            matches_rec1 = ids1[idxs2]
            matches_rec2 = ids2[idxs1]

            # create new meta ids
            lastmax = 0
            if len(previous_pointers) != 0:
                lastmax = np.max(previous_pointers) 
            pointers_tmp = np.arange(lastmax + 1 , len(matches_rec1) + 1)

            # reshape matching ids
            matches = [[x, y] for x, y in zip(matches_rec1, matches_rec2)]

            # make recording references
            recording_ref = [[rec1.name, rec2.name] for i in range(len(matches))]

            # check if track ids from currently first recording are already referenced in the previously last recording
            
            # step 1: get ids in current rec1 that where already in previous rec2
            repeated_matches = [x for x in matches_rec1 if x in previous_matches_rec2] # get repeated
            
            if len(repeated_matches) != 0:

                # step 2: find pointers of these cases
                indices = np.arange(len(previous_matches_rec2))
                rep_indices = np.asarray([indices[previous_matches_rec2 == x] for x in repeated_matches])
                print(rep_indices)
                rep_pointers = np.asarray(previous_pointers)[rep_indices]

                # step 3: swap the newly created pointers for the previous pointers
                indices = np.arange(len(matches_rec1))
                rep_indices = np.asarray([indices[matches_rec1 == x] for x in repeated_matches])
                np.asarray(pointers_tmp)[rep_indices] = rep_pointers

            # step 4: append all to lists
            pointers.extend(pointers_tmp)
            track_ids.extend(matches)
            rec_names.extend(recording_ref)
            
            # step 5: save data from current iteration to buffer to compare in nex iteration
            previous_poiners = pointers_tmp 
            previous_matches_rec2 = matches_rec2

            if len(recording_ref) != 0:
                embed()

        print(pointers)
        print(track_ids)
        print(rec_names)

if __name__ == "__main__":

    datapath = "/mnt/backups/@data/output/2016_colombia.nix"
    grid = ConnectFish(datapath, "ReadWrite")
    grid.create_pointers()
