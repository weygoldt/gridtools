import datetime
import matplotlib.pyplot as plt

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

        self.pointers = None
        self.pointer_tracks = None
        self.pointer_recs = None

    def __repr__(self) -> str:
        return "NixGrid({})".format(self._filepath)

    def __str__(self) -> str:
        return "Grid recording set at {}".format(self._filepath)

    def create_pointers(self) -> None:

        def get_pairs(matrix):

            # get indices of pairs sorted by difference in ascending order
            pairs = np.unravel_index(np.argsort(
                matrix, axis=None), np.shape(matrix))

            # build matrix to store used indices to not use the same twice
            done = np.ones_like(matrix, dtype=bool)

            # go through pairs and append pairs if not used already
            track_rec1 = []
            track_rec2 = []

            for i in range(len(pairs[0])):

                pair_idx = np.asarray([pairs[0][i], pairs[1][i]])

                if done[pair_idx[0], pair_idx[1]] == True:

                    track_rec1.append(pair_idx[0])
                    track_rec2.append(pair_idx[1])

                    done[pair_idx[0], :] = False
                    done[:, pair_idx[1]] = False

                else:
                    continue

            return track_rec1, track_rec2

        # set thresholds
        # gap threshold between recordings
        dt_between_recordings = int(25 * 60)
        # threshold for gap between track end and recording end
        dt_to_end = int(120)
        dt_baseline = int(
            np.round(800 * self.samplingrate)
        )  # window to compute baseline for pointing

        # Get start and end points of all recordings
        pointers = []  # the newly generated pointers go here
        previous_pointers = []  # save previous pointers in this buffer
        track_ids = []  # the old track ids go here
        previous_matches_rec2 = []  # save previous pairs in this buffer
        rec_names = []  # the recording names go here

        # stash data arrays here to plot
        times = []
        frequencies = []
        identities = []
        indices = []

        for it, (rec1, rec2) in enumerate(zip(self.recordings[:-1], self.recordings[1:])):

            # get time distance between recordings
            stop1 = datetime.timedelta(0, rec1.times[-1]) + rec1.starttime
            start2 = rec2.starttime
            print(f"Stop{stop1}")
            print(f"Start{start2}")
            gap = (start2 - stop1).total_seconds()

            # stash data of first recording as well
            if it == 0:
                times.append(rec1.times)
                frequencies.append(rec1.frequencies)
                identities.append(rec1.identities)
                indices.append(rec1.indices)

            # time to add to next time array so that they dont start at 0
            addtime = times[-1][-1] + gap

            # stash data of this recording
            times.append(rec2.times + addtime)
            frequencies.append(rec2.frequencies)
            identities.append(rec2.identities)
            indices.append(rec2.indices)

            # check if time gap is small enough
            print(gap)
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
                if diff_time > dt_to_end:
                    continue

                # get data at end of recording
                ids1.append(int(track_id))
                baseline1.append(
                    estimateMode(
                        rec1.frequencies[rec1.identities ==
                                         track_id][-dt_baseline:]
                    )
                )

            # get starts of rec2
            ids2 = []
            baseline2 = []

            for track_id in rec2.ids:

                track_id = int(track_id)

                # check if id has data at beginning of recording
                track_starttime = rec2.times[rec2.indices[rec2.identities == track_id]][0]

                abs_starttime = rec2.times[0]
                diff_time = abs(abs_starttime - track_starttime)
                if diff_time > dt_to_end:
                    continue

                # get data at start of recording
                ids2.append(track_id)
                baseline2.append(
                    estimateMode(
                        rec2.frequencies[rec2.identities ==
                                         track_id][-dt_baseline:]
                    )
                )

            # convert to numpy arrays
            ids1 = np.asarray(ids1)
            baseline1 = np.asarray(baseline1)
            ids2 = np.asarray(ids2)
            baseline2 = np.asarray(baseline2)

            # compute difference matrix
            diff_matrix = np.asarray([abs(baseline1 - x) for x in baseline2])

            # get indices for unique pairs with minimum difference frequency
            idxs1, idxs2 = get_pairs(diff_matrix)

            # get ids for these indices
            matches_rec1 = ids1[idxs2]
            matches_rec2 = ids2[idxs1]

            # create new meta ids
            lastmax = 100000
            if len(np.ravel(previous_pointers)) != 0:
                lastmax = np.max(np.ravel(previous_pointers))
            pointers_tmp = np.arange(
                lastmax + 1, lastmax+len(matches_rec1) + 1)

            # reshape matching ids
            matches = [[x, y] for x, y in zip(matches_rec1, matches_rec2)]

            # make recording references
            recording_ref = [[it, it+1] for i in range(len(matches))]

            # check if track ids from currently first recording are already referenced in the previously last recording

            # step 1: get ids in current rec1 that where already in previous rec2
            repeated_matches = [
                x for x in matches_rec1 if x in previous_matches_rec2]  # get repeated

            if len(repeated_matches) != 0:

                # step 2: find pointers of these cases
                indices = np.arange(len(previous_matches_rec2))
                rep_indices = np.asarray(
                    [indices[previous_matches_rec2 == x] for x in repeated_matches])
                rep_pointers = np.asarray(previous_pointers)[rep_indices]

                # step 3: swap the newly created pointers for the previous pointers
                indices = np.arange(len(matches_rec1))
                rep_indices = np.asarray(
                    [indices[matches_rec1 == x] for x in repeated_matches])
                np.asarray(pointers_tmp)[rep_indices] = rep_pointers

            # step 4: append all to lists
            pointers.extend(pointers_tmp)
            track_ids.extend(matches)
            rec_names.extend(recording_ref)

            # step 5: save data from current iteration to buffer to compare in nex iteration
            previous_pointers.append(pointers_tmp)
            previous_matches_rec2 = matches_rec2

        self.pointers = np.asarray(pointers)
        self.pointer_tracks = np.asarray(track_ids)
        self.pointer_recs = np.asarray(rec_names)

        print(self.pointers)
        print(self.pointer_recs)
        print(self.pointer_tracks)

        fig, ax = plt.subplots()
        for t, f, idx, ident in zip(times, frequencies, indices, identities):
            for track_id in np.unique(ident):
                ax.plot(t[idx[ident == track_id]], f[ident == track_id])
                xpos = t[idx[ident == track_id]][0]
                ypos = f[ident == track_id][0]
                ax.annotate(f"{track_id}", xy=(xpos, ypos))
                print(xpos)
                print(ypos)

    def apply_pointers(self) -> None:

        times = []
        frequencies = []
        identities = []
        indices = []
        for it, rec in enumerate(self.recordings):

            # Get pointers that point to this recording
            rec_pointer_idx = np.where(self.pointer_recs == it)[0]
            rec_pointer = self.pointers[rec_pointer_idx]

            # get ids that pointers point to in this recording
            rec_ids_idx = np.where(self.pointer_recs == it)
            rec_pointer_tracks = self.pointer_tracks[rec_ids_idx]

            # iterate through pointed ids and reassign ids in current rec
            for track_id in rec_pointer_tracks:

                # get pointer that points to this track id
                pointer = rec_pointer[rec_pointer_tracks == track_id]

                # rewrite track ids in recording to the pointer
                rec.identities[rec.identities == track_id] = pointer
                rec.ids[rec.ids == track_id] = pointer

            if len(times) != 0:
                times.append(rec.times + times[-1][-1])
            else:
                times.append(rec.times)
            frequencies.append(rec.frequencies)
            identities.append(rec.identities)
            indices.append(rec.indices)

        # plot to see if this works
        fig, ax = plt.subplots()
        for t, f, idx, ident in zip(times, frequencies, indices, identities):
            for track_id in np.unique(ident):
                ax.plot(t[idx[ident == track_id]], f[ident == track_id])
                xpos = t[idx[ident == track_id]][0]
                ypos = f[ident == track_id][0]
                ax.annotate(f"{track_id}", xy=(xpos, ypos))
                print(xpos)
                print(ypos)

        plt.show()


if __name__ == "__main__":

    datapath = "/mnt/backups/@data/output/2016_colombia.nix"
    grid = ConnectFish(datapath, "ReadWrite")
    grid.create_pointers()
    grid.apply_pointers()
