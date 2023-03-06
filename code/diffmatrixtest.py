import numpy as np
from IPython import embed

ids1 = np.unique(np.random.randint(1, 10, 10))
ids2 = np.unique(np.random.randint(22, 33, 10))

baseline1 = np.random.rand(len(ids1))
baseline2 = np.random.rand(len(ids2))

diffm = np.asarray([baseline1 - x for x in baseline2])


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


pair_indices = get_pairs(diffm)
print(pair_indices)
