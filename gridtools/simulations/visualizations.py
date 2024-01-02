"""Visualize the results of simulated grid recordings."""

import pathlib
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

def plot_positions(
    original: Tuple,
    upsampled: Tuple,
    downsampled: Tuple,
    grid: Tuple,
    boundaries: List,
    path: pathlib.Path
    ) -> None:
    """Plot position traces of simulated fish on electrode grid."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(grid[0], grid[1], color="black")
    colors = ["orange", "red", "green", "blue", "purple", "pink", "grey"]
    for i in range(len(original[0])):
        col = colors[i]
        ax.plot(downsampled[0][i], downsampled[1][i], color=col, alpha=1, lw=1.5)
        ax.plot(original[0][i], original[1][i], color=col, alpha=0.5, ls='dashed', lw=1)
        ax.plot(upsampled[0][i], upsampled[1][i], color=col, alpha=0.5, ls='dotted', lw=1)

    ax.set_xlim(boundaries[0], boundaries[2])
    ax.set_ylim(boundaries[1], boundaries[3])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Fish positions")
    plt.show()
