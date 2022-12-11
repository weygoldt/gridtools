import datetime
import os

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import plottools.colors as c
import seaborn as sns
from cmocean import cm
from matplotlib.colors import ListedColormap


def PlotStyle(darkmode=False):
    class s:

        lightcmap = cmocean.tools.lighten(cmocean.cm.haline, 0.8)
        specmap = sns.color_palette("Spectral_r", as_cmap=True)
        colors = c.colors_muted
        id1_color = c.lighter(colors["orange"], 0.75)
        id2_color = c.lighter(colors["red"], 0.75)
        id3_color = c.lighter("#9d3695", 0.75)

        cm = 1 / 2.54
        mm = 1 / 25.4

        @classmethod
        def lims(cls, track1, track2):
            """Helper function to get frequency y axis limits from two fundamental frequency tracks.

            Args:
                track1 (array): First track
                track2 (array): Second track
                start (int): Index for first value to be plotted
                stop (int): Index for second value to be plotted
                padding (int): Padding for the upper and lower limit

            Returns:
                lower (float): lower limit
                upper (float): upper limit

            """
            allfunds_tmp = (
                np.concatenate(
                    [
                        track1,
                        track2,
                    ]
                )
                .ravel()
                .tolist()
            )
            lower = np.min(allfunds_tmp)
            upper = np.max(allfunds_tmp)
            return lower, upper

        @classmethod
        def fancy_title(cls, axis, title):
            if " " in title:
                split_title = title.split(" ", 1)
                axis.set(
                    title=r"$\bf{{{}}}$".format(split_title[0]) + f" {split_title[1]}"
                )
            else:
                axis.set_title(r"$\bf{{{}}}$".format(title.replace(" ", r"\;")), pad=8)

        @classmethod
        def fancy_suptitle(cls, fig, title):
            split_title = title.split(" ", 1)
            fig.suptitle(
                r"$\bf{{{}}}$".format(split_title[0]) + f" {split_title[1]}",
                ha="left",
                x=0.078,
            )

        @classmethod
        def circled_annotation(cls, text, axis, xpos, ypos, padding=0.25):
            axis.text(
                xpos,
                ypos,
                text,
                ha="center",
                va="center",
                zorder=1000,
                bbox=dict(
                    boxstyle=f"circle, pad={padding}", fc="white", ec="black", lw=1
                ),
            )

        @classmethod
        def fade_cmap(cls, cmap):

            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
            my_cmap = ListedColormap(my_cmap)

            return my_cmap

        @classmethod
        def label_subplots(cls, labels, axes, fig):
            for axis, label in zip(axes, labels):
                X = axis.get_position().x0
                Y = axis.get_position().y1
                fig.text(X, Y, label, weight="bold")

        @classmethod
        def clocktime(cls, path, axis, times, xlims):
            def get_datetime(path):
                folder = path[:-1]
                rec_year, rec_month, rec_day, rec_time = os.path.split(
                    os.path.split(folder)[-1]
                )[-1].split("-")
                rec_year = int(rec_year)
                rec_month = int(rec_month)
                rec_day = int(rec_day)
                try:
                    rec_time = [
                        int(rec_time.split("_")[0]),
                        int(rec_time.split("_")[1]),
                        0,
                    ]
                except:
                    rec_time = [
                        int(rec_time.split(":")[0]),
                        int(rec_time.split(":")[1]),
                        0,
                    ]

                rec_datetime = datetime.datetime(
                    year=rec_year,
                    month=rec_month,
                    day=rec_day,
                    hour=rec_time[0],
                    minute=rec_time[1],
                    second=rec_time[2],
                )

                return rec_datetime

            def clock_time(axis, xlims, times, rec_datetime):
                xlim = xlims
                dx = np.diff(xlim)[0]

                label_idx0 = 0
                if dx <= 20:
                    res = 1
                elif dx > 20 and dx <= 120:
                    res = 10
                elif dx > 120 and dx <= 1200:
                    res = 60
                elif dx > 1200 and dx <= 3600:
                    res = 600  # 10 min
                elif dx > 3600 and dx <= 7200:
                    res = 1800  # 30 min
                else:
                    res = 3600  # 60 min

                if dx > 1200:
                    if rec_datetime.minute % int(res / 60) != 0:
                        dmin = int(res / 60) - rec_datetime.minute % int(res / 60)
                        label_idx0 = dmin * 60

                xtick = np.arange(label_idx0, times[-1], res)
                datetime_xlabels = list(
                    map(lambda x: rec_datetime + datetime.timedelta(seconds=x), xtick)
                )

                if dx > 120:
                    xlabels = list(
                        map(
                            lambda x: (
                                "%2s:%2s" % (str(x.hour), str(x.minute))
                            ).replace(" ", "0"),
                            datetime_xlabels,
                        )
                    )
                    rotation = 0
                else:
                    xlabels = list(
                        map(
                            lambda x: (
                                "%2s:%2s:%2s"
                                % (str(x.hour), str(x.minute), str(x.second))
                            ).replace(" ", "0"),
                            datetime_xlabels,
                        )
                    )
                    rotation = 45
                # ToDo: create mask
                mask = np.arange(len(xtick))[(xtick > xlim[0]) & (xtick < xlim[1])]
                axis.set_xticks(xtick[mask])
                axis.set_xticklabels(np.array(xlabels)[mask], rotation=rotation)
                axis.set_xlim(xlim)
                print(xlim)

            rec_datetime = get_datetime(path)
            clock_time(axis, xlims, times, rec_datetime)

        @classmethod
        def hide_helper_xax(cls, ax):
            ax.xaxis.set_visible(False)
            plt.setp(ax.spines.values(), visible=False)
            ax.tick_params(left=False, labelleft=False)
            ax.patch.set_visible(False)

        @classmethod
        def set_boxplot_color(cls, bp, color):
            plt.setp(bp["boxes"], color=color)
            plt.setp(bp["whiskers"], color=color)
            plt.setp(bp["caps"], color=color)
            plt.setp(bp["medians"], color=color)

        @classmethod
        def letter_subplots(
            cls, axes=None, letters=None, xoffset=-0.1, yoffset=1.0, **kwargs
        ):
            """Add letters to the corners of subplots (panels). By default each axis is
            given an uppercase bold letter label placed in the upper-left corner.
            Args
                axes : list of pyplot ax objects. default plt.gcf().axes.
                letters : list of strings to use as labels, default ["A", "B", "C", ...]
                xoffset, yoffset : positions of each label relative to plot frame
                (default -0.1,1.0 = upper left margin). Can also be a list of
                offsets, in which case it should be the same length as the number of
                axes.
                Other keyword arguments will be passed to annotate() when panel letters
                are added.
            Returns:
                list of strings for each label added to the axes
            Examples:
                Defaults:
                    >>> fig, axes = plt.subplots(1,3)
                    >>> letter_subplots() # boldfaced A, B, C

                Common labeling schemes inferred from the first letter:
                    >>> fig, axes = plt.subplots(1,4)
                    >>> letter_subplots(letters='(a)') # panels labeled (a), (b), (c), (d)
                Fully custom lettering:
                    >>> fig, axes = plt.subplots(2,1)
                    >>> letter_subplots(axes, letters=['(a.1)', '(b.2)'], fontweight='normal')
                Per-axis offsets:
                    >>> fig, axes = plt.subplots(1,2)
                    >>> letter_subplots(axes, xoffset=[-0.1, -0.15])

                Matrix of axes:
                    >>> fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
                    >>> letter_subplots(fig.axes) # fig.axes is a list when axes is a 2x2 matrix
            """

            # get axes:
            if axes is None:
                axes = plt.gcf().axes
            # handle single axes:
            try:
                iter(axes)
            except TypeError:
                axes = [axes]

            # set up letter defaults (and corresponding fontweight):
            fontweight = "bold"
            ulets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(axes)])
            llets = list("abcdefghijklmnopqrstuvwxyz"[: len(axes)])
            if letters is None or letters == "A":
                letters = ulets
            elif letters == "(a)":
                letters = ["({})".format(lett) for lett in llets]
                fontweight = "normal"
            elif letters == "(A)":
                letters = ["({})".format(lett) for lett in ulets]
                fontweight = "normal"
            elif letters in ("lower", "lowercase", "a"):
                letters = llets

            # make sure there are x and y offsets for each ax in axes:
            if isinstance(xoffset, (int, float)):
                xoffset = [xoffset] * len(axes)
            else:
                assert len(xoffset) == len(axes)
            if isinstance(yoffset, (int, float)):
                yoffset = [yoffset] * len(axes)
            else:
                assert len(yoffset) == len(axes)

            # defaults for annotate (kwargs is second so it can overwrite these defaults):
            my_defaults = dict(
                fontweight=fontweight,
                fontsize="large",
                ha="center",
                va="center",
                xycoords="axes fraction",
                annotation_clip=False,
            )
            kwargs = dict(list(my_defaults.items()) + list(kwargs.items()))

            list_txts = []
            for ax, lbl, xoff, yoff in zip(axes, letters, xoffset, yoffset):
                t = ax.annotate(lbl, xy=(xoff, yoff), **kwargs)
                list_txts.append(t)
            return list_txts

        pass

    # general purpose plot elements
    s.id1 = dict(color=s.id1_color, lw=2, zorder=10)
    s.id2 = dict(color=s.id2_color, lw=2, zorder=10)
    s.id3 = dict(color=s.id3_color, lw=2, zorder=10)

    s.id1_slim = dict(color=s.id1_color, lw=1.5, zorder=10)
    s.id2_slim = dict(color=s.id2_color, lw=1.5, zorder=10)
    s.id3_slim = dict(color=s.id3_color, lw=1.5, zorder=10)

    # black traces
    if darkmode:
        s.id1 = dict(color=c.lighter("black", 0.86), lw=1, zorder=10)
        s.id2 = dict(color=c.lighter("black", 0.86), lw=1, zorder=10)
        s.id3 = dict(color=c.lighter("black", 0.86), lw=1, zorder=10)

    # for CovarianceEvents class
    s.coarse_spec = dict(
        aspect="auto", cmap="virids", alpha=0.7, interpolation="gaussian"
    )
    s.fine_spec = dict(
        aspect="auto", cmap=s.specmap, alpha=0.8, interpolation="gaussian"
    )

    s.cov_heatmap = dict(
        aspect="auto", cmap=s.lightcmap, alpha=0.8, interpolation="gaussian"
    )
    s.center_line = dict(
        color="black",
        alpha=0.2,
        linestyle="dashed",
        linewidth=1,
    )
    s.lags = dict(color=s.colors["black"], lw=1, alpha=0.8, label="maxima")

    s.maxcovs_h = dict(color="#7D96B0", lw=2, label="higher bp")
    s.maxcovs_l = dict(color="#8CB78D", lw=2, label="lower bp")
    s.maxcovs_h_peaks = dict(marker="o", color="#7D96B0", alpha=0.6)
    s.maxcovs_l_peaks = dict(marker="o", color="#8CB78D", alpha=0.6)

    # for grid plots
    s.grid_electrodes = dict(marker=".", color="grey", zorder=-10, alpha=0.2)

    # for grid kde / distances / fundamental tracks subplot

    s.kde1_shading1 = dict(
        colors=c.lighter(s.colors["orange"], 0.4),
        alpha=0.3,
        zorder=3,
    )
    s.kde2_shading1 = dict(
        colors=c.lighter(s.colors["red"], 0.4),
        alpha=0.3,
        zorder=3,
    )
    s.kde1_contours = dict(
        colors=c.lighter(s.colors["gray"], 0.8),
        linestyles="solid",
        linewidths=1,
        alpha=0.5,
        zorder=4,
    )
    s.kde2_contours = dict(
        colors=c.lighter(s.colors["gray"], 0.8),
        linestyles="solid",
        linewidths=1,
        alpha=0.5,
        zorder=4,
    )
    s.distance = dict(color="#7D96B0", lw=2, zorder=10)

    s.timewindow = dict(
        color=c.lighter(s.colors["gray"], 0.8),
        alpha=0.6,
        lw=0,
        zorder=-10,
    )

    # for stat plots
    s.kde1_shading = dict(
        color=s.id1_color,
        alpha=0.15,
        zorder=3,
    )
    s.kde2_shading = dict(
        color=s.id2_color,
        alpha=0.15,
        zorder=3,
    )

    # for eventstats plot

    clrs = mpl.cm.get_cmap(cmocean.cm.haline)
    c1, c2, c3 = clrs(0.2), clrs(0.6), clrs(0.8)

    c1 = "#7D96B0"
    c2 = "#8CB78D"
    c3 = "#94C0C0"

    # c1, c2, c3 = s.colors["blue"], s.colors["green"], s.colors["cyan"]

    # clrs = sns.color_palette("Spectral", as_cmap=True)
    # clrs = mpl.cm.get_cmap(clrs)
    # c1, c2, c3 = clrs(0.3), clrs(0.6), clrs(0.9)

    s.kde1 = dict(
        color=c1,
        alpha=0.5,
    )
    s.kde2 = dict(
        color=c2,
        alpha=0.5,
    )

    s.kde3 = dict(
        color=c3,
        alpha=0.5,
    )

    s.kde1_line = dict(
        color=c1,
        lw=1.5,
    )

    s.kde1_line_error = dict(
        color=c1,
        alpha=0.5,
        lw=0,
        zorder=-1000,
    )

    s.kde2_line = dict(
        color=c2,
        lw=1.5,
    )

    s.kde2_line_error = dict(
        color=c2,
        alpha=0.5,
        lw=0,
        zorder=-1001,
    )

    s.kde3_line = dict(
        color=c3,
        alpha=1,
        lw=1.5,
    )

    s.scatter1 = dict(
        linewidth=0,
        marker=".",
        color=c1,
    )
    s.scatter2 = dict(
        linewidth=0,
        marker=".",
        color=c2,
    )

    # rcparams text setup
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    # rcparams
    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rcParams["image.cmap"] = s.lightcmap
    plt.rcParams["axes.xmargin"] = 0
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["axes.ymargin"] = 0
    plt.rcParams["axes.titlelocation"] = "left"
    plt.rcParams["axes.titlesize"] = BIGGER_SIZE
    plt.rcParams["axes.titlepad"] = 12
    # plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.loc"] = "best"
    # plt.rcParams["legend.borderpad"] = 0.4
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "white"
    plt.rcParams["legend.framealpha"] = 0.7
    plt.rcParams["legend.borderaxespad"] = 0.5
    plt.rcParams["legend.fancybox"] = False

    # specify the custom font to use
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Helvetica Now Text"
    return s


if __name__ == "__main__":
    s = PlotStyle()
