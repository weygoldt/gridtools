"""Annotated spectrograms of simulated scenes (needs the ``plot`` extra).

Two PNGs per scene:

``spectrogram_overview.png``
    The whole scene: electrode-summed power spectrogram in dB (power summed
    across electrodes before taking the log), with each fish's ground-truth
    track as a thin line in its colour, every chirp as a vertical segment
    from its baseline to baseline + excursion at its centre time, and the
    emitter's EODf at the chirp as a dot on the emitter's track. Clicks are
    ticked along the top edge.
``spectrogram_zooms.png``
    Zoomed panels at higher time resolution, placed on near-simultaneous
    chirps of different fish and chirps close in frequency to another fish
    first. Here the segment is drawn just left of the chirp and a caret
    marks its peak, so the chirp itself stays visible.

Fish colours follow a fixed categorical order (fish id = slot); fish are
also identified by direct labels at the left edge and in the legend.
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.fft
from numpy.lib.stride_tricks import sliding_window_view

from gridtools.io import Tracks, load_tracks, open_recording
from gridtools.sim.config import PlotConfig
from gridtools.sim.footprint import HannSpectrogram, chirp_footprint

__all__ = ["FISH_COLORS", "plot_scene", "plot_scene_dir", "summed_spectrogram"]

logger = logging.getLogger(__name__)

#: Categorical colours by fish id (validated default palette, fixed order).
FISH_COLORS: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_FRAME_CHUNK = 1024


def _color(fish_id: int) -> str:
    return FISH_COLORS[fish_id % len(FISH_COLORS)]


def summed_spectrogram(
    data: npt.ArrayLike,
    samplerate_hz: float,
    *,
    nfft: int,
    hop: int,
    f_min_hz: float,
    f_max_hz: float,
    start_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Electrode-summed power spectrogram in dB.

    Hann window, frames every ``hop`` samples, power spectral density
    (units^2/Hz) per electrode summed over electrodes, then
    ``10 log10``.

    Parameters
    ----------
    data : array_like, shape (n, E)
        Samples.
    samplerate_hz : float
    nfft, hop : int
        Window length and hop in samples.
    f_min_hz, f_max_hz : float
        Frequency band kept.
    start_index : int
        Scene sample of ``data[0]`` (for absolute frame times).

    Returns
    -------
    times_s : numpy.ndarray, shape (T,)
        Frame centres in seconds.
    freqs_hz : numpy.ndarray, shape (F,)
    power_db : numpy.ndarray, shape (F, T)
    """
    x = np.asarray(data)
    if x.ndim == 1:
        x = x[:, None]
    n, n_electrodes = x.shape
    if n < nfft:
        msg = f"need at least nfft={nfft} samples, got {n}"
        raise ValueError(msg)
    n_frames = 1 + (n - nfft) // hop
    freqs = np.fft.rfftfreq(nfft, 1.0 / samplerate_hz)
    lo = int(np.searchsorted(freqs, f_min_hz))
    hi = int(np.searchsorted(freqs, f_max_hz, side="right"))
    window = np.hanning(nfft + 1)[:-1].astype(np.float32)
    scale = 2.0 / (samplerate_hz * float(np.sum(window.astype(np.float64) ** 2)))
    power = np.zeros((n_frames, hi - lo), dtype=np.float32)
    for e in range(n_electrodes):
        channel = np.ascontiguousarray(x[:, e], dtype=np.float32)
        frames = sliding_window_view(channel, nfft)[::hop]
        for first in range(0, n_frames, _FRAME_CHUNK):
            block = frames[first : first + _FRAME_CHUNK] * window
            spectrum = np.asarray(scipy.fft.rfft(block, axis=1, workers=-1))
            band = spectrum[:, lo:hi]
            power[first : first + block.shape[0]] += (
                np.real(band) ** 2 + np.imag(band) ** 2
            )
    power = power.astype(np.float64) * scale
    times = (start_index + np.arange(n_frames) * hop + nfft / 2) / samplerate_hz
    return times, freqs[lo:hi], 10.0 * np.log10(np.maximum(power.T, 1e-30))


@dataclass(frozen=True)
class _SceneView:
    """What the plots need, from memory or from a scene directory."""

    recording: Any  # (n, E) array or memmap
    samplerate_hz: float
    chirps: pd.DataFrame
    clicks: pd.DataFrame
    tracks: Tracks
    fish: list[dict[str, Any]]
    title: str
    plot: PlotConfig
    model: Any = None  # SceneModel, for footprint boxes in the zooms


def _track_lines(tracks: Tracks) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {int(t.ident): (t.times_s, t.freqs_hz) for t in tracks}


#: Style of real (background) fish tracks: never chirp, not simulated.
_REAL_STYLE: dict[str, Any] = {"color": "#8a8985", "ls": (0, (4, 2)), "lw": 0.9}


def _simulated_ids(view: "_SceneView") -> set[int]:
    return {int(f["fish_id"]) for f in view.fish}


def _line_style(fid: int, simulated: set[int]) -> dict[str, Any]:
    if fid in simulated:
        return {"color": _color(fid), "lw": 0.8, "alpha": 0.9}
    return {**_REAL_STYLE, "lw": 0.8, "alpha": 0.9}


def _color_limits(power_db: np.ndarray) -> tuple[float, float]:
    vmax = float(np.percentile(power_db, 99.9))
    vmin = max(float(np.percentile(power_db, 5.0)), vmax - 70.0)
    return vmin, vmax


def _new_figure(width_in: float, height_in: float) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width_in, height_in))
    FigureCanvasAgg(fig)
    return fig


def _style_axes(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=_INK_SECONDARY, labelsize=8)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(_INK)
        label.set_fontsize(9)


def _image(ax: Any, times: np.ndarray, freqs: np.ndarray, power: np.ndarray) -> Any:
    vmin, vmax = _color_limits(power)
    dt = times[1] - times[0] if times.size > 1 else 1.0
    df = freqs[1] - freqs[0] if freqs.size > 1 else 1.0
    return ax.imshow(
        power,
        origin="lower",
        aspect="auto",
        cmap="Greys",
        vmin=vmin,
        vmax=vmax,
        extent=(
            times[0] - dt / 2,
            times[-1] + dt / 2,
            freqs[0] - df / 2,
            freqs[-1] + df / 2,
        ),
        interpolation="antialiased",
        rasterized=True,
    )


def _overview(view: _SceneView, path: Path) -> None:
    cfg = view.plot
    fs = view.samplerate_hz
    times, freqs, power = summed_spectrogram(
        view.recording,
        fs,
        nfft=max(8, round(cfg.window_s * fs)),
        hop=max(1, round(cfg.hop_s * fs)),
        f_min_hz=cfg.f_min_hz,
        f_max_hz=cfg.f_max_hz,
    )
    duration_s = view.recording.shape[0] / fs
    fig = _new_figure(16.0, 5.6)
    ax = fig.add_axes((0.05, 0.1, 0.89, 0.74))
    cax = fig.add_axes((0.955, 0.1, 0.008, 0.74))
    image = _image(ax, times, freqs, power)
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("summed power (dB)", fontsize=8, color=_INK_SECONDARY)
    colorbar.ax.tick_params(labelsize=7, colors=_INK_SECONDARY)

    lines = _track_lines(view.tracks)
    handles = []
    for fish in view.fish:
        fid = int(fish["fish_id"])
        color = _color(fid)
        t, f = lines.get(fid, (np.empty(0), np.empty(0)))
        (line,) = ax.plot(t, f, color=color, lw=0.9, alpha=0.9)
        own = view.chirps[view.chirps["emitter_id"] == fid]
        line.set_label(
            f"fish {fid}: {np.mean(f):.0f} Hz, {len(own)} chirps"
            if f.size
            else f"fish {fid}"
        )
        handles.append(line)
        if f.size and cfg.f_min_hz <= f[0] <= cfg.f_max_hz:
            ax.text(
                -0.004 * duration_s,
                f[0],
                str(fid),
                color=color,
                fontsize=8,
                fontweight="bold",
                ha="right",
                va="center",
                clip_on=False,
            )
    simulated = _simulated_ids(view)
    real = [fid for fid in sorted(lines) if fid not in simulated]
    for fid in real:
        t, f = lines[fid]
        (line,) = ax.plot(t, f, alpha=0.9, **_REAL_STYLE)
        if f.size and cfg.f_min_hz <= f[0] <= cfg.f_max_hz:
            ax.text(
                -0.004 * duration_s,
                f[0],
                str(fid),
                color=_REAL_STYLE["color"],
                fontsize=7,
                ha="right",
                va="center",
                clip_on=False,
            )
    if real:
        ax.lines[-1].set_label(f"real fish ({len(real)}, background)")
        handles.append(ax.lines[-1])
    _mark_chirps(ax, view.chirps, offset_s=0.0, lw=1.0)
    if len(view.clicks):
        ax.plot(
            view.clicks["t_start_s"],
            np.full(len(view.clicks), cfg.f_max_hz),
            ls="none",
            marker="v",
            ms=4,
            color=_INK,
            clip_on=False,
            label=f"click ({len(view.clicks)})",
        )
        handles.append(ax.lines[-1])
    ax.set_xlim(0.0, duration_s)
    ax.set_ylim(cfg.f_min_hz, cfg.f_max_hz)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    fig.text(0.05, 0.955, view.title, fontsize=10, color=_INK, ha="left")
    _style_axes(ax)
    legend = fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.05, 0.845),
        ncol=len(handles),
        fontsize=8,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.5,
    )
    for text in legend.get_texts():
        text.set_color(_INK)
    fig.text(
        0.94,
        0.955,
        "line = track, dot = emitter EODf at the chirp, bar = chirp excursion",
        fontsize=8,
        color=_INK_SECONDARY,
        ha="right",
    )
    fig.savefig(path, dpi=cfg.dpi)


def _mark_chirps(
    ax: Any,
    chirps: pd.DataFrame,
    *,
    offset_s: float,
    lw: float,
    caret: bool = False,
) -> None:
    if chirps.empty:
        return
    for fid, group in chirps.groupby("emitter_id"):
        color = _color(int(fid))  # type: ignore[arg-type]
        t = group["t_center_s"].to_numpy()
        base = group["baseline_hz"].to_numpy()
        top = base + group["excursion_hz"].to_numpy()
        ax.vlines(t + offset_s, base, top, color=color, lw=lw, alpha=0.85)
        ax.plot(
            t,
            base,
            ls="none",
            marker="o",
            ms=4.5,
            mfc=color,
            mec="white",
            mew=0.8,
        )
        if caret:
            ax.plot(
                t,
                top,
                ls="none",
                marker="v",
                ms=5,
                mfc=color,
                mec="white",
                mew=0.6,
            )


def _zoom_windows(
    chirps: pd.DataFrame, duration_s: float, n: int, width_s: float
) -> list[tuple[float, float]]:
    """Up to ``n`` non-overlapping windows, hard cases first, in time order."""
    if n == 0 or width_s >= duration_s:
        return [(0.0, duration_s)] if n else []
    if chirps.empty:
        centres = np.linspace(width_s / 2, duration_s - width_s / 2, n)
        return [(c - width_s / 2, c + width_s / 2) for c in centres]
    min_df = chirps["min_df_hz"].fillna(np.inf).to_numpy()
    priority = (
        3.0 * chirps["overlaps_other_chirp"].to_numpy(dtype=float)
        + 2.0 * (min_df < 10.0)
        + 1.0 * (min_df < 20.0)
        + 0.5 * (chirps["family"].to_numpy() == "big")
    )
    order = np.lexsort((chirps["t_center_s"].to_numpy(), -priority))
    windows: list[tuple[float, float]] = []
    for k in order:
        centre = float(chirps["t_center_s"].iat[k])
        start = min(max(centre - width_s / 2, 0.0), duration_s - width_s)
        if all(start + width_s <= a or start >= b for a, b in windows):
            windows.append((start, start + width_s))
        if len(windows) == n:
            break
    return sorted(windows)


def _zooms(view: _SceneView, path: Path) -> None:
    cfg = view.plot
    fs = view.samplerate_hz
    n_samples = view.recording.shape[0]
    duration_s = n_samples / fs
    windows = _zoom_windows(view.chirps, duration_s, cfg.n_zooms, cfg.zoom_s)
    if not windows:
        return
    n_rows = len(windows)
    fig = _new_figure(16.0, 2.6 * n_rows + 0.5)
    axes = fig.subplots(n_rows, 1, squeeze=False)
    lines = _track_lines(view.tracks)
    half = round(cfg.zoom_window_s * fs) // 2
    centres = view.chirps["t_center_s"].to_numpy()
    for ax, (t0, t1) in zip(axes.flat, windows, strict=False):
        first = max(0, round(t0 * fs) - half)
        last = min(n_samples, round(t1 * fs) + half)
        inside = cast(pd.DataFrame, view.chirps.loc[(centres >= t0) & (centres <= t1)])
        f_lo, f_hi = _zoom_band(inside, lines, t0, t1, cfg)
        panel_fn = HannSpectrogram(
            window_s=cfg.zoom_window_s,
            hop_s=cfg.zoom_hop_s,
            f_min_hz=f_lo,
            f_max_hz=f_hi,
        )
        power, freqs, times = panel_fn(view.recording[first:last], fs, first / fs)
        power = 10.0 * np.log10(np.maximum(power, 1e-30))
        _image(ax, times, freqs, power)
        if view.model is not None:
            _draw_boxes(ax, view, inside, panel_fn)
        simulated = _simulated_ids(view)
        for fid, (t, f) in lines.items():
            keep = (t >= t0 - 1.0) & (t <= t1 + 1.0)
            ax.plot(t[keep], f[keep], **_line_style(fid, simulated))
        _mark_chirps(ax, inside, offset_s=-0.006 * (t1 - t0), lw=1.4, caret=True)
        n_overlap = int(np.sum(inside["overlaps_other_chirp"].to_numpy(dtype=bool)))
        df_values = inside["min_df_hz"].to_numpy(dtype=float)
        min_df = (
            float(np.nanmin(df_values)) if np.any(np.isfinite(df_values)) else math.nan
        )
        ax.set_title(
            f"{t0:.1f}-{t1:.1f} s: {len(inside)} chirps, {n_overlap} near-simultaneous"
            + (f", min dF {min_df:.1f} Hz" if np.isfinite(min_df) else ""),
            fontsize=8.5,
            color=_INK,
            loc="left",
        )
        ax.set_xlim(t0, t1)
        ax.set_ylim(f_lo, f_hi)
        ax.set_ylabel("frequency (Hz)")
        _style_axes(ax)
    axes.flat[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"{view.title}\nzooms (window {1e3 * cfg.zoom_window_s:.1f} ms, hop "
        f"{1e3 * cfg.zoom_hop_s:.1f} ms): box = chirp "
        "footprint at this resolution (dashed: not visible at the reference front end), "
        "grey = 2nd harmonic; bar = excursion (left of the chirp), caret = peak, "
        "dot = emitter EODf",
        fontsize=9,
        color=_INK,
        x=0.05,
        ha="left",
    )
    top = 1.0 - 1.0 / fig.get_figheight()
    bottom = 0.45 / fig.get_figheight()
    fig.subplots_adjust(left=0.05, right=0.99, bottom=bottom, top=top, hspace=0.45)
    fig.savefig(path, dpi=cfg.dpi)


def _draw_boxes(
    ax: Any, view: _SceneView, inside: pd.DataFrame, panel_fn: HannSpectrogram
) -> None:
    """Footprint boxes computed with the panel's own spectrogram parameters."""
    from matplotlib.patches import Rectangle

    visible = (
        inside["visible"].to_numpy(dtype=bool)
        if "visible" in inside.columns
        else np.ones(len(inside), dtype=bool)
    )
    for chirp_id, emitter, is_visible in zip(
        inside["chirp_id"].to_numpy(dtype=int),
        inside["emitter_id"].to_numpy(dtype=int),
        visible,
        strict=True,
    ):
        footprint = chirp_footprint(
            view.model,
            int(chirp_id),
            panel_fn,
            margin_s=0.02,
            harmonic_band=(panel_fn.f_min_hz, panel_fn.f_max_hz),
        )
        t0, t1, f0, f1 = footprint.box
        if np.isfinite(t0):
            ax.add_patch(
                Rectangle(
                    (t0, f0),
                    t1 - t0,
                    f1 - f0,
                    fill=False,
                    edgecolor=_color(int(emitter)),
                    lw=1.2,
                    ls="-" if is_visible else (0, (3, 2)),
                    alpha=0.95,
                )
            )
        h0, h1, g0, g1 = footprint.harmonic_box
        if np.isfinite(h0):
            ax.add_patch(
                Rectangle(
                    (h0, g0),
                    h1 - h0,
                    g1 - g0,
                    fill=False,
                    edgecolor="#8a8a8a",
                    lw=0.7,
                    alpha=0.9,
                )
            )


def _zoom_band(
    inside: pd.DataFrame,
    lines: dict[int, tuple[np.ndarray, np.ndarray]],
    t0: float,
    t1: float,
    cfg: PlotConfig,
) -> tuple[float, float]:
    """Frequency band of a zoom: the chirps in it and their nearest fish."""
    if inside.empty:
        return cfg.f_min_hz, cfg.f_max_hz
    lows = list(inside["f_low_hz"]) + list(
        inside["baseline_hz"] - inside["min_df_hz"].fillna(0).abs()
    )
    highs = list(inside["f_high_hz"]) + list(
        inside["baseline_hz"] + inside["min_df_hz"].fillna(0).abs()
    )
    for fid in inside["nearest_fish_id"].unique():
        if fid in lines:
            t, f = lines[int(fid)]
            keep = (t >= t0) & (t <= t1)
            if np.any(keep):
                lows.append(float(f[keep].min()))
                highs.append(float(f[keep].max()))
    lo = max(cfg.f_min_hz, min(lows) - 40.0)
    hi = min(cfg.f_max_hz, max(highs) + 40.0)
    if hi - lo < 150.0:
        mid = 0.5 * (lo + hi)
        lo, hi = max(cfg.f_min_hz, mid - 75.0), min(cfg.f_max_hz, mid + 75.0)
    return lo, hi


def _write(view: _SceneView, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    overview = out_dir / "spectrogram_overview.png"
    zooms = out_dir / "spectrogram_zooms.png"
    _overview(view, overview)
    _zooms(view, zooms)
    return [p for p in (overview, zooms) if p.exists()]


def plot_scene(scene: Any, out_dir: str | Path) -> list[Path]:
    """Write the annotated spectrograms of an in-memory scene.

    Parameters
    ----------
    scene : gridtools.sim.scene.Scene
    out_dir : path
        Directory for the PNGs.

    Returns
    -------
    list of pathlib.Path
        The PNG files written.
    """
    view = _SceneView(
        recording=scene.recording,
        samplerate_hz=scene.samplerate_hz,
        chirps=scene.chirps,
        clicks=scene.clicks,
        tracks=scene.tracks,
        fish=scene.plan.fish_info,
        title=_title(
            scene.config.name,
            scene.seed,
            scene.plan.levels,
            len(scene.plan.fish_info),
            _layout_text(
                scene.plan.layout.kind,
                scene.plan.layout.n_electrodes,
                scene.plan.rows,
                scene.plan.columns,
            ),
        ),
        plot=scene.config.plot,
        model=scene.model,
    )
    return _write(view, Path(out_dir))


def _layout_text(kind: str, n_electrodes: int, rows: int, columns: int) -> str:
    if kind == "background":
        if rows > 1:
            return f"{rows}x{columns} real grid"
        return f"{n_electrodes} real electrode" + ("s" if n_electrodes > 1 else "")
    if kind in ("real", "grid", "line"):
        return f"{rows}x{columns} grid" if kind != "line" else f"line of {n_electrodes}"
    if kind == "single":
        return "1 electrode"
    if kind == "pair":
        return "electrode pair"
    return f"{n_electrodes} electrodes ({kind})"


def _title(
    name: str,
    seed: int,
    levels: dict[str, Any],
    n_fish: int,
    layout: str,
) -> str:
    floor = levels.get("noise_floor_db_v2_hz")
    noise = f", floor {floor:.0f} dB re 1 V²/Hz" if floor is not None else ""
    if levels.get("harder"):
        noise += f" (+{levels['harder_db']:.0f} dB harder)"
    segment = levels.get("background_segment")
    if segment:
        noise = (
            f", real background {segment} at SNR "
            f"{levels.get('background_snr_db', math.nan):.0f} dB"
        )
    spacing = levels.get("spacing_m")
    grid = layout + (f", spacing {spacing:.2f} m" if spacing else "")
    reference = levels.get("reference")
    ref = f", {reference} reference" if reference else ""
    return f"{name} / {seed:06d}: {n_fish} fish, {grid}{noise}{ref}"


def plot_scene_dir(
    directory: str | Path,
    out_dir: str | Path | None = None,
    plot: PlotConfig | None = None,
) -> list[Path]:
    """Write the annotated spectrograms of a written scene directory.

    Parameters
    ----------
    directory : path
        Scene directory (``recording.wav``, tracks, ``chirps.parquet``,
        ``clicks.csv``, ``scene.json``).
    out_dir : path, optional
        Where to write the PNGs (default: the scene directory).
    plot : PlotConfig, optional
        Plot settings (default: those in ``scene.json``).

    Returns
    -------
    list of pathlib.Path
    """
    directory = Path(directory)
    info = json.loads((directory / "scene.json").read_text())
    if plot is None:
        plot = PlotConfig(**_tuple_free(info["config"]["plot"]))
    recording = open_recording(directory / "recording.wav")
    chirps_path = directory / "chirps.parquet"
    chirps = (
        pd.read_parquet(chirps_path)
        if chirps_path.exists()
        else pd.read_csv(directory / "chirps.csv")
    )
    view = _SceneView(
        recording=recording.data,
        samplerate_hz=recording.samplerate_hz,
        chirps=chirps,
        clicks=pd.read_csv(directory / "clicks.csv"),
        tracks=load_tracks(directory),
        fish=info["fish"],
        title=_title(
            info["name"],
            int(info["seed"]),
            info["levels"],
            len(info["fish"]),
            _layout_text(
                str(info["grid"].get("kind", "grid")),
                int(
                    info["grid"].get(
                        "n_electrodes",
                        int(info["grid"]["rows"]) * int(info["grid"]["columns"]),
                    )
                ),
                int(info["grid"]["rows"]),
                int(info["grid"]["columns"]),
            ),
        ),
        plot=plot,
        model=_load_model(directory),
    )
    return _write(view, Path(out_dir) if out_dir is not None else directory)


def _load_model(directory: Path) -> Any:
    from gridtools.sim.scene import load_scene_model

    return load_scene_model(directory)


def _tuple_free(data: dict[str, Any]) -> dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
