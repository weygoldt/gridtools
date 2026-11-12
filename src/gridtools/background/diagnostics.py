"""Diagnostics of a screened background library (needs the ``plot`` extra).

:func:`write_diagnostics` draws, into ``<library>/figures/diagnostics/``:

``rates/<dataset>_<recording>.png``
    Screen boxes per 30 min over each competition recording, at each
    threshold (default 0.1 and 0.5), with the initial dark phase, the
    library window and the recording gate's rate, so the light-phase
    activity of a recording can be compared with its dark phase.
``gate.png``
    Every recording's gate rate (boxes per hour of its window), the gate
    limit and which recordings were dropped.
``snippets/<recording>/*.png`` and ``snippets/<recording>.html``
    10-s spectrograms (electrode sum, 102-ms window) around screen boxes
    found in the light phase of chosen recordings, with all boxes above the
    lowest threshold, the tracked fish, and a 1-s zoom (25.6-ms window) on
    the box, so real chirps can be told from artefacts. Boxes are drawn
    with at most half above and half below 0.5 (seeded).
"""

from __future__ import annotations

import html
import itertools
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridtools.background.review import spectrogram_power
from gridtools.background.screen import ScreenTable, read_screen

__all__ = ["plot_gate", "plot_rates", "plot_snippet", "write_diagnostics"]

log = logging.getLogger(__name__)

BIN_S = 1800.0


def _screened_builds(root: Path) -> tuple[dict[str, Any], dict[tuple[str, str], Any]]:
    """Config of the last build with chirp screens, and the sources of all of them.

    Builds are incremental (each replaces its recordings' rows), so the
    sources of earlier screened builds stay in the library; the latest build
    of a recording wins.
    """
    meta = json.loads((root / "library.json").read_text())
    builds = [b for b in meta.get("builds", []) if b.get("config", {}).get("screens")]
    if not builds:
        msg = f"{root}: no build with chirp screens in library.json"
        raise ValueError(msg)
    sources = {
        (s["dataset"], s["name"]): s
        for b in builds
        for s in b["sources"]
        if "recording_path" in s
    }
    return builds[-1]["config"], sources


def plot_rates(
    table: ScreenTable,
    row: dict[str, Any],
    out: Path,
    *,
    thresholds: Sequence[float],
    dark_phase_s: float | None,
    gate: dict[str, Any] | None,
) -> dict[str, float]:
    """Boxes per 30 min over one recording; returns rates per hour by phase."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration = float(row["duration_s"])
    edges = np.arange(0.0, duration + BIN_S, BIN_S)
    centres = 0.5 * (table.boxes["t_start_s"] + table.boxes["t_end_s"]).to_numpy()
    covered = np.array(
        [table.covered_within(a, min(b, duration)) for a, b in _pairs(edges)]
    )
    fig, ax = plt.subplots(figsize=(10, 3.4))
    colours = ["tab:orange", "tab:red", "tab:purple", "tab:blue"]
    stats: dict[str, float] = {}
    for k, thr in enumerate(sorted(thresholds)):
        sel = table.boxes["score"].to_numpy() >= thr
        counts, _ = np.histogram(centres[sel], edges)
        # per 30 min of covered audio (bins the detector skipped are blank)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(covered > 0, counts / covered * BIN_S, np.nan)
        ax.stairs(
            rate + 1,
            edges / 3600,
            color=colours[k % len(colours)],
            lw=2,
            label=f"score >= {thr:g}",
            baseline=1,
        )
        for name, (a, b) in {
            "dark": (0.0, dark_phase_s or 0.0),
            "light": (dark_phase_s or 0.0, duration),
            "window": (
                float(row.get("window_start_s", math.nan)),
                float(row.get("window_stop_s", math.nan)),
            ),
        }.items():
            if not (math.isfinite(a) and math.isfinite(b)) or b <= a:
                continue
            hours = table.covered_within(a, b) / 3600
            n = int(np.sum(sel & (centres >= a) & (centres <= b)))
            stats[f"{name}_rate_{thr:g}_per_h"] = n / hours if hours > 0 else math.nan
    if dark_phase_s:
        ax.axvspan(0, dark_phase_s / 3600, color="0.85", zorder=0, label="dark phase")
    a, b = row.get("window_start_s"), row.get("window_stop_s")
    if a is not None and b is not None and math.isfinite(float(a)):
        ax.axvspan(
            float(a) / 3600,
            float(b) / 3600,
            color="tab:green",
            alpha=0.15,
            zorder=0,
            label="library window",
        )
    if gate is not None:
        ax.axhline(
            gate["max_rate_per_h"] * BIN_S / 3600 + 1,
            color="k",
            ls="--",
            lw=1,
            label=f"gate {gate['max_rate_per_h']:g}/h (score >= {gate['threshold']:g})",
        )
    ax.set_yscale("log")
    ax.set_xlim(0, duration / 3600)
    ax.set_xlabel("time from recording start (h)")
    ax.set_ylabel("boxes per 30 min + 1")
    gate_rate = row.get("gate_rate_per_h")
    passed = row.get("gate_passed")
    verdict = (
        ""
        if gate_rate is None or not math.isfinite(float(gate_rate))
        else f" | gate {float(gate_rate):.1f}/h: {'kept' if passed else 'DROPPED'}"
    )
    ax.set_title(f"{row['dataset']}/{row['recording']}{verdict}", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return stats


def _pairs(edges: np.ndarray) -> list[tuple[float, float]]:
    """Consecutive pairs of bin edges."""
    return [(float(a), float(b)) for a, b in itertools.pairwise(edges)]


def plot_gate(recordings: pd.DataFrame, gate: dict[str, Any], out: Path) -> None:
    """Gate rate of every recording, sorted, with the limit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = recordings.dropna(subset=["gate_rate_per_h"]).sort_values("gate_rate_per_h")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    colours = {d: f"C{k}" for k, d in enumerate(sorted(df["dataset"].unique()))}
    x = np.arange(len(df))
    ax.bar(
        x,
        df["gate_rate_per_h"].to_numpy(dtype=float) + 0.1,
        color=[colours[d] for d in df["dataset"]],
    )
    ax.axhline(gate["max_rate_per_h"], color="k", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(x, df["recording"], rotation=90, fontsize=6)
    ax.set_ylabel(
        f"{gate['screen']} boxes/h in window (score >= {gate['threshold']:g})"
    )
    for d, c in colours.items():
        ax.bar([], [], color=c, label=d)
    ax.legend(fontsize=8)
    ax.set_title(
        f"recording gate: {int((df['gate_rate_per_h'] > gate['max_rate_per_h']).sum())} "
        f"of {len(df)} recordings above {gate['max_rate_per_h']:g}/h",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def plot_snippet(
    x: np.ndarray,
    samplerate_hz: float,
    t0_s: float,
    box: dict[str, Any],
    boxes: pd.DataFrame,
    tracks: Any,
    out: Path,
    *,
    title: str,
    min_score: float,
) -> None:
    """10-s overview and 1-s zoom around one box (see module docstring)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    duration = x.shape[0] / samplerate_hz
    centre = 0.5 * (float(box["t_start_s"]) + float(box["t_end_s"]))
    lo_hz = max(0.0, float(box["f_low_hz"]) - 400.0)
    hi_hz = min(2000.0, float(box["f_high_hz"]) + 300.0)
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 5.0), gridspec_kw={"width_ratios": [3, 1]}
    )
    for ax, nfft, span, f_lim in (
        (axes[0], None, (0.0, duration), (0.0, 2000.0)),
        (
            axes[1],
            int(2 ** round(math.log2(0.0256 * samplerate_hz))),
            (centre - t0_s - 0.5, centre - t0_s + 0.5),
            (lo_hz, hi_hz),
        ),
    ):
        a = max(0, round(span[0] * samplerate_hz))
        b = min(x.shape[0], round(span[1] * samplerate_hz))
        part = x[a:b]
        hop = None if nfft is None else max(1, nfft // 16)
        times, freqs, power = spectrogram_power(
            part, samplerate_hz, 2000.0, nfft=nfft, hop=hop
        )
        times = times + a / samplerate_hz + t0_s
        db = 10 * np.log10(np.maximum(power, 1e-300))
        band = (freqs >= f_lim[0]) & (freqs <= f_lim[1])
        floor = float(np.median(db[band]))
        ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=(times[0], times[-1], freqs[0], freqs[-1]),
            cmap="magma",
            vmin=floor - 5,
            vmax=floor + 55,
            interpolation="nearest",
        )
        if tracks is not None:
            for k, track in enumerate(tracks):
                inside = (track.times_s >= times[0] - 1) & (
                    track.times_s <= times[-1] + 1
                )
                if inside.any():
                    ax.plot(
                        track.times_s[inside],
                        track.freqs_hz[inside],
                        color=f"C{k % 10}",
                        lw=0.8,
                        alpha=0.7,
                    )
        near = pd.DataFrame(
            boxes.loc[
                (
                    (boxes["t_end_s"] >= times[0])
                    & (boxes["t_start_s"] <= times[-1])
                    & (boxes["score"] >= min_score)
                ).to_numpy()
            ]
        )
        for b_row in near.to_dict("records"):
            strong = b_row["score"] >= 0.5
            ax.add_patch(
                Rectangle(
                    (b_row["t_start_s"], b_row["f_low_hz"]),
                    b_row["t_end_s"] - b_row["t_start_s"],
                    b_row["f_high_hz"] - b_row["f_low_hz"],
                    fill=False,
                    ec="cyan" if strong else "white",
                    lw=1.2,
                    ls="-" if strong else "--",
                )
            )
            if f_lim[0] <= b_row["f_high_hz"] < f_lim[1]:
                ax.text(
                    b_row["t_start_s"],
                    b_row["f_high_hz"] + 8,
                    f"{b_row['score']:.2f}",
                    color="cyan" if strong else "white",
                    fontsize=7,
                    clip_on=True,
                )
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(*f_lim)
        ax.set_xlabel("recording time (s)")
        ax.set_ylabel("Hz")
    axes[1].set_title("1-s zoom (25.6-ms window)", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)


def _snippets(
    root: Path,
    source: dict[str, Any],
    table: ScreenTable,
    *,
    min_score: float,
    n: int,
    seed: int,
) -> list[str]:
    from gridtools.io import load_tracks, open_recording

    dark = source.get("dark_phase_s") or 0.0
    boxes = table.above(min_score)
    centres = 0.5 * (boxes["t_start_s"] + boxes["t_end_s"])
    light = pd.DataFrame(boxes.loc[(centres >= dark).to_numpy()])
    rng = np.random.default_rng(seed)
    strong = light[light["score"] >= 0.5]
    weak = light[light["score"] < 0.5]
    picks = []
    for part, k in ((strong, n // 2), (weak, n - n // 2)):
        if len(part):
            idx = rng.choice(len(part), size=min(k, len(part)), replace=False)
            picks.append(part.iloc[np.sort(idx)])
    if not picks:
        return []
    chosen = pd.concat(picks).sort_values("t_start_s")
    rec = open_recording(source["recording_path"])
    tracks = load_tracks(source["tracks_dir"]) if source.get("tracks_dir") else None
    fs = rec.samplerate_hz
    out_dir = root / "figures" / "diagnostics" / "snippets" / source["name"]
    names = []
    for box in chosen.to_dict("records"):
        centre = 0.5 * (box["t_start_s"] + box["t_end_s"])
        t0 = max(0.0, centre - 5.0)
        a, b = round(t0 * fs), min(rec.n_samples, round((t0 + 10.0) * fs))
        x = rec.read(a, b)
        name = f"{source['name']}_{centre:09.2f}.png"
        plot_snippet(
            x,
            fs,
            a / fs,
            box,
            boxes,
            tracks,
            out_dir / name,
            title=f"{source['dataset']}/{source['name']} | box at {centre:.2f} s "
            f"({centre / 3600:.2f} h), {box['f_low_hz']:.0f}-{box['f_high_hz']:.0f} Hz, "
            f"score {box['score']:.2f} | cyan: >= 0.5, white dashed: "
            f"{min_score:g}-0.5",
            min_score=min_score,
        )
        names.append(name)
    page = out_dir.parent / f"{source['name']}.html"
    page.write_text(
        "<!doctype html><meta charset=utf-8><title>"
        f"{html.escape(source['name'])}</title><style>body{{font:12px sans-serif}}"
        "img{width:100%;max-width:1440px}</style>"
        f"<h1>{html.escape(source['name'])}: {len(names)} light-phase boxes</h1>"
        + "".join(
            f'<p><img src="{html.escape(source["name"])}/{html.escape(m)}"></p>'
            for m in names
        )
    )
    return names


def write_diagnostics(
    root: str | Path,
    *,
    screen: str = "v1",
    thresholds: Sequence[float] = (0.1, 0.5),
    snippet_recordings: Sequence[str] = (),
    n_snippets: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Draw the diagnostics of a screened library (module docstring).

    Parameters
    ----------
    root : path
        Library directory (built with chirp screens).
    screen : str
        Screen whose boxes are drawn (from the last screened build).
    thresholds : sequence of float
        Score thresholds of the rate curves.
    snippet_recordings : sequence of str
        Recording names to draw light-phase snippets of.
    n_snippets : int
        Snippets per recording.
    seed : int

    Returns
    -------
    dict
        Per recording: rates per hour in the dark phase, the light phase and
        the window, for each threshold (also written as
        ``figures/diagnostics/rates.csv``).
    """
    root = Path(root)
    config, sources = _screened_builds(root)
    screens = {s["name"]: s for s in config["screens"]}
    if screen not in screens:
        msg = f"screen {screen!r} not in {sorted(screens)}"
        raise ValueError(msg)
    screen_root = Path(screens[screen]["root"])
    gate = config.get("gate")
    recordings = pd.read_csv(root / "recordings.csv")
    out = root / "figures" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in recordings.to_dict("records"):
        key = (str(row["dataset"]), str(row["recording"]))
        if key not in sources:
            continue
        table = read_screen(screen_root, *key)
        if table is None or not math.isfinite(float(row.get("duration_s", math.nan))):
            continue
        stats = plot_rates(
            table,
            row,
            out / "rates" / f"{key[0]}_{key[1]}.png",
            thresholds=thresholds,
            dark_phase_s=sources[key].get("dark_phase_s"),
            gate=gate,
        )
        covered_h = sum(b - a for a, b in table.covered_s.tolist()) / 3600
        rows.append(
            {
                "dataset": key[0],
                "recording": key[1],
                "duration_h": float(row["duration_s"]) / 3600,
                "screen_covered_h": covered_h,
                "gate_rate_per_h": row.get("gate_rate_per_h"),
                "gate_passed": row.get("gate_passed"),
                "skipped": row.get("skipped"),
                **stats,
            }
        )
    rates = pd.DataFrame(rows)
    rates.to_csv(out / "rates.csv", index=False, float_format="%.6g")
    if gate is not None and "gate_rate_per_h" in recordings.columns:
        plot_gate(recordings, gate, out / "gate.png")
    snippets: dict[str, list[str]] = {}
    for name in snippet_recordings:
        match = [s for (d, r), s in sources.items() if r == name]
        if not match:
            log.warning("snippets: %s is not a source of the last build", name)
            continue
        source = match[0]
        table = read_screen(screen_root, source["dataset"], name)
        if table is None:
            log.warning("snippets: no %s screen for %s", screen, name)
            continue
        snippets[name] = _snippets(
            root,
            source,
            table,
            min_score=min(thresholds),
            n=n_snippets,
            seed=seed,
        )
    log.info("diagnostics: %d recordings, snippets of %s", len(rates), list(snippets))
    return {"rates": rates, "snippets": snippets}
