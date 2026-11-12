# ruff: noqa: RUF001
"""Report of a merged library (v2 + harvested long chirps), Markdown + PNG.

Needs the optional ``plot`` extra (matplotlib), imported lazily. The
chirp-chamber part is documented in the base library's report; this one
covers what changed: the author decisions on exclusions, the long chirps
from the competition recordings, the template length and the equalization.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from gridtools.templates.library import (
    BRIEF_CYCLES_MAX,
    DEFAULT_ALPHA,
    DEFAULT_BINS,
    DEFAULT_EXCLUDE,
    DEFAULT_SPACING,
    MIN_DURATION_S,
    Balance,
    Spacing,
    TemplateLibrary,
    load_library,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

log = logging.getLogger(__name__)

Frame = Any

#: Duration classes (full width at 10%, ms) of the sampling comparison.
DURATION_CLASSES_MS: tuple[tuple[str, float, float], ...] = (
    ("< 20 ms", 0.0, 20.0),
    ("20–50 ms", 20.0, 50.0),
    ("50–150 ms", 50.0, 150.0),
    ("≥ 150 ms", 150.0, np.inf),
)


@dataclass(frozen=True, slots=True)
class Scheme:
    """A sampling scheme for the comparison table."""

    label: str
    equalize: bool
    spacing: Spacing = "log"
    alpha: float = 1.0
    balance: Balance = "template"


def draw_probabilities(
    lib: TemplateLibrary, scheme: Scheme, bins: tuple[int, int] = DEFAULT_BINS
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Pool template ids and their draw probability under ``scheme``."""
    if not scheme.equalize:
        ids = lib.unique_ids()
        return ids, np.full(ids.size, 1.0 / ids.size)
    eq = lib.equalization_bins(bins, spacing=scheme.spacing)
    return eq.template_ids, eq.template_probabilities(scheme.alpha, scheme.balance)


def individual_summary(
    lib: TemplateLibrary, scheme: Scheme, n_draws: int = 100_000, top: int = 10
) -> dict[str, Any]:
    """How draws spread over individuals under ``scheme``.

    Returns
    -------
    dict
        ``top_share`` (draws from the ``top`` individuals with most templates
        in the pool), ``effective_individuals`` (exp of the entropy of the
        draw shares of individuals), ``distinct_individuals`` (expected number
        drawn at least once in ``n_draws``), ``n_individuals``.
    """
    ids, p = draw_probabilities(lib, scheme)
    labels = lib.individuals()[lib.rows_of(ids)]
    names, codes = np.unique(labels, return_inverse=True)
    share = np.bincount(codes, weights=p, minlength=names.size)
    size = np.bincount(codes, minlength=names.size)
    largest = np.argsort(-size, kind="stable")[:top]
    positive = share[share > 0]
    return {
        "top_share": float(share[largest].sum()),
        "effective_individuals": float(np.exp(-np.sum(positive * np.log(positive)))),
        "distinct_individuals": float(np.sum(1.0 - (1.0 - share) ** n_draws)),
        "n_individuals": int(names.size),
        "largest": [(str(names[k]), int(size[k])) for k in largest],
    }


def sampling_summary(
    lib: TemplateLibrary, scheme: Scheme, n_draws: int = 100_000
) -> dict[str, Any]:
    """Draw shares by family, duration class and source, and repetition.

    Returns
    -------
    dict
        ``big`` and ``small`` shares, one share per duration class, the
        ``competition`` share, ``distinct`` (expected number of distinct
        templates in ``n_draws``), ``effective`` (exp of the entropy) and
        ``max_repeats`` (expected draws of the most likely template).
    """
    ids, p = draw_probabilities(lib, scheme)
    meta: Frame = lib.metadata.set_index("template_id").loc[ids]
    family = meta["family"].to_numpy()
    duration_ms = 1e3 * meta["duration_10_s"].to_numpy(dtype=np.float64)
    source = (
        meta["source"].to_numpy()
        if "source" in meta.columns
        else np.full(ids.size, "chirp_chamber")
    )
    row: dict[str, Any] = {
        "scheme": scheme.label,
        "big": float(p[family == "big"].sum()),
        "small": float(p[family == "small"].sum()),
    }
    for label, lo, hi in DURATION_CLASSES_MS:
        row[label] = float(p[(duration_ms >= lo) & (duration_ms < hi)].sum())
    row["competition"] = float(p[source == "competition"].sum())
    row["distinct"] = float(np.sum(1.0 - (1.0 - p) ** n_draws))
    positive = p[p > 0]
    row["effective"] = float(np.exp(-np.sum(positive * np.log(positive))))
    row["max_repeats"] = float(n_draws * p.max())
    return row


def _fmt_share(x: float) -> str:
    return f"{100 * x:.1f}%"


def _table(frame: pd.DataFrame) -> list[str]:
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = (
        "|" + "|".join("---" if i == 0 else "---:" for i in range(frame.shape[1])) + "|"
    )
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.to_numpy()]
    return [header, rule, *rows]


def _bin_table(counts: pd.DataFrame) -> list[str]:
    header = "| | " + " | ".join(str(c) for c in counts.columns) + " |"
    rule = "|---|" + "---:|" * counts.shape[1]
    rows = [
        f"| {index} | " + " | ".join(str(v) if v else "·" for v in row) + " |"
        for index, row in zip(counts.index, counts.to_numpy().tolist(), strict=True)
    ]
    return [header, rule, *rows]


def _q(
    values: Any, qs: tuple[float, ...] = (0.05, 0.5, 0.95), scale: float = 1.0
) -> str:
    v = scale * np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n/a"
    p = np.quantile(v, qs)
    return f"{p[1]:.3g} ({p[0]:.3g}–{p[2]:.3g})"


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _save(fig: Figure, path: Path, dpi: int = 100) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _fig_long_chirps(lib: TemplateLibrary, default: Frame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fs = lib.samplerate_hz
    centre = lib.n_samples // 2
    meta: Frame = default
    base = meta[meta["source"] == "chirp_chamber"]
    harvested = meta[meta["source"] == "competition"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.scatter(
        1e3 * base["duration_10_s"],
        base["excursion_hz"],
        s=3,
        color="0.65",
        label=f"chirp chamber (v2), n = {len(base)}",
        rasterized=True,
    )
    for k, (individual, group) in enumerate(
        harvested.groupby(harvested["individual"].replace("", "unknown"))
    ):
        ax.scatter(
            1e3 * group["duration_10_s"],
            group["excursion_hz"],
            s=7,
            color=f"C{k}",
            label=f"competition {individual}, n = {len(group)}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("full width at 10% (ms)")
    ax.set_ylabel("excursion (Hz)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("default selection", fontsize=9)
    ax = axes[1]
    bins = np.geomspace(4, 700, 60)
    ax.hist(1e3 * base["duration_10_s"], bins=bins, histtype="step", lw=1.4, label="v2")
    ax.hist(
        1e3 * harvested["duration_10_s"],
        bins=bins,
        histtype="step",
        lw=1.4,
        label="competition",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("full width at 10% (ms)")
    ax.set_ylabel("templates")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    long_ids = harvested.loc[harvested["duration_10_s"] >= 0.3, "template_id"]
    t = (np.arange(lib.n_samples) - centre) / fs
    for template_id in long_ids:
        row = int(lib.rows_of([int(template_id)])[0])
        ax.plot(t, lib.delta_f_hz[row], lw=0.6, alpha=0.7)
    ax.set_xlim(-0.35, 0.65)
    ax.set_xlabel("time from alignment point (s)")
    ax.set_ylabel("Δf (Hz)")
    ax.set_title(f"{len(long_ids)} harvested templates ≥ 300 ms at 10%", fontsize=9)
    _save(fig, path)


def _fig_sampling(
    lib: TemplateLibrary, schemes: list[Scheme], table: pd.DataFrame, path: Path
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), constrained_layout=True)
    classes = [c[0] for c in DURATION_CLASSES_MS]
    x = np.arange(len(classes) + 1)
    width = 0.8 / len(schemes)
    for k, scheme in enumerate(schemes):
        row = table[table["scheme"] == scheme.label].iloc[0]
        values = [float(row[c]) for c in classes] + [float(row["big"])]
        axes[0].bar(x + k * width, values, width, label=scheme.label)
    axes[0].set_xticks(x + 0.4 - width / 2, [*classes, "big (≥ 200 Hz)"])
    axes[0].set_ylabel("share of draws")
    axes[0].legend(fontsize=7, frameon=False)
    eq = lib.equalization_bins(DEFAULT_BINS, spacing=DEFAULT_SPACING)
    image = axes[1].imshow(
        np.log10(np.where(eq.counts > 0, eq.counts, np.nan)),
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[1].set_xticks(
        range(eq.counts.shape[1]),
        [f"{1e3 * v:.0f}" for v in eq.duration_edges_s[:-1]],
        fontsize=7,
    )
    axes[1].set_yticks(
        range(eq.counts.shape[0]),
        [f"{v:.0f}" for v in eq.excursion_edges_hz[:-1]],
        fontsize=7,
    )
    axes[1].set_xlabel("width at 10%, lower bin edge (ms)")
    axes[1].set_ylabel("excursion, lower bin edge (Hz)")
    axes[1].set_title("templates per log bin (log10)", fontsize=9)
    fig.colorbar(image, ax=axes[1])
    _save(fig, path)


def _fig_brief(lib: TemplateLibrary, ids: npt.NDArray[np.int64], path: Path) -> None:
    import matplotlib.pyplot as plt

    fs = lib.samplerate_hz
    centre = lib.n_samples // 2
    n = len(ids)
    columns = 6
    rows = max(int(np.ceil(n / columns)), 1)
    fig, axes = plt.subplots(
        rows, columns, figsize=(2.6 * columns, 1.9 * rows), squeeze=False
    )
    meta: Frame = lib.metadata.set_index("template_id")
    t = 1e3 * (np.arange(lib.n_samples) - centre) / fs
    for ax, template_id in zip(axes.ravel(), ids, strict=False):
        row = int(lib.rows_of([int(template_id)])[0])
        ax.plot(t, lib.delta_f_hz[row], lw=0.8)
        m = meta.loc[int(template_id)]
        cycles = float(m["duration_50_s"]) * float(m["baseline_hz"])
        ax.set_xlim(-15, 15)
        ax.set_title(
            f"{int(template_id)}: {1e3 * float(m['duration_10_s']):.1f} ms at 10%, "
            f"{cycles:.1f} cycles, {float(m['baseline_hz']):.0f} Hz",
            fontsize=6,
        )
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(
        "Brief events kept by the 5 ms cut at 10% width (FWHM ≤ 2 EOD cycles); "
        "x: ms from alignment point",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def write_merged_report(
    library_dir: Path | str,
    harvest_out_dir: Path | str,
    *,
    n_draws: int = 100_000,
    seed: int = 0,
) -> Path:
    """Write ``report.md`` and figures for a merged library.

    Parameters
    ----------
    library_dir
        Library written by :func:`gridtools.templates.merge.build_merged_library`.
    harvest_out_dir
        Harvest output directory (``candidates.csv``, ``recordings.csv``,
        ``contact_sheets/``).
    n_draws
        Draw count for the repetition statistics.
    seed
        Seed for example picks.

    Returns
    -------
    pathlib.Path
        Path of the report.
    """
    library_dir, harvest_out_dir = Path(library_dir), Path(harvest_out_dir)
    fig_dir = library_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    lib = load_library(library_dir, mmap=True)
    manifest = lib.manifest
    meta: Frame = lib.metadata
    default = lib.select()
    dmeta: Frame = default.metadata
    candidates: Frame = pd.read_csv(harvest_out_dir / "candidates.csv")
    recordings: Frame = pd.read_csv(harvest_out_dir / "recordings.csv")
    harvest_cfg = manifest.get("harvest_config", {})
    lines: list[str] = []
    add = lines.append

    from gridtools.templates.build import _code_version

    code = _code_version()
    fs = lib.samplerate_hz
    n_base = int(manifest["source"]["base_library"]["n_templates"])
    harvested: Frame = meta[meta["source"] == "competition"]
    h_default: Frame = dmeta[dmeta["source"] == "competition"]
    counts = manifest["counts"]

    add("# Chirp template library v3: build report\n")
    add(
        f"Generated {datetime.datetime.now(tz=datetime.UTC):%Y-%m-%d %H:%M} UTC. "
        f"Library built by gridtools {manifest['code']['gridtools_version']} at "
        f"`{(manifest['code'].get('git_sha') or 'unknown')[:10]}`"
        f"{' (dirty)' if manifest['code'].get('git_dirty') else ''}, report by "
        f"`{(code.get('git_sha') or 'unknown')[:10]}`"
        f"{' (dirty)' if code.get('git_dirty') else ''}. "
        f"**{lib.n_templates} templates** × {lib.n_samples} samples at {fs:.0f} Hz "
        f"({1e3 * lib.n_samples / fs:.0f} ms), alignment point at sample "
        f"{lib.n_samples // 2}: the {n_base} chirp-chamber templates of v2 "
        f'(`source = "chirp_chamber"`, same ids 0–{n_base - 1}, padded) plus '
        f"{len(harvested)} chirps harvested from the competition recordings "
        f'(`source = "competition"`, ids {n_base}–{lib.n_templates - 1}). '
        "The chirp-chamber part is documented in `../v2/report.md`; v1 and v2 are "
        "unchanged.\n"
    )

    add("## What changed since v2\n")
    add(
        "- **Atypical-EODf group kept** (author: “maybe *Eigenmannia*, let's keep them "
        "in”). `eodf_atypical` is metadata only and no longer in `DEFAULT_EXCLUDE`: "
        f"{counts['informational_flags']['eodf_atypical']['in_default_selection']} of "
        f"its {counts['informational_flags']['eodf_atypical']['all']} templates are in "
        "the default selection. Templates are baseline-relative, so they apply to any "
        "fish.\n"
        f"- **Minimum duration** (author: “nothing below 5 ms is real”): new flag "
        f"`too_short` (full width at 10% < {1e3 * MIN_DURATION_S:.0f} ms), excluded by "
        "default (section 4).\n"
        "- **Equalization**: log-spaced bins and tempered weights; default "
        f"`sample(rng, n)` = log {DEFAULT_BINS[0]}×{DEFAULT_BINS[1]} bins, "
        f"`alpha = {DEFAULT_ALPHA}` (section 5). The v2 scheme stays available as "
        '`spacing="linear", alpha=1.0`.\n'
        f"- **Long chirps from the competition recordings** (section 1): "
        f"{len(harvested)} templates, {int((harvested['duration_10_s'] >= 0.3).sum())} "
        "of them ≥ 300 ms at 10% of the excursion (v2: none above 151 ms).\n"
        f"- **Template length** {1e3 * lib.n_samples / fs:.0f} ms instead of 400 ms "
        "(section 2).\n"
    )

    # 1. long chirps -----------------------------------------------------------
    add("## 1. Long chirps from competition recordings\n")
    n_rec = len(recordings)
    with_cand = recordings[recordings["n_candidates"] > 0]
    status = candidates["status"].value_counts()
    add(
        "**Candidates.** chirpdetector v1 boxes at least "
        f"{1e3 * harvest_cfg.get('min_box_s', 0.25):.0f} ms wide and "
        f"{harvest_cfg.get('min_height_hz', 150):.0f} Hz high (box width ≈ chirp + "
        f"~160 ms of spectrogram smear), in all {n_rec} recordings under "
        "`/mnt/data2/2019_tube_competition` (read only; short spans around each "
        f"candidate). {int(recordings['n_candidates'].sum())} candidates in "
        f"{len(with_cand)} recordings; {int(status.get('harvested', 0))} harvested, "
        f"{int(status.get('rejected', 0))} rejected. Four 2020 recordings end with a "
        "partial frame of the raw file (the writer stopped mid-frame); their complete "
        "frames are used. The 2019 recordings have 9 channels but `sign_v` has 8 "
        "columns, so electrode powers come from the raw data everywhere; where `sign_v` "
        "maps (all 2020 recordings), its strongest electrode agrees with the data's in "
        f"{100 * float(np.mean(harvested['sign_v_best_channel'] == harvested['best_channel'])):.0f}% "
        "of the harvested chirps.\n"
    )
    per_rec = with_cand.copy()
    reason_cols = [c for c in per_rec.columns if c.startswith("rejected_")]
    per_rec["rejected"] = (
        per_rec[reason_cols]
        .apply(
            lambda r: ", ".join(
                f"{c.removeprefix('rejected_')} {int(v)}" for c, v in r.items() if v
            ),
            axis=1,
        )
        .replace("", "·")
    )
    per_rec["individuals"] = (
        per_rec["individuals"]
        .astype(str)
        .replace({"nan": "", "0": "", "0.0": "", "None": ""})
    )
    long_counts = harvested.groupby("recording")["duration_10_s"].apply(
        lambda x: int((x >= 0.3).sum())
    )
    per_rec["≥ 300 ms"] = [int(long_counts.get(r, 0)) for r in per_rec["recording"]]
    table = per_rec[
        [
            "recording",
            "n_detections",
            "n_candidates",
            "n_harvested",
            "≥ 300 ms",
            "rejected",
            "individuals",
        ]
    ].astype({"n_detections": int, "n_candidates": int, "n_harvested": int})
    lines.extend(_table(table))
    add(
        "\nThe other "
        f"{n_rec - len(with_cand)} recordings have no candidate box. Individuals come "
        "from `order_meta.csv` (`g<group>f<fish>` for the track ids `rec_id1/2`); "
        "2020-03-10 is a failed trial without an assignment.\n"
    )
    reasons = candidates.loc[
        candidates["status"] == "rejected", "reason"
    ].value_counts()
    explain = {
        "no_emitter": "no suspect's baseline band lost ≥ 6 dB during the chirp core "
        "(no chirp of a tracked fish, or a weak overlapping event)",
        "baseline_mismatch": "extracted baseline ≠ departure frequency by > 5 Hz "
        "(extraction followed something else)",
        "no_chirp": "extracted excursion < 100 Hz",
        "onset_outside_box": "the strongest excursion in the box starts before it "
        "(a neighbouring chirp)",
        "fish_in_span": "another fish's EODf inside (baseline, chirp top)",
        "harmonic_in_span": "another fish's second harmonic inside (baseline, chirp top)",
        "duplicate": "a second v1 box on the same chirp",
    }
    add("**Rejections by reason** (`candidates.csv` has every candidate):\n")
    add("| reason | n | meaning |\n|---|---:|---|")
    for reason, n in reasons.items():
        add(f"| `{reason}` | {int(n)} | {explain.get(str(reason), '')} |")
    add(
        "\nContamination rule as specified: a candidate is rejected when another fish's "
        "EODf or second harmonic lies inside the chirp's frequency span, taken as "
        "(baseline, max(baseline + excursion + "
        f"{harvest_cfg.get('contamination_margin_hz', 20):.0f} Hz, v1 box top)). The box "
        "top enters because a line near the chirp top hides the top from the "
        "extraction (the box overestimates the top by its smear, median 86 Hz). In "
        "these trials the second fish is 80–130 Hz below the emitter and its second "
        "harmonic above the chirps, so only "
        f"{int(reasons.get('fish_in_span', 0) + reasons.get('harmonic_in_span', 0))} "
        "candidates go for contamination.\n"
    )
    hint_counts: dict[bool | None, int] = {True: 0, False: 0, None: 0}
    for value in harvested["hint_agrees"]:
        hint_counts[None if value is None or value is pd.NA else bool(value)] += 1
    add(
        "**Extraction.** Emitter: every track alive near the box is a suspect (v1's "
        "track only a hint); the emitter is the one whose baseline band loses most power "
        "during the chirp core. v1's track agreed in "
        f"{hint_counts[True]} harvested chirps, disagreed in "
        f"{hint_counts[False]}, and v1 had no track in "
        f"{hint_counts[None]}. All electrodes are band-passed "
        "(4th-order Butterworth, forward-backward) from 50 Hz below the baseline to "
        "80 Hz above the chirp top and combined with weights matched to the emitter's "
        "field pattern, with the other fish's fundamental and second-harmonic patterns "
        f"projected out ({100 * float(harvested['null_applied'].mean()):.0f}% of chirps; "
        "the emitter keeps "
        f"{_q(harvested['null_keep'])} of its matched gain); their lines outside the "
        "chirp are notched. The frequency is the power-weighted phase advance of the "
        "analytic signal (pulse-pair estimator), window 1 ms at baseline amplitude "
        "widening to 8 ms where the amplitude drops (`(A0/A)²`), refined by three "
        "rounds of demodulation (±150, ±80, ±50 Hz). Per-cycle zero crossings, as in "
        "the thesis extraction, give ±300 Hz jumps on the plateau of these chirps, "
        "whose EOD amplitude falls to 5–20% of baseline; the power-weighted estimate "
        "does not, and demodulation removes most of its bias toward the band centre "
        "(without it plateaus came out 20–50 Hz low with dips). The amplitude envelope "
        "is the analytic magnitude, corrected for the filter gain. Baseline = the "
        "frequency the chirp departs from (in dense trains the EODf between chirps "
        "can sit well above its level elsewhere in the window). Then the library's "
        "own construction (`template_from_traces`): alignment on the 50% crossings, "
        "raised-cosine taper to exact 0 / 1, measurements, with the **decay "
        "extension** switched on: the support runs until the 5 ms-smoothed trace has "
        "fallen to max(2% of the excursion, 3 Hz) (or the trough before a following "
        "chirp), plus 30 ms. This adds "
        f"{_q(harvested['decay_extension_s'], scale=1e3)} ms after the 10% offset "
        "(median, 5–95%); the v2 rule (10% offset + 30 ms) would have tapered the "
        "slow decays away. v2 templates are untouched (the extension is off by "
        "default and all 13,015 rebuild bit for bit).\n"
    )
    add("**Harvested templates** (all, median (5–95%)):\n")
    add("| | competition | v2 default selection, big |\n|---|---|---|")
    v2_big = dmeta[(dmeta["source"] == "chirp_chamber") & (dmeta["family"] == "big")]
    for label, column, scale in (
        ("full width at 10% (ms)", "duration_10_s", 1e3),
        ("full width at 50% (ms)", "duration_50_s", 1e3),
        ("excursion (Hz)", "excursion_hz", 1.0),
        ("baseline EODf (Hz)", "baseline_hz", 1.0),
    ):
        add(
            f"| {label} | {_q(harvested[column], scale=scale)} | "
            f"{_q(v2_big[column], scale=scale)} |"
        )
    support_ms = 1e3 * (harvested["support_stop_s"] - harvested["support_start_s"])
    add(f"| support incl. decay (ms) | {_q(support_ms)} | |")
    d10 = 1e3 * harvested["duration_10_s"]
    add(
        f"\nWidth at 10%: {int((d10 >= 100).sum())} ≥ 100 ms, {int((d10 >= 150).sum())} "
        f"≥ 150 ms, {int((d10 >= 200).sum())} ≥ 200 ms, {int((d10 >= 300).sum())} ≥ "
        f"300 ms, {int((d10 >= 400).sum())} ≥ 400 ms, longest {d10.max():.0f} ms "
        f"(template {int(harvested.loc[d10.idxmax(), 'template_id'])}). "
        f"{len(h_default)} of the {len(harvested)} are in the default selection. "
        "`figures/long_chirps.png`: excursion vs width, width histograms and the "
        "templates ≥ 300 ms overlaid.\n"
    )
    by_ind = (
        harvested.assign(individual=harvested["individual"].replace("", "unassigned"))
        .groupby("individual")
        .agg(
            n=("template_id", "size"),
            recordings=("recording", lambda x: len(set(x))),
            long=("duration_10_s", lambda x: int((x >= 0.3).sum())),
            longest=("duration_10_s", lambda x: f"{1e3 * x.max():.0f}"),
        )
        .sort_values("n", ascending=False)
        .reset_index()
    )
    by_ind.columns = [
        "individual",
        "templates",
        "recordings",
        "≥ 300 ms",
        "longest (ms)",
    ]
    add(
        "**Provenance: few individuals.** One fish (g6f5, in 2020-05-29, 2020-06-08 and "
        "2020-06-11) contributes most templates, and every chirp ≥ 300 ms comes from "
        "two fish (g6f1, g6f5), both of group 6:\n"
    )
    lines.extend(_table(by_ind))
    add(
        "\nThe chirps are therefore not independent samples of the species' "
        "repertoire; `inferred_group` is one group per recording and emitter track "
        "(`group_label` = `<recording>/track<id>`), after the v2 groups. Every row "
        "records `recording`, `emitter_track_id`, `individual`, `onset_10_abs_s` "
        "(recording time of the 10% onset), `electrodes` and `electrode_weights`, "
        "`band_low_hz`/`band_high_hz`, `notches_hz`, the v1 box, and the harvest "
        "configuration is in the manifest.\n"
    )
    flag_text = {
        "line_in_span": "a stationary spectral line (untracked source, e.g. ~1190 Hz "
        "in 2020-06-08/-11) inside the span within 35 dB of the baseline peak; the "
        "plateau, at 5–20% amplitude, can be pulled toward it",
        "low_snr": "amplitude below 10% of baseline over > 30% of the support "
        "(plateau frequency less certain)",
        "baseline_drift": "post-chirp level > 10 Hz from the baseline (chirp in a dense "
        "train on an elevated EODf; the template still tapers to 0)",
        "decay_interrupted": "the next chirp started before the decay reached its level",
        "multi_peak": "second maximum outside the chirp",
        "ripple_flag": "baseline ripple > 8% of the excursion",
        "step_flag": "single-sample jumps",
        "truncated": "support runs into the window edge",
    }
    add(
        "**QC flags** (none rejects; shown in `review.csv` and on the contact sheets):\n"
    )
    add("| flag | harvested | meaning |\n|---|---:|---|")
    for flag, text in flag_text.items():
        if flag in harvested.columns:
            add(
                f"| `{flag}` | {int(harvested[flag].fillna(False).astype(bool).sum())} | "
                f"{text} |"
            )
    add(
        "\n**Review.** Contact sheets of every harvested template: "
        f"`{harvest_out_dir}/contact_sheets/sheet_NN.png` (spectrogram cutout of the "
        "combined signal with the extracted EODf, white inside the template support, "
        "and the tapered template below; titles carry the `harvest_id` and flags). "
        f"To veto, set `accept` to False in `{harvest_out_dir}/review.csv` and rebuild "
        "(`build_merged_library`); the sheet survives a re-run of the harvest, and the "
        "build copies it into the library and lists the vetoed keys in the manifest "
        f"(now: {len(manifest['source']['review']['vetoed'])} vetoed). Rejected "
        f"examples: `{harvest_out_dir}/rejections.png`.\n"
    )

    # 2. template length ---------------------------------------------------------
    add("## 2. Template length\n")
    extent = 1e3 * max(
        -float(harvested["support_start_s"].min()),
        float(harvested["support_stop_s"].max()),
    )
    size_gb = 2 * lib.n_templates * lib.n_samples * 4 / 1e9
    add(
        f"Templates stay dense, fixed-length float32 arrays ({lib.n_templates} × "
        f"{lib.n_samples}), centred on the alignment point and padded with exact 0 "
        "(frequency) and 1 (amplitude), so `TemplateLibrary`, `get`, `sample` and "
        "`TemplateSample.delta_f_hz`/`amp_rel` (2-D, n × L) are unchanged and the scene "
        "simulator's row access to the memory-mapped arrays keeps working. The length "
        "is the smallest multiple of 0.1 s that holds the widest support: the decay of "
        f"the longest chirp ends {extent:.0f} ms from its alignment point (the "
        "alignment point is the midpoint of the 50% crossings, and the slow decay "
        f"makes long chirps asymmetric), so L = {1e3 * lib.n_samples / fs:.0f} ms. "
        f"Cost: {size_gb:.1f} GB on disk for both arrays (v2: 0.83 GB); with "
        "`load_library(path, mmap=True)` only drawn rows are read, so memory stays at "
        "a few rows per chirp (96 kB each); `select()` copies its rows, so prefer a "
        "metadata-only selection (as the simulator does) or `mmap=True` plus "
        "`get`/`sample`. Ragged storage would save ~2 GB of disk but break direct row "
        "access; cropping v2 rows to their support at draw time would change their "
        "length per row. The v2 rows are byte-identical after removing the padding.\n"
    )

    # 3. too short ---------------------------------------------------------------
    pool: Frame = lib.select(
        exclude=("invalid", "has_nan", "duplicate", "truncated", "step_flag")
    ).metadata
    t10 = pool["duration_10_s"] < MIN_DURATION_S
    t50 = pool["duration_50_s"] < MIN_DURATION_S
    atyp = pool["eodf_atypical"].astype(bool)
    cycles = pool["duration_50_s"] * pool["baseline_hz"]
    kept_brief = ~t10 & (cycles <= 2.0)
    add("## 3. Minimum duration (5 ms)\n")
    add(
        "Measured on the pool before the cut (default exclusions except `too_short`, "
        f"{len(pool)} templates):\n\n"
        "| criterion | below 5 ms | of these atypical-EODf (*Eigenmannia*?) |\n"
        "|---|---:|---:|\n"
        f"| full width at 10% (used) | {int(t10.sum())} | {int((t10 & atyp).sum())} |\n"
        f"| full width at 50% (FWHM) | {int(t50.sum())} | {int((t50 & atyp).sum())} |\n"
    )
    add(
        f"The 10% cut removes only {int(t10.sum())} templates. It **does leave visible "
        f"one- and two-cycle events**: {int(kept_brief.sum())} kept templates have a "
        "FWHM of at most two EOD cycles (FWHM × baseline EODf ≤ 2), "
        f"{int((kept_brief & atyp).sum())} of them in the atypical-EODf group (at "
        "320–440 Hz one cycle lasts 2.3–3.1 ms, so nearly that whole group is one- or "
        "two-cycle events) and "
        f"{int((kept_brief & ~atyp).sum())} at 700–900 Hz (FWHM of two cycles or "
        "less, about 2.5 ms; the brief events of the v2 report). Their 10% widths "
        "exceed 5 ms because their flanks are broad. FWHM < 5 ms would remove "
        f"{int(t50.sum())} instead, {int((t50 & atyp).sum())} of them atypical. Not "
        "applied: the decision is the author's (`figures/brief_events.png`, "
        "`exclude=(..., 'too_short')` plus a FWHM filter on the metadata if wanted).\n"
    )

    # 4. exclusions --------------------------------------------------------------
    add("## 4. Exclusions (default selection)\n")
    add("| criterion | flagged | left |\n|---|---:|---:|")
    add(f"| all rows | | {counts['cumulative']['all']} |")
    for name, flagged in counts["flagged"].items():
        add(f"| {name} | {flagged} | {counts['cumulative'][f'after_{name}']} |")
    add(
        f"\nDefault selection: **{default.n_templates}** templates "
        f"({counts['default_selection_per_family']['small']} small, "
        f"{counts['default_selection_per_family']['big']} big), "
        f"{len(dmeta) - len(h_default)} chirp-chamber and {len(h_default)} competition "
        f"({len(harvested) - len(h_default)} harvested templates carry a flag that the "
        "default selection drops).\n"
    )

    # 5. equalization ------------------------------------------------------------
    add("## 5. Histogram equalization\n")
    schemes = [
        Scheme("natural", False),
        Scheme("v2: linear 10×10, α = 1", True, "linear", 1.0),
        Scheme("log 10×10, α = 1", True, "log", 1.0),
        Scheme("log 10×10, α = 0.8", True, "log", 0.8),
        Scheme("log 10×10, α = 0.7", True, "log", 0.7),
        Scheme(
            f"**log 10×10, α = {DEFAULT_ALPHA} (default)**", True, "log", DEFAULT_ALPHA
        ),
        Scheme("log 10×10, α = 0.5", True, "log", 0.5),
    ]
    rows = [sampling_summary(default, s, n_draws) for s in schemes]
    mix_ids, p_eq = draw_probabilities(default, Scheme("", True, "log", 1.0))
    comparison = pd.DataFrame(rows)
    shown = comparison.copy()
    for column in ["big", "small", *(c[0] for c in DURATION_CLASSES_MS), "competition"]:
        shown[column] = shown[column].map(_fmt_share)
    shown["distinct"] = shown["distinct"].map(lambda v: f"{v:,.0f}")
    shown["effective"] = shown["effective"].map(lambda v: f"{v:,.0f}")
    shown["max_repeats"] = shown["max_repeats"].map(lambda v: f"{v:,.0f}")
    shown.columns = [
        "scheme",
        "big",
        "small",
        *(c[0] for c in DURATION_CLASSES_MS),
        "competition",
        f"distinct / {n_draws // 1000}k",
        "effective n",
        f"max repeats / {n_draws // 1000}k",
    ]
    eq = default.equalization_bins(DEFAULT_BINS, spacing=DEFAULT_SPACING)
    add(
        "Draw shares under each scheme on the default selection "
        f"({default.n_templates} templates; big = excursion ≥ 200 Hz; durations = full "
        "width at 10%). *distinct*: expected number of different templates in "
        f"{n_draws:,} draws; *effective n*: exp(entropy) of the draw distribution; "
        "*max repeats*: expected draws of the most likely template.\n"
    )
    lines.extend(_table(shown))
    linear = default.equalization_bins(DEFAULT_BINS, spacing="linear")
    add(
        "\nWith 10 linear bins the "
        f"{int(linear.counts.max()):,}"
        " ordinary small chirps shared one bin and full equalization gave chirps "
        "shorter than 20 ms only "
        f"{_fmt_share(rows[1]['< 20 ms'])} of draws; with the harvest, half of all "
        "draws would go to chirps ≥ 150 ms. Log bins split the small chirps over "
        "several excursion bins; tempering (a bin with c templates is drawn ∝ "
        "c^(1 − α), each of its templates ∝ c^−α; α = 1 full equalization, α = 0 "
        "natural) sets the balance. **Default α = "
        f"{DEFAULT_ALPHA}**: big chirps get {_fmt_share(rows[5]['big'])} of draws (about "
        "a third, natural "
        f"{_fmt_share(rows[0]['big'])}), small chirps keep the majority "
        f"({_fmt_share(rows[5]['small'])}) and their variety (about "
        f"{rows[5]['distinct']:,.0f} of {default.n_templates} templates drawn at least "
        f"once per {n_draws:,}), chirps ≥ 50 ms get "
        f"{_fmt_share(rows[5]['50–150 ms'] + rows[5]['≥ 150 ms'])} (natural "
        f"{_fmt_share(rows[0]['50–150 ms'] + rows[0]['≥ 150 ms'])}), and no template is "
        f"drawn more than {rows[5]['max_repeats']:,.0f} times per {n_draws:,}. A "
        "natural/equalized mixture with a similar big share (50/50 of natural and log "
        "α = 1) repeats its rarest templates "
        f"{n_draws * (0.5 / mix_ids.size + 0.5 * p_eq.max()):,.0f} times per "
        f"{n_draws:,}, tempering "
        f"{rows[5]['max_repeats']:,.0f}. Within a bin, templates still come in "
        "successive random permutations; unequal bin weights are drawn in shuffled "
        "systematic blocks (bin counts within one draw of their expectation per "
        "block). `figures/sampling.png`.\n"
    )
    add(
        f"Occupancy of the default bins ({eq.n_nonempty} of "
        f"{eq.counts.size} occupied; rows: excursion, columns: full width at 10%). "
        "The long-duration columns are filled almost only by the harvested chirps "
        "(v2's longest is 151 ms):\n"
    )
    lines.extend(_bin_table(eq.table()))
    p_tpl = eq.template_probabilities(DEFAULT_ALPHA)
    expected = n_draws * p_tpl
    dur_ms = 1e3 * default.metadata.set_index("template_id").loc[
        eq.template_ids, "duration_10_s"
    ].to_numpy(dtype=np.float64)
    add(
        f"\nExpected draws per template per {n_draws:,} (default scheme): all "
        f"{_q(expected)}; templates ≥ 300 ms {_q(expected[dur_ms >= 300])} "
        f"({int((dur_ms >= 300).sum())} templates, together "
        f"{_fmt_share(float(p_tpl[dur_ms >= 300].sum()))} of draws); 150–300 ms "
        f"{_q(expected[(dur_ms >= 150) & (dur_ms < 300)])}; < 20 ms "
        f"{_q(expected[dur_ms < 20])}. Repeating rare templates tens to a few hundred "
        "times per 100k is intended: what the detector learns from is where the chirp "
        "lies in the spectrogram, the noise around it and the other fish, which "
        "change on every draw.\n"
    )

    # 6. open questions ------------------------------------------------------------
    add("## 6. Open questions for the author\n")
    add(
        "1. Review the contact sheets; flagged cells (red titles) first, especially "
        "`line_in_span` and `low_snr`. Plateau dips of 100–300 Hz that the spectrogram "
        "does not show are estimation artefacts at near-zero amplitude.\n"
        "2. Two individuals produce all chirps ≥ 300 ms, one fish 79% of the harvest. "
        "Cap templates per individual, or accept the bias (equalization already spreads "
        "draws over the bins, not the individuals)?\n"
        "3. One- and two-cycle events survive the 5 ms cut at 10% width (section 3), "
        "most of them the atypical-EODf group. Cut on FWHM (or on cycles) instead?\n"
        "4. Chirps in dense trains on an elevated EODf (`baseline_drift`) are kept "
        "and tapered back to the departure frequency; veto them if the residual step "
        "matters.\n"
    )
    _fig_long_chirps(lib, dmeta, fig_dir / "long_chirps.png")
    _fig_sampling(
        default, [*schemes[1:6:2], schemes[0]], comparison, fig_dir / "sampling.png"
    )
    brief_ids = pool.loc[kept_brief, "template_id"].to_numpy(dtype=np.int64)
    if brief_ids.size:
        picks = np.sort(
            rng.choice(brief_ids, size=min(24, brief_ids.size), replace=False)
        )
        _fig_brief(lib, picks, fig_dir / "brief_events.png")
    add("## Files\n")
    add(
        "- `delta_f_hz.npy`, `amp_rel.npy`: float32 (n, "
        f"{lib.n_samples}); row i ↔ `template_id` i.\n"
        "- `templates.parquet`: metadata (v2 columns plus `source`, `group_label`, "
        "`too_short`, `review_accept`, `review_note` and, for harvested rows, the "
        "harvest provenance and QC columns).\n"
        "- `manifest.json`: base and harvest manifests' hashes and configurations, "
        "review sheet hash and vetoed keys, code version, output hashes, counts.\n"
        "- `review.csv`: copy of the review sheet used.\n"
        "- `figures/`: long_chirps, sampling, brief_events.\n"
    )
    path = library_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    (library_dir / "sampling_comparison.json").write_text(
        json.dumps(rows, indent=2, default=float)
    )
    log.info("report written to %s", path)
    return path


# --------------------------------------------------------------------------
# Library v4: clean samples and fish-first sampling
# --------------------------------------------------------------------------


def _duration_class(duration_s: Any) -> npt.NDArray[np.str_]:
    ms = 1e3 * np.asarray(duration_s, dtype=np.float64)
    out = np.full(ms.shape, "", dtype=object)
    for label, lo, hi in DURATION_CLASSES_MS:
        out[(ms >= lo) & (ms < hi)] = label
    return np.asarray(out, dtype=str)


def write_v4_report(
    library_dir: Path | str,
    *,
    previous: str = "../v3/report.md",
    n_draws: int = 100_000,
) -> Path:
    """Write ``report.md`` of library v4 (clean samples, fish-first sampling).

    Library v4 has the arrays and harvest of v3; this report covers the
    author decisions that change its default selection and sampling. The
    harvest and template-length sections live in ``previous``.

    Parameters
    ----------
    library_dir
        Library written by :func:`gridtools.templates.merge.build_merged_library`.
    previous
        Path (relative to the library) of the v3 report, for references.
    n_draws
        Draw count for the repetition statistics.

    Returns
    -------
    pathlib.Path
        Path of the report.
    """
    from gridtools.templates.build import _code_version
    from gridtools.templates.library import flag_mask_of

    library_dir = Path(library_dir)
    lib = load_library(library_dir, mmap=True)
    manifest = lib.manifest
    meta: Frame = lib.metadata
    default = lib.select()
    dmeta: Frame = default.metadata
    code = _code_version()
    fs = lib.samplerate_hz
    competition = (meta["source"] == "competition").to_numpy()
    atypical = meta["eodf_atypical"].to_numpy(dtype=bool)
    lines: list[str] = []
    add = lines.append

    add("# Chirp template library v4: build report\n")
    add(
        f"Generated {datetime.datetime.now(tz=datetime.UTC):%Y-%m-%d %H:%M} UTC. "
        f"Library built by gridtools {manifest['code']['gridtools_version']} at "
        f"`{(manifest['code'].get('git_sha') or 'unknown')[:10]}`"
        f"{' (dirty)' if manifest['code'].get('git_dirty') else ''}, report by "
        f"`{(code.get('git_sha') or 'unknown')[:10]}`"
        f"{' (dirty)' if code.get('git_dirty') else ''}. "
        f"{lib.n_templates} templates × {lib.n_samples} samples at {fs:.0f} Hz: the "
        "same arrays, harvest and template length as v3 (v2 chirp-chamber templates "
        f"plus {int(competition.sum())} long chirps from the competition recordings; "
        f"see `{previous}`). v4 changes the default selection and the sampling. v1–v3 "
        "are unchanged.\n"
    )
    add("## What changed since v3 (author decisions)\n")
    add(
        "1. **Fish-first sampling** (“keep them all, spread by fish”): within every "
        "equalization bin an individual is drawn first, uniformly among those present, "
        'then one of its templates; `balance="individual"` is the new default of '
        '`sample` and `cycler` (`balance="template"` gives v3). Bins and weights '
        f"unchanged: log {DEFAULT_BINS[0]}×{DEFAULT_BINS[1]}, α = {DEFAULT_ALPHA}.\n"
        f"2. **One- and two-cycle events cut** (“yup, let's cut them”): new flag "
        f"`brief_cycles` (FWHM × baseline EODf ≤ {BRIEF_CYCLES_MAX:g}), excluded by "
        "default.\n"
        "3. **Only clean samples** (“veto them all”): every quality flag excludes by "
        "default, for chirp-chamber and harvested templates alike: `ripple_flag`, "
        "`multi_peak`, `line_in_span`, `baseline_drift`, `low_snr`, "
        "`decay_interrupted`, besides the artefact flags, `too_short` and "
        "`brief_cycles`. `amp_ripple_flag` (measured amplitude only; it marks 70% of "
        "the templates) stays informational, as does `eodf_atypical`. The flagged "
        "templates stay in the library; `select(exclude=...)` brings them back.\n"
    )

    # exclusions -------------------------------------------------------------
    add("## 1. Exclusions\n")
    add(
        "Each criterion on its own (all templates, chirp chamber, competition) and "
        "what is left after applying the criteria in order:\n"
    )
    add("| criterion | flagged | chirp chamber | competition | left |")
    add("|---|---:|---:|---:|---:|")
    keep = np.ones(len(meta), dtype=bool)
    add(f"| all templates | | | | {len(meta)} |")
    kind = meta["duplicate_kind"].fillna("").to_numpy()
    for name in DEFAULT_EXCLUDE:
        parts = (
            [("exact duplicate", kind == "exact"), ("near duplicate", kind == "near")]
            if name == "duplicate"
            else [(f"`{name}`", flag_mask_of(meta, name))]
        )
        for label, mask in parts:
            keep &= ~mask
            add(
                f"| {label} | {int(mask.sum())} | {int((mask & ~competition).sum())} | "
                f"{int((mask & competition).sum())} | {int(keep.sum())} |"
            )
    assert int(keep.sum()) == default.n_templates
    brief = flag_mask_of(meta, "brief_cycles")
    pool_before = np.ones(len(meta), dtype=bool)
    for name in DEFAULT_EXCLUDE:
        if name not in ("brief_cycles",):
            pool_before &= ~flag_mask_of(meta, name)
    cut = brief & pool_before
    left_atyp: Frame = dmeta[dmeta["eodf_atypical"].astype(bool)]
    add(
        f"\n**`brief_cycles`.** {int(brief.sum())} templates have a FWHM of at most "
        f"{BRIEF_CYCLES_MAX:g} cycles of their baseline EODf: "
        f"{int((brief & atypical).sum())} of the {int(atypical.sum())} atypical-EODf "
        f"(320–440 Hz) templates and {int((brief & ~atypical).sum())} others "
        f"({int((brief & competition).sum())} harvested). Among templates that pass "
        f"every other criterion the cut removes {int(cut.sum())}: "
        f"{int((cut & atypical).sum())} atypical and {int((cut & ~atypical).sum())} "
        f"others. What remains of the atypical group in the default selection: "
        f"{len(left_atyp)} template{'s' if len(left_atyp) != 1 else ''}"
        + (
            " ("
            + ", ".join(
                f"id {int(r.template_id)}: {r.excursion_hz:.0f} Hz, FWHM "
                f"{1e3 * r.duration_50_s:.1f} ms = "
                f"{r.duration_50_s * r.baseline_hz:.1f} cycles at {r.baseline_hz:.0f} Hz"
                for r in left_atyp.itertuples()
            )
            + ")"
            if len(left_atyp)
            else ""
        )
        + ". Templates without a recovered baseline EODf "
        f"({int(np.isnan(meta['baseline_hz'].to_numpy(dtype=float)).sum())}) cannot "
        "be counted in cycles and are not flagged.\n"
    )
    add("**Default selection** by family and duration class (full width at 10%):\n")
    classes = _duration_class(dmeta["duration_10_s"])
    header = [c[0] for c in DURATION_CLASSES_MS]
    add("| | " + " | ".join(header) + " | total |")
    add("|---|" + "---:|" * (len(header) + 1))
    for label, mask in (
        ("small (< 200 Hz)", dmeta["family"].to_numpy() == "small"),
        ("big (≥ 200 Hz)", dmeta["family"].to_numpy() == "big"),
        ("of which competition", dmeta["source"].to_numpy() == "competition"),
        ("all", np.ones(len(dmeta), dtype=bool)),
    ):
        cells = [str(int((mask & (classes == h)).sum())) for h in header]
        add(f"| {label} | " + " | ".join(cells) + f" | {int(mask.sum())} |")
    long_left = int((dmeta["duration_10_s"] >= 0.3).sum())
    add(
        f"\nDefault selection: **{default.n_templates}** templates "
        f"({int((dmeta['source'] == 'chirp_chamber').sum())} chirp chamber, "
        f"{int((dmeta['source'] == 'competition').sum())} competition, of these "
        f"{long_left} ≥ 300 ms; longest "
        f"{1e3 * float(dmeta['duration_10_s'].max()):.0f} ms).\n"
    )

    # fish-first ---------------------------------------------------------------
    add("## 2. Fish-first sampling\n")
    labels = default.individuals()
    n_ind = np.unique(labels).size
    n_cc = np.unique(labels[(dmeta["source"] == "chirp_chamber").to_numpy()]).size
    n_comp = np.unique(labels[(dmeta["source"] == "competition").to_numpy()]).size
    add(
        "Individuals: harvested chirps use `group_label` = `<recording>/track<id>` "
        "(one wavetracker track of one recording; the same fish in two recordings "
        "counts twice, e.g. g6f5 in three recordings). Chirp-chamber chirps use "
        "`inferred_group`, runs of consecutive source rows whose baseline EODf "
        "changes by less than 10 Hz: the extraction appended chirps file by file, so "
        "a run approximates one recording, not reliably one fish. Two fish of similar "
        "EODf recorded one after the other merge into one group, a recording whose "
        "EODf jumps splits, and rows without a recovered EODf start new groups. The "
        f"default selection holds {n_ind} groups ({n_cc} chirp chamber, {n_comp} "
        "competition). Within a bin, groups come in successive random permutations "
        "and each group's templates in successive random permutations of its own, so "
        "no template repeats more often than the balance needs.\n"
    )
    v3_scheme = Scheme("v3: log, α = 0.6, template balance", True, "log", 0.6)
    v4_scheme = Scheme(
        "**v4: log, α = 0.6, individual balance**", True, "log", 0.6, "individual"
    )
    natural = Scheme("natural", False)
    effect = []
    for scheme in (natural, v3_scheme, v4_scheme):
        ind = individual_summary(default, scheme, n_draws)
        tpl = sampling_summary(default, scheme, n_draws)
        effect.append(
            {
                "scheme": scheme.label,
                "10 largest groups": _fmt_share(ind["top_share"]),
                f"effective groups / {n_draws // 1000}k": f"{ind['effective_individuals']:,.0f}",
                f"distinct groups / {n_draws // 1000}k": f"{ind['distinct_individuals']:,.0f}",
                f"distinct templates / {n_draws // 1000}k": f"{tpl['distinct']:,.0f}",
                f"max repeats / {n_draws // 1000}k": f"{tpl['max_repeats']:,.0f}",
            }
        )
    largest = individual_summary(default, v4_scheme, n_draws)["largest"]
    add(
        "Effect on the default selection (*10 largest groups*: share of draws from the "
        "10 groups with most templates; *effective groups*: exp(entropy) of the "
        "groups' draw shares; *distinct*: expected number drawn at least once):\n"
    )
    lines.extend(_table(pd.DataFrame(effect)))
    add(
        "\nThe 10 largest groups: "
        + ", ".join(f"`{name}` ({size})" for name, size in largest)
        + ".\n"
    )

    # sampling table -------------------------------------------------------------
    add("## 3. Sampling with the v4 defaults\n")
    schemes = [
        natural,
        Scheme("v2: linear 10×10, α = 1", True, "linear", 1.0),
        v3_scheme,
        Scheme("log, α = 1, individual", True, "log", 1.0, "individual"),
        Scheme("log, α = 0.7, individual", True, "log", 0.7, "individual"),
        v4_scheme,
        Scheme("log, α = 0.5, individual", True, "log", 0.5, "individual"),
    ]
    rows = [sampling_summary(default, s, n_draws) for s in schemes]
    shown = pd.DataFrame(rows)
    for column in ["big", "small", *(c[0] for c in DURATION_CLASSES_MS), "competition"]:
        shown[column] = shown[column].map(_fmt_share)
    for column in ("distinct", "effective", "max_repeats"):
        shown[column] = shown[column].map(lambda v: f"{v:,.0f}")
    shown.columns = [
        "scheme",
        "big",
        "small",
        *(c[0] for c in DURATION_CLASSES_MS),
        "competition",
        f"distinct / {n_draws // 1000}k",
        "effective n",
        f"max repeats / {n_draws // 1000}k",
    ]
    add(
        f"Draw shares on the v4 default selection ({default.n_templates} templates); "
        "columns as in the v3 report.\n"
    )
    lines.extend(_table(shown))
    v4 = rows[5]
    add(
        f"\nWith the v4 defaults big chirps get {_fmt_share(v4['big'])} of draws, small "
        f"chirps {_fmt_share(v4['small'])}, chirps ≥ 150 ms "
        f"{_fmt_share(v4['≥ 150 ms'])}, harvested chirps "
        f"{_fmt_share(v4['competition'])}; no template is drawn more than "
        f"{v4['max_repeats']:,.0f} times per {n_draws:,}. Bin shares are the same as "
        "with template balance; fish-first moves draws, within a bin, from prolific "
        "groups to the others.\n"
    )
    eq = default.equalization_bins(DEFAULT_BINS, spacing=DEFAULT_SPACING)
    add(
        f"Occupancy of the default bins ({eq.n_nonempty} of {eq.counts.size} occupied):\n"
    )
    lines.extend(_bin_table(eq.table()))
    groups_per_bin = np.zeros(eq.counts.size, dtype=np.int64)
    individuals = eq.individual_of if eq.individual_of is not None else eq.bin_of
    for b in np.flatnonzero(eq.counts.ravel()):
        groups_per_bin[b] = np.unique(individuals[eq.bin_of == b]).size
    add("\nGroups per bin (fish-first draws a group uniformly within its bin):\n")
    lines.extend(
        _bin_table(
            pd.DataFrame(
                groups_per_bin.reshape(eq.counts.shape),
                index=eq.table().index,
                columns=eq.table().columns,
            )
        )
    )
    add(
        "\n## Files\n\n- Same layout as v3: `delta_f_hz.npy`, `amp_rel.npy` "
        f"(float32, n × {lib.n_samples}), `templates.parquet` (every flag as a "
        "bool column, `group_label` for every template), `manifest.json`, "
        "`review.csv`.\n"
    )
    path = library_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    (library_dir / "sampling_comparison.json").write_text(
        json.dumps({"schemes": rows, "fish_first": effect}, indent=2, default=float)
    )
    log.info("report written to %s", path)
    return path
