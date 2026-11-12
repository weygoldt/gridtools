# ruff: noqa: RUF001
# (The report text uses en dashes, multiplication signs and Greek letters on purpose.)
"""Write ``calibration.json`` and a Markdown report with figures.

The figures need the optional ``plot`` extra (matplotlib), imported lazily.
"""

from __future__ import annotations

import datetime
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gridtools.calibrate.aggregate import Calibration

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["to_jsonable", "write_calibration_json", "write_report"]

# Categorical slots 1-3 of the reference palette, text inks and a context gray.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRAY = "#0b0b0b", "#52514e", "#a3a29c"


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and non-finite floats for JSON."""
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def write_calibration_json(
    calibration: Calibration, path: str | Path, provenance: Mapping[str, Any]
) -> None:
    """Write proposed parameters, new parameters and summaries as JSON.

    Parameters
    ----------
    calibration : Calibration
    path : str or Path
    provenance : mapping
        Code version, seed, data root, trials, configuration, ...
    """
    doc = {
        "version": "v1",
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "units": "recording units = raw ADC volts (electrode potential x 100); frequencies in Hz; times in s",
        "provenance": dict(provenance),
        "parameters": calibration.parameters,
        "proposed_new": calibration.proposed_new,
        "summary": calibration.summary,
    }
    Path(path).write_text(
        json.dumps(to_jsonable(doc), indent=1, allow_nan=False) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------- figures ----


def _style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig: Figure, path: Path) -> str:
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path.name


def _fig_eodf(cal: Calibration, path: Path) -> str:
    import matplotlib.pyplot as plt

    rows = cal.figures["fish_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, key, title in zip(
        axes,
        ("baseline_hz", "baseline_25c_hz"),
        ("at recording temperature", "normalised to 25 °C (Q10 1.37)"),
        strict=True,
    ):
        for role, color in (("winner", BLUE), ("loser", ORANGE)):
            for xi, sex in enumerate(("f", "m")):
                values = [r[key] for r in rows if r["role"] == role and r["sex"] == sex]
                jitter = rng.uniform(-0.12, 0.12, len(values)) + (
                    -0.15 if role == "winner" else 0.15
                )
                ax.scatter(
                    xi + jitter,
                    values,
                    s=14,
                    color=color,
                    edgecolor="#fcfcfb",
                    linewidth=0.6,
                    label=role if xi == 0 else None,
                )
        ax.set_xticks([0, 1], ["female", "male"])
        ax.set_title(title, fontsize=9)
        _style(ax)
    axes[0].set_ylabel("baseline EODf (Hz)")
    axes[0].legend(frameon=False, fontsize=8)
    return _save(fig, path)


def _fig_delta_f(cal: Calibration, path: Path) -> str:
    import matplotlib.pyplot as plt

    df = np.asarray(cal.figures["abs_delta_f"], dtype=float)
    df = df[np.isfinite(df) & (df > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    bins = np.geomspace(1, 400, 50)
    axes[0].hist(df, bins=bins, color=BLUE, edgecolor="#fcfcfb", linewidth=0.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("|ΔF| (Hz), 10-s samples")
    axes[0].set_ylabel("count")
    x = np.sort(df)
    axes[1].plot(x, np.arange(1, x.size + 1) / x.size, color=BLUE, linewidth=2)
    for th in (5, 10, 20, 50):
        frac = float(np.mean(df < th))
        axes[1].axvline(th, color=GRAY, linewidth=0.8, linestyle="--")
        axes[1].annotate(
            f"{100 * frac:.1f}%",
            (th, frac),
            fontsize=7,
            color=INK_2,
            xytext=(3, 4),
            textcoords="offset points",
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("|ΔF| (Hz)")
    axes[1].set_ylabel("fraction of time below")
    for ax in axes:
        _style(ax)
    return _save(fig, path)


def _fig_drift(cal: Calibration, path: Path) -> str:
    import matplotlib.pyplot as plt

    from gridtools.calibrate.frequency import DriftFit

    d = cal.figures["drift"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    ax = axes[0]
    for g in d["gammas"]:
        ax.loglog(d["lags"], g, color=GRAY, linewidth=0.4, alpha=0.6)
    ax.loglog(
        d["lags"], d["pooled"], color=BLUE, linewidth=2, label="median of 66 fish"
    )
    ax.loglog(
        d["lags"],
        d["pooled_fit"].model(d["lags"]),
        color=ORANGE,
        linestyle="--",
        linewidth=1.5,
        label="Gaussian kernel fit",
    )
    ax.loglog(
        d["lags"],
        d["pooled_exp"].model(d["lags"]),
        color=AQUA,
        linestyle=":",
        linewidth=1.5,
        label="exponential fit",
    )
    default = DriftFit(0.5, 30.0, 0.0, 0.0)
    ax.loglog(
        d["lags"],
        default.model(d["lags"]),
        color=INK_2,
        linestyle="-.",
        linewidth=1,
        label="current default",
    )
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("semivariogram (Hz²)")
    ax.set_title("individual drift (light phase)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for g in d["shared_gammas"]:
        ax.loglog(d["shared_lags"], g, color=GRAY, linewidth=0.4, alpha=0.6)
    ax.loglog(
        d["shared_lags"],
        d["shared_pooled"],
        color=BLUE,
        linewidth=2,
        label="median of 33 trials",
    )
    ax.loglog(
        d["shared_lags"],
        d["shared_fit"].model(d["shared_lags"]),
        color=ORANGE,
        linestyle="--",
        linewidth=1.5,
        label="Gaussian kernel fit",
    )
    default = DriftFit(1e-3, 300.0, 0.0, 0.0)
    ax.loglog(
        d["shared_lags"],
        default.model(d["shared_lags"]),
        color=INK_2,
        linestyle="-.",
        linewidth=1,
        label="current default",
    )
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("semivariogram (relative²)")
    ax.set_title("shared relative drift", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    for ax in axes:
        _style(ax)
    return _save(fig, path)


def _fig_field_profiles(cal: Calibration, path: Path) -> str:
    import matplotlib.pyplot as plt

    fit = cal.field_fit
    names = list(fit.real_quantiles)
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.3 * len(names), 3.2), sharey=True
    )
    axes = np.atleast_1d(axes)
    q = np.asarray(fit.quantiles)
    lo, mid, hi = (
        int(np.argmin(np.abs(q - 0.05))),
        int(np.argmin(np.abs(q - 0.5))),
        int(np.argmin(np.abs(q - 0.95))),
    )
    for ax, name in zip(axes, names, strict=True):
        real = fit.real_quantiles[name]
        sim = fit.best_quantiles[name]
        rank = np.arange(1, real.shape[1] + 1)
        ax.fill_between(
            rank,
            real[lo],
            real[hi],
            color=BLUE,
            alpha=0.18,
            linewidth=0,
            label="real 5–95%",
        )
        ax.plot(rank, real[mid], color=BLUE, linewidth=2, label="real median")
        ax.plot(
            rank,
            sim[mid],
            color=ORANGE,
            linewidth=2,
            linestyle="--",
            label="fit median",
        )
        ax.plot(
            rank, sim[lo], color=ORANGE, linewidth=1, linestyle=":", label="fit 5/95%"
        )
        ax.plot(rank, sim[hi], color=ORANGE, linewidth=1, linestyle=":")
        ax.set_title(f"layout {name}", fontsize=9)
        ax.set_xlabel("electrode rank")
        _style(ax)
    axes[0].set_ylabel("power re. strongest (dB)")
    axes[0].legend(frameon=False, fontsize=7)
    return _save(fig, path)


def _fig_field_loss(cal: Calibration, path: Path) -> str:
    import matplotlib.pyplot as plt

    fit = cal.field_fit
    loss = fit.loss.min(axis=2)
    fig, axes = plt.subplots(
        1, 2, figsize=(8, 3.3), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    ax = axes[0]
    im = ax.imshow(
        np.log10(loss),
        origin="lower",
        aspect="auto",
        cmap="Blues_r",
        extent=(-0.5, loss.shape[1] - 0.5, -0.5, loss.shape[0] - 0.5),
    )
    ax.set_xticks(
        range(loss.shape[1]), [f"{v:g}" for v in fit.regularisers_rel], fontsize=7
    )
    ax.set_yticks(range(loss.shape[0]), [f"{v:g}" for v in fit.exponents], fontsize=7)
    i = int(np.argmin(np.abs(fit.exponents - fit.best["distance_exponent"])))
    j = int(np.argmin(np.abs(fit.regularisers_rel - fit.best["regulariser_rel"])))
    ax.plot(
        j,
        i,
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgecolor=ORANGE,
        markeredgewidth=2,
    )
    ax.set_xlabel("effective regulariser a_eff / spacing")
    ax.set_ylabel("distance exponent n")
    ax.set_title("log10 loss (dB²), min over margin", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=7)
    ax = axes[1]
    ax.plot(
        fit.exponents,
        fit.loss.min(axis=(1, 2)),
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.set_xlabel("distance exponent n")
    ax.set_ylabel("loss (dB²), min over a_eff and margin")
    ax.set_title("profile loss", fontsize=9)
    _style(ax)
    return _save(fig, path)


def _fig_noise(cal: Calibration, path: Path) -> str:
    import matplotlib.pyplot as plt

    spectra = cal.figures["noise_spectra"]
    extra = cal.figures["noise"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    ax = axes[0]
    for name, (freqs, floor, dead) in spectra.items():
        color = ORANGE if name.startswith("2019") else BLUE
        ax.loglog(
            freqs,
            10 ** np.log10(np.median(floor[:, ~dead], axis=1)),
            color=color,
            linewidth=0.8,
            alpha=0.7,
        )
    ax.plot([], [], color=ORANGE, label="2019 (3x3)")
    ax.plot([], [], color=BLUE, label="2020 (15 el.)")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("noise floor (V²/Hz)")
    ax.set_title("floor per trial (median electrode)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    alphas = extra["alphas"]
    ax.hist(
        alphas[np.isfinite(alphas)],
        bins=40,
        color=BLUE,
        edgecolor="#fcfcfb",
        linewidth=0.5,
    )
    ax.axvline(1.0, color=INK_2, linestyle="-.", linewidth=1)
    ax.annotate(
        "current default",
        (1.0, 0),
        xytext=(4, 60),
        textcoords="offset points",
        fontsize=7,
        color=INK_2,
    )
    ax.set_xlabel("floor exponent α (100–5000 Hz)")
    ax.set_ylabel("electrodes")
    ax = axes[2]
    hum = extra["hum1"]
    years = np.asarray(extra["years"])
    for k, (year, color) in enumerate((("2019", ORANGE), ("2020", BLUE))):
        sel = years == year
        values = np.maximum(hum[sel], 1e-8)
        ax.scatter(
            np.full(sel.sum(), k)
            + np.random.default_rng(k).uniform(-0.15, 0.15, sel.sum()),
            values,
            s=14,
            color=color,
            edgecolor="#fcfcfb",
            linewidth=0.6,
        )
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["2019", "2020"])
    ax.set_ylabel("50-Hz amplitude, worst electrode (V)")
    for ax in axes:
        _style(ax)
    return _save(fig, path)


def _fig_amplitude(cal: Calibration, path: Path) -> str:
    import matplotlib.pyplot as plt

    a = cal.figures["amplitudes"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    axes[0].hist(a["cn0"], bins=60, color=BLUE, edgecolor="#fcfcfb", linewidth=0.3)
    axes[0].set_xlabel("C/N0 at strongest electrode (dB-Hz)")
    axes[0].set_ylabel("frames")
    bins = np.geomspace(1e-5, 1e-2, 50)
    axes[1].hist(a["raw"], bins=bins, color=BLUE, edgecolor="#fcfcfb", linewidth=0.3)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("fundamental amplitude, strongest electrode (V at ADC)")
    axes[1].set_ylabel("segments x fish")
    for ax in axes:
        _style(ax)
    return _save(fig, path)


def _fig_movement(cal: Calibration, path: Path) -> str:
    import matplotlib.pyplot as plt

    rows = cal.figures["movement_rows"]
    sims = cal.summary["movement"]["simulated"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, key, label in (
        (axes[0], "switches_per_min_3db", "strongest-electrode switches / min"),
        (axes[1], "still_fraction_3db", "fraction of time in dwells ≥ 60 s"),
    ):
        dark = np.array([r[f"{key}_dark"] for r in rows])
        light = np.array([r[f"{key}_light"] for r in rows])
        for d_value, l_value in zip(dark, light, strict=True):
            ax.plot([0, 1], [d_value, l_value], color=GRAY, linewidth=0.5)
        ax.scatter(np.zeros(dark.size), dark, s=12, color=BLUE, zorder=3)
        ax.scatter(np.ones(light.size), light, s=12, color=BLUE, zorder=3)
        for k, (name, s) in enumerate(sims.items()):
            ax.scatter(
                [1.6 + 0.12 * k], [s[key]], s=24, marker="D", color=ORANGE, zorder=3
            )
            ax.annotate(
                name,
                (1.6 + 0.12 * k, s[key]),
                fontsize=6,
                color=INK_2,
                rotation=60,
                xytext=(2, 3),
                textcoords="offset points",
            )
        ax.set_xticks(
            [0, 1, 1.6 + 0.06 * len(sims)], ["dark (real)", "light (real)", "simulated"]
        )
        ax.set_ylabel(label)
        _style(ax)
    return _save(fig, path)


def _fig_chirps(cal: Calibration, path: Path) -> str:
    import matplotlib.pyplot as plt

    c = cal.figures["chirps"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    ax = axes[0]
    for k, (role, color) in enumerate((("winner", BLUE), ("loser", ORANGE))):
        values = np.maximum(np.asarray(c["rates"][role]), 0.1)
        ax.scatter(
            np.full(values.size, k)
            + np.random.default_rng(k).uniform(-0.15, 0.15, values.size),
            values,
            s=14,
            color=color,
            edgecolor="#fcfcfb",
            linewidth=0.6,
        )
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["winner", "loser"])
    ax.set_ylabel("v1 chirps per hour (0 shown at 0.1)")
    ax = axes[1]
    ici = c["ici"]
    ax.hist(
        ici[ici > 0],
        bins=np.geomspace(0.05, 3600, 60),
        color=BLUE,
        edgecolor="#fcfcfb",
        linewidth=0.3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("inter-chirp interval within fish (s)")
    ax.set_ylabel("count")
    ax = axes[2]
    if c["latency_obs"] is not None:
        ax.plot(c["latency_centres"], c["latency_ratio"], color=BLUE, linewidth=2)
        ax.axhline(1.0, color=GRAY, linewidth=1, linestyle="--")
    ax.set_xlabel("latency to the other fish's next chirp (s)")
    ax.set_ylabel("observed / shifted null")
    for ax in axes:
        _style(ax)
    return _save(fig, path)


# ----------------------------------------------------------------- text ----


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        v = float(value)
        if not math.isfinite(v):
            return "–"
        if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e5):
            return f"{v:.{digits - 1}e}"
        return f"{v:.{digits}g}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_fmt(v, digits) for v in value) + "]"
    return str(value)


def _qtext(
    q: Mapping[str, Any], digits: int = 3, keys: tuple[str, ...] = ("q05", "q50", "q95")
) -> str:
    lo, mid, hi = (q.get(k) for k in keys)
    return f"{_fmt(mid, digits)} ({_fmt(lo, digits)}–{_fmt(hi, digits)})"


def _param_table(cal: Calibration) -> list[str]:
    lines = [
        "| Simulator parameter | Current default | Proposed | Uncertainty / range |",
        "|---|---|---|---|",
    ]
    for name, p in cal.parameters.items():
        unc = p.get("uncertainty")
        if isinstance(unc, Mapping):
            parts = []
            for k, v in unc.items():
                if isinstance(v, Mapping) and "q50" in v:
                    parts.append(f"{k}: {_qtext(v)}")
                elif isinstance(v, list | tuple):
                    parts.append(f"{k}: {_fmt(list(v))}")
                elif isinstance(v, int | float):
                    parts.append(f"{k}: {_fmt(v)}")
            unc_text = "; ".join(parts) if parts else "see JSON"
        else:
            unc_text = "–"
        short = name.replace("gridtools.sim.", "")
        lines.append(
            f"| `{short}` | {_fmt(p['current_default'])} | **{_fmt(p['value'])}** | {unc_text} |"
        )
    return lines


def write_report(
    calibration: Calibration,
    out_dir: str | Path,
    provenance: Mapping[str, Any],
) -> Path:
    """Write ``report.md`` and ``figures/*.png`` into ``out_dir``.

    Parameters
    ----------
    calibration : Calibration
    out_dir : str or Path
    provenance : mapping
        Shown at the end of the report.

    Returns
    -------
    Path
        The report file.
    """
    import matplotlib

    matplotlib.use("Agg")
    out = Path(out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        "eodf": _fig_eodf(calibration, fig_dir / "eodf.png"),
        "delta_f": _fig_delta_f(calibration, fig_dir / "delta_f.png"),
        "drift": _fig_drift(calibration, fig_dir / "drift.png"),
        "profiles": _fig_field_profiles(calibration, fig_dir / "field_profiles.png"),
        "loss": _fig_field_loss(calibration, fig_dir / "field_loss.png"),
        "noise": _fig_noise(calibration, fig_dir / "noise.png"),
        "amplitude": _fig_amplitude(calibration, fig_dir / "amplitude.png"),
        "movement": _fig_movement(calibration, fig_dir / "movement.png"),
        "chirps": _fig_chirps(calibration, fig_dir / "chirps.png"),
    }
    text = _report_text(calibration, figs, provenance)
    path = out / "report.md"
    path.write_text(text, encoding="utf-8")
    return path


def _report_text(
    cal: Calibration, figs: Mapping[str, str], provenance: Mapping[str, Any]
) -> str:
    s = cal.summary
    new = cal.proposed_new
    e, dfs, d, f, n, a, m, c = (
        s[k]
        for k in (
            "eodf",
            "delta_f",
            "drift",
            "field",
            "noise",
            "amplitudes",
            "movement",
            "chirps",
        )
    )
    layouts = s["layouts"]
    best = f["best"]
    sens = f["sensitivity"]
    slopes = sens.get("effective_slope_at_one_spacing") or {"none": float("nan")}
    regimes = new.get("movement.regimes", {}).get("value", {})
    spacing = float(provenance.get("spacing_m", 0.5))
    per_trial_noise = n["per_trial"]
    sign_ok = all(
        v["sign_v_columns_match_raw_channels"] for v in per_trial_noise.values()
    )
    sign_err = max(v["sign_v_check_max_abs_db"] for v in per_trial_noise.values())
    dead = {k: v["dead"] for k, v in per_trial_noise.items() if v["dead"]}
    dead_groups: dict[str, list[str]] = {}
    for trial, channels in dead.items():
        dead_groups.setdefault(", ".join(str(ch + 1) for ch in channels), []).append(
            trial[:10]
        )
    noisy = {
        k[:10]: [ch + 1 for ch in v["noisy"]]
        for k, v in per_trial_noise.items()
        if v["noisy"]
    }
    lines: list[str] = []
    w = lines.append
    w("# Simulator calibration v1: tube competition recordings")
    w("")
    w(
        f"Generated {datetime.date.today().isoformat()} by `gridtools.calibrate` from {s['n_trials']} trials "
        f"({', '.join(f'{k}: {len(v)}' for k, v in layouts.items())}; layout = electrodes-electrodes with fish signal). "
        "Numbers are medians with 5–95% ranges across fish, trials or samples unless stated. "
        "All proposals with uncertainty and method are in `calibration.json`."
    )
    w("")
    w("## Proposed defaults vs current defaults")
    w("")
    w("Recording units are taken to be raw ADC volts (see *Data findings*).")
    w("")
    lines.extend(_param_table(cal))
    w("")
    w(
        "Proposed parameters without a simulator field yet (details in `calibration.json` → `proposed_new`):"
    )
    w("")
    pop = new["population.baseline_hz"]["value"]
    w(
        f"- `population.baseline_hz`: female N({pop['female']['mean']:.0f}, {pop['female']['sd']:.0f}) Hz, "
        f"male N({pop['male']['mean']:.0f}, {pop['male']['sd']:.0f}) Hz at 23–26 °C."
    )
    w(
        f"- `scene.eod_amplitude`: {_fmt(new['scene.eod_amplitude']['value'])} V (log SD across fish "
        f"{_fmt(new['scene.eod_amplitude']['uncertainty']['log_sd_across_fish'])}), for the fitted field on the 0.5-m geometry."
    )
    w(
        f"- `scene.arena_margin_rel`: {_fmt(new['scene.arena_margin_rel']['value'])} spacings beyond the outer electrodes."
    )
    w(
        f"- `chirps.rate_per_h`: loser {_qtext(c['rate_per_h']['loser'])}, winner {_qtext(c['rate_per_h']['winner'])}; "
        f"`chirps.min_interval_s` {_fmt(new['chirps.min_interval_s']['value'])} s."
    )
    w(
        f"- `noise.click_rate_per_h`: {_qtext(n['clicks_per_h'])} (periodic artefacts, see below)."
    )
    for phase, regime in regimes.items():
        w(
            f'- `movement.regimes.{phase}`: closest variant "{regime["closest_variant"]}" '
            f"(changed `MovementParams`: {regime['params'] or 'none'}); simulated "
            f"{regime['simulated_switches_per_min']:.1f} switches/min, still {regime['simulated_still_fraction']:.2f} "
            f"vs real {regime['real_switches_per_min']:.1f}/min, {regime['real_still_fraction']:.2f}."
        )
    w("")
    w("## Data findings that affect the simulator and the pipeline")
    w("")
    w(
        f"- **`sign_v` definition verified.** `sign_v` = 10·log10 of the one-sided Hann PSD (matplotlib `specgram` scaling, "
        f"mean removed) of each raw channel at the bin nearest the fundamental, frames centred at `times.npy`; nfft = 32768 "
        f"(2019 and May/June 2020) or 16384 (March 2020). Recomputed from raw for random frames: columns match raw channels "
        f"in order in {'all' if sign_ok else 'not all'} trials (largest median deviation {_fmt(sign_err)} dB)."
    )
    w(
        "- **Raw units are volts at the ADC** (quantisation step 0.625 µV = 40.96 mV / 2¹⁶, i.e. ±20.48 mV); electrode "
        "potentials are raw / 100. thunderlab labels the data mV; that is wrong for these files."
    )
    for chans, trials in dead_groups.items():
        w(
            f"- **Channel {chans} carries no fish signal** in {len(trials)} trial(s): {', '.join(trials)}. "
            + (
                "In 2019 `sign_v` has 8 columns because channel 9 was dropped; it is not a 2019-12-09-only quirk."
                if chans == "9"
                else "It is a column of `sign_v` (noise only)."
            )
        )
    if noisy:
        w(
            f"- Noisy electrodes (floor > 6 dB above the trial median): {noisy} (1-based channels)."
        )
    w(
        "- **Partial trailing frame** in `traces-grid1.raw` of 2020-03-19, -03-20 (analysed) and 2020-03-24, -03-25: "
        "`gridtools.io.open_recording` refuses them; `gridtools.calibrate.trials.open_raw` drops the partial frame."
    )
    w(
        "- **Geometry conflict.** `fishgrid.cfg` gives 0.5-m spacing for every grid (a 3×5 grid spans 1 × 2 m), but the paper's "
        "methods describe a 100 cm × 50 cm tank. One of them is wrong. The field fit is done in units of the spacing; "
        "the profile shapes do not depend on the absolute scale, but `min_distance_m` and the EOD amplitude for a given "
        "distance do. Resolve before relying on metre values (e.g. from a setup photo or the 2022 SLEAP videos)."
    )
    w(
        "- **Clicks are mostly recording artefacts**: high-band transients come in pairs 30 ms apart "
        f"({_fmt(100 * (n['click_interval_fractions']['near_0.03s'] or 0))}% of intervals within a segment), in 2019 at "
        "10.24-s intervals (a buffer period), and in several 2020 trials as onset/offset pairs 0.1 s apart about once per "
        "second (consistent with the 100-ms LED sync pulses). Their rate varies by orders of magnitude between trials."
    )
    w("")
    w("## 1. Baseline EODf")
    w("")
    w(f"![EODf]({'figures/' + figs['eodf']})")
    w("")
    w(
        "Method: per fish, the slow baseline is the 5th percentile of the 1-s median frequency in 5-min windows (as in "
        "Raab et al. 2021; robust against rises); its median over the trial is the fish's baseline. Sex and dominance from "
        "`trial_summary.csv`. 25 °C normalisation with Q10 = 1.37 (Raab et al. 2021) and the logged water temperature."
    )
    w("")
    w(
        f"- All fish: {_qtext(e['all'])} Hz (n = {e['all']['n']} fish × trials; fish recur across trials, groups of 4–6)."
    )
    w(
        f"- Females: {_qtext(e['sex_f'])} Hz; males: {_qtext(e['sex_m'])} Hz. At 25 °C: females {_qtext(e['sex_f_25c'])}, "
        f"males {_qtext(e['sex_m_25c'])} Hz."
    )
    w(
        f"- Winners {_qtext(e['role_winner'])} Hz, losers {_qtext(e['role_loser'])} Hz (dominance is confounded with sex "
        "and size here; not a dominance effect)."
    )
    w(
        "- Maps to: the population distribution of baseline EODf (no simulator field yet). The current plan draws "
        "uniformly from 600–1100 Hz (v1); the real range is narrower and bimodal by sex."
    )
    w("")
    w("## 2. Frequency difference ΔF")
    w("")
    w(f"![dF]({'figures/' + figs['delta_f']})")
    w("")
    w(
        "Method: difference of the two fish's despiked 1-s traces (5-s running median; rises included), sampled every 10 s."
    )
    w("")
    w(f"- |ΔF| {_qtext(dfs['abs_delta_f_hz'])} Hz over {dfs['hours']:.0f} h.")
    fb = dfs["fraction_below_pooled"]
    tb = dfs["trials_with_time_below"]
    w(
        f"- Fraction of time below 5 / 10 / 20 / 50 Hz: {100 * fb['5.0']:.2f}% / {100 * fb['10.0']:.1f}% / "
        f"{100 * fb['20.0']:.1f}% / {100 * fb['50.0']:.0f}%; trials that ever go below: {tb['5.0']} / {tb['10.0']} / {tb['20.0']} / {tb['50.0']} of {s['n_trials']}."
    )
    w(
        f"- Frequency crossings: {dfs['n_crossings']} in {dfs['hours']:.0f} h ({dfs['crossings_per_h']:.3f}/h; 1-Hz hysteresis) "
        f"for the despiked traces, {dfs['n_crossings_baseline']} for the slow baselines."
        + (
            " Crossings are essentially absent in this data set (tracker identity swaps at crossings would also "
            "hide them), so crossings in training scenes serve robustness, not realism for these trials."
            if dfs["crossings_per_h"] < 0.05
            else ""
        )
    )
    w(
        "- Maps to: scene construction (pairs with small ΔF are rare; oversample them deliberately for training, as "
        "planned, but report test metrics on the real ΔF distribution)."
    )
    w("")
    w("## 3. Drift")
    w("")
    w(f"![drift]({'figures/' + figs['drift']})")
    w("")
    ind = d["individual_pooled_gaussian"]
    w(
        "Individual drift. Method: despiked 1-s trace minus the fish's median baseline × (1 + shared drift), light phase "
        "only (≥ 3.25 h; fewer rises), semivariogram over lags 1–1800 s with the top 1% of squared increments trimmed "
        "(rises), Gaussian-kernel model with nugget fitted in log space (`gridtools.sim.eod.DriftParams` uses the same "
        "kernel)."
    )
    w("")
    w(
        f"- Pooled fit: SD {ind['sd']:.2f} Hz, τ {ind['timescale_s']:.1f} s, nugget SD {math.sqrt(ind['nugget']):.3f} Hz "
        f"(tracking jitter). Across fish: SD {_qtext(d['individual_sd_hz'])} Hz, τ {_qtext(d['individual_timescale_s'])} s. "
        f"Winners {_qtext(d['individual_sd_by_role']['winner'])} Hz, losers {_qtext(d['individual_sd_by_role']['loser'])} Hz."
    )
    w(
        f"- Fit quality (RMS log residual): Gaussian {_qtext(d['individual_rms_log_gaussian'], 2)}, exponential "
        f"{_qtext(d['individual_rms_log_exponential'], 2)}. Neither kernel is exact: the variogram keeps rising slowly "
        "beyond the plateau (minute-scale wander), which a single time-scale does not capture."
    )
    w("")
    sh = d["shared_pooled_gaussian"]
    w(
        "Shared drift. Method: mean of the two fish's relative slow baselines per trial; semivariogram over 60 s – 4 h."
    )
    w("")
    w(
        f"- The shared component is a decline with water temperature: correlation of the two fish's relative baselines "
        f"{_qtext(d['corr_level'], 2)}, correlation of the shared drift with temperature {_qtext(d['shared_temp_corr'], 2)}; "
        f"slope {_qtext(d['shared_slope_percent_per_h'], 2)} %/h while the water cooled by {_qtext(d['temperature_range_c'], 2)} °C "
        f"({_qtext(d['temperature_slope_c_per_h'], 2)} °C/h). 30-min increments of the two fish correlate only weakly "
        f"({_qtext(d['corr_increments_30min'], 2)}): over tens of minutes individual wander dominates."
    )
    w(
        f"- Apparent Q10 from regressing log EODf on temperature: {_qtext(d['q10_per_fish'], 3)} per fish, higher than "
        "the 1.37 of Raab et al. 2021 because the decline also contains the dark-to-light transition (activity), which "
        "coincides with cooling."
    )
    w(
        f"- Pooled Gaussian fit: SD {sh['sd']:.4f} (relative), τ {sh['timescale_s']:.0f} s; across trials SD "
        f"{_qtext(d['shared_sd_rel'])}, τ {_qtext(d['shared_timescale_s'])} s. Baseline slopes per fish "
        f"{_qtext(d['baseline_slope_hz_per_h'])} Hz/h; |change over 10 min| median {_fmt(d['baseline_change_10min_hz']['q50'])} Hz, "
        f"90th percentile {_fmt(d['baseline_change_10min_hz']['q90'])} Hz."
    )
    w(
        "- Maps to: `DriftParams` (individual) and `SharedDriftParams` (shared); see the table. The current shared default "
        f"(SD 0.1%, τ 300 s) is {sh['sd'] / 1e-3:.0f}× smaller in amplitude and {sh['timescale_s'] / 300.0:.0f}× faster "
        "than measured."
    )
    w("")
    w("## 4. Per-electrode power profiles and the dipole falloff")
    w("")
    w(f"![profiles]({'figures/' + figs['profiles']})")
    w("")
    w(f"![loss]({'figures/' + figs['loss']})")
    w("")
    w(
        "Real profiles: up to 4000 random frames per fish and trial (frames with |ΔF| < 5 Hz excluded), electrodes without "
        "fish signal removed, each frame sorted (descending) and referenced to its maximum. Channel order is never used."
    )
    w("")
    for name in layouts:
        w(
            f"- Layout {name}: strongest minus median electrode {_qtext(f['max_minus_median_db'][name], 3)} dB; drop to the second "
            f"electrode {_qtext(f['drop_to_second_db'][name], 3)} dB; strongest-electrode SNR (PSD bin over floor) "
            f"{_qtext(f['peak_snr_db'][name], 3)} dB."
        )
    w("")
    w(
        "Fit (simulation-based, order-free). Fish positions uniform over the electrode grid plus a margin, headings "
        "uniform, gains from `gridtools.sim.field.dipole_gain`; a frame's power is |√S + η|² with complex Gaussian noise η "
        "(a single periodogram bin), per-electrode noise levels spread by the measured "
        f"{f['noise_offset_sd_db']:.1f} dB, and the signal scaled to the real strongest-electrode SNR distribution of that "
        "layout. 2019: 3×3 grid without the corner electrode (channel 9 is a corner under row-major, column-major and "
        "serpentine order); 2020 trials with a dead channel: one electrode dropped at random. Loss = mean squared "
        "difference of the 5/10/25/50/75/90/95% quantiles of ranks 2..E (dB²), summed over layouts; grid search over "
        "n ∈ 1–4, a_eff/spacing ∈ 0.02–1, margin ∈ 0–1 spacing; 200 bootstrap resamples of trials."
    )
    w("")
    w(
        f"- Best: n = {best['distance_exponent']:g}, a_eff = {best['regulariser_rel']:g} spacing "
        f"(= {best['regulariser_rel'] * spacing:.3f} m at 0.5 m), margin {best['margin_rel']:g} spacing; loss "
        f"{best['loss']:.1f} dB² (RMS quantile error {math.sqrt(best['loss'] / max(1, len(layouts))):.1f} dB per layout). "
        f"Bootstrap 90%: n {f['bootstrap_90']['distance_exponent']}, a_eff/spacing {f['bootstrap_90']['regulariser_rel']}, "
        f"margin {f['bootstrap_90']['margin_rel']}."
    )
    w(
        f"- Physical dipole n = 2: best a_eff = {sens['n = 2 (physical dipole)']['regulariser_rel']:g} spacing, margin "
        f"{sens['n = 2 (physical dipole)']['margin_rel']:g}, loss {sens['n = 2 (physical dipole)']['loss']:.1f} dB²."
    )
    for name in layouts:
        key = f"layout {name} alone"
        w(
            f"- Layout {name} alone ({sens[key]['n_trials']} trials): n = {sens[key]['distance_exponent']:g}, a_eff = "
            f"{sens[key]['regulariser_rel']:g}, margin {sens[key]['margin_rel']:g} (loss {sens[key]['loss']:.1f} dB²)."
        )
    if "2019 missing electrode at random position" in sens:
        r = sens["2019 missing electrode at random position"]
        w(
            f"- 2019 with the missing electrode at a random position instead of a corner: n = {r['distance_exponent']:g}, "
            f"a_eff = {r['regulariser_rel']:g} (corner assumption does not matter)."
        )
    r = sens["equal noise floors on all electrodes"]
    w(
        f"- Equal noise floors on all electrodes: n = {r['distance_exponent']:g}, a_eff = {r['regulariser_rel']:g}."
    )
    w("")
    w(
        "Identifiability. n and a_eff trade off along a ridge (left panel): a steeper falloff with a larger "
        "regulariser produces similar profiles over the 0.5–2 spacing distances that the grid samples. The bootstrap "
        "interval reflects only trial-to-trial sampling, not this ridge, and the remaining misfit "
        "(several dB per quantile) is larger than the differences along the ridge: real profiles have a smaller drop "
        "from the strongest to the second electrode in the 5% tail and deeper weakest electrodes than the best-fitting "
        "point dipole with uniform positions. Plausible causes: fish are extended (15–20 cm) sources, not point "
        "dipoles; positions are not uniform (shelter, walls); tank boundaries. What the data constrain is the effective "
        "falloff over the grid distances: the log-log slope of the on-axis gain at one spacing is "
        f"{_fmt(min(slopes.values()), 2)} to {_fmt(max(slopes.values()), 2)} across the near-optimal set and the best "
        "n = 2 fit (a free-space dipole far from the source: −2)."
    )
    w(
        "- Maps to: `DipoleParams.distance_exponent` and `DipoleParams.min_distance_m` (table). Because a_eff includes "
        "the fish–electrode depth offset, with the proposed values the default 5-cm offset already exceeds a_eff; "
        "set `min_distance_m = 0` or lower the depth offset. For training data, consider drawing (n, a_eff) per scene "
        "along the ridge (n ∈ [1.5, 2], a_eff growing with n) instead of fixing one pair."
    )
    w("")
    w("## 5. Noise floor, hum and fish amplitude")
    w("")
    w(f"![noise]({'figures/' + figs['noise']})")
    w("")
    w(f"![amplitude]({'figures/' + figs['amplitude']})")
    w("")
    w(
        "Method: 20 random 10-s segments per trial (stratified over the recording, seeded), Welch PSD (8192 points, "
        "Hann). Floor = median PSD in 400–1500 Hz per electrode; band-wise floors 20 Hz – 9.5 kHz; hum = line amplitude "
        "(PSD summed over the Hann main lobe above the local floor) at 50 Hz and its harmonics up to 2 kHz."
    )
    w("")
    w(
        f"- Floor: {_qtext(n['floor_db_v2_hz'], 4)} dB re 1 V²/Hz across trials; 2019 {_qtext(n['floor_by_year']['2019'], 4)}, "
        f"2020 {_qtext(n['floor_by_year']['2020'], 4)} dB. Electrode-to-electrode SD within a trial {_qtext(n['electrode_spread_db'], 2)} dB; "
        f"variation between segments {_qtext(n['segment_variation_db'], 2)} dB."
    )
    w(
        f"- Spectral shape: exponent α {_qtext(n['alpha'], 2)} (100–5000 Hz): white. The amplifier's 100-Hz high-pass "
        "is not visible as a drop in the floor."
    )
    w(
        f"- Dead electrodes {n['dead_electrodes']} of {n['electrodes']}; noisy {n['noisy_electrodes']} (noise amplitude "
        f"factor {_qtext(n['noisy_factor'], 2)})."
    )
    w(
        f"- Hum (50 Hz, electrode with most hum): 2019 {_qtext(n['hum_50hz_by_year_v']['2019'])} V, 2020 "
        f"{_qtext(n['hum_50hz_by_year_v']['2020'])} V. Coupling across electrodes: SD of ln amplitude "
        f"{_qtext(n['hum_coupling_log_sd'], 2)} (factor ~10 between electrodes). Harmonics detected: {_qtext(n['hum_harmonics_detected'], 2)} "
        "of 40; ratios to the fundamental (median across trials): "
        + ", ".join(
            f"{h}: {_fmt(v['q50'], 2)}" for h, v in n["hum_harmonic_ratio"].items()
        )
        + ". 2019 hum is odd-harmonic dominated (150, 250 Hz strong), 2020 hum is weak with 100 Hz often above 50 Hz."
    )
    w(
        f"- Largest |sample| per trial {_qtext(n['max_abs_v'])} V against the ±20.48 mV range; no clipped samples."
    )
    w(
        f"- Fish fundamental at its strongest electrode: {_qtext(a['peak_fundamental_v_raw'])} V (raw segments; per fish "
        f"medians {_qtext(a['per_fish_median_v_raw'])} V). Carrier-to-noise density there: {_qtext(a['cn0_db_hz'], 3)} dB-Hz, "
        f"i.e. an SNR of {_qtext(a['snr_db_in_v1_spectrogram_4096'], 3)} dB in a 4096-point Hann spectrogram bin (v1 front end)."
    )
    w(
        "- Maps to: `ColoredNoiseParams` (white, SD from the floor), `HumParams`, `ElectrodeFaultParams`, `ClipParams` "
        "(table) and the EOD amplitude: with recording units = volts, an EOD amplitude of "
        f"{_fmt(new['scene.eod_amplitude']['value'])} V reproduces the median fundamental at the strongest electrode for the "
        f"fitted field (median strongest-electrode gain {_fmt(a['simulated_max_gain_quantiles']['q50'])}). "
        "Gain mismatch is not identifiable without positions; the floor spread bounds it."
    )
    w("")
    w("## 6. Movement proxies")
    w("")
    w(f"![movement]({'figures/' + figs['movement']})")
    w("")
    w(
        "Method: strongest electrode per track frame (electrodes without signal excluded) with 3-dB hysteresis; switches "
        "per minute, completed dwell times, fraction of time in dwells ≥ 60 s; dark phase (first 3 h) vs light phase. "
        "Simulated fish: `gridtools.sim.movement.simulate_path` in an arena of the grid plus the fitted margin, the "
        "fitted field and noise, sampled at the track frame interval, same statistics (four 1-h paths per variant)."
    )
    w("")
    for name in layouts:
        mm = m[name]
        w(
            f"- {name}: switches/min dark {_qtext(mm['switches_per_min_3db_dark'], 3)}, light {_qtext(mm['switches_per_min_3db_light'], 2)}; "
            f"still fraction dark {_qtext(mm['still_fraction_3db_dark'], 2)}, light {_qtext(mm['still_fraction_3db_light'], 2)}."
        )
    w(f"- Dwell times (all): {_qtext(m['dwell_times_3db_s'], 3)} s.")
    w(
        f"- Simulated ({m['simulated_layout']}): "
        + "; ".join(
            f"{k}: {v['switches_per_min_3db']:.1f}/min, still {v['still_fraction_3db']:.2f}"
            for k, v in m["simulated"].items()
        )
    )
    for phase, regime in regimes.items():
        w(
            f"- Closest variant for the {phase} phase: {regime['closest_variant']} "
            f"({regime['simulated_switches_per_min']:.1f}/min, still {regime['simulated_still_fraction']:.2f}; real "
            f"{regime['real_switches_per_min']:.1f}/min, {regime['real_still_fraction']:.2f})."
        )
    w(
        "- Interpretation: fish switch between an active regime (dark phase, many switches, no long dwells) and a "
        "resting regime (light phase, dwells of minutes); a scene could mix both. Resting fish in the simulator "
        "rotate in place (`rest_turn_factor`), which alone produces switches. What cannot be inferred: speeds, "
        "positions, headings, turning, wall use; the switch rate also depends on the field model and on the "
        "(uncertain) electrode spacing."
    )
    w("")
    w("## 7. Chirp timing (v1 detections)")
    w("")
    w(f"![chirps]({'figures/' + figs['chirps']})")
    w("")
    w(
        f"Source: `chirpdetector_bboxes.csv` of each trial (v1 YOLO detections, v1 emitter assignment). {c['bias_note']}. "
        f"{c['n_detections_assigned']} detections assigned to the two fish; {c['n_duplicates_merged']} same-fish detections "
        "closer than 50 ms merged (v1 duplicates)."
    )
    w("")
    w(
        f"- Rates per fish: losers {_qtext(c['rate_per_h']['loser'])}/h (dark {_qtext(c['rate_per_h_dark']['loser'])}, light "
        f"{_qtext(c['rate_per_h_light']['loser'])}); winners {_qtext(c['rate_per_h']['winner'])}/h (dark "
        f"{_qtext(c['rate_per_h_dark']['winner'])}, light {_qtext(c['rate_per_h_light']['winner'])})."
    )
    ici = c["ici_s"]
    w(
        f"- Inter-chirp intervals within a fish: 1st percentile {_fmt(ici['q01'])} s, median {_fmt(ici['q50'])} s; "
        f"fraction below 0.5 / 1 / 2 s: {100 * c['ici_fraction_below']['0.5']:.0f}% / {100 * c['ici_fraction_below']['1.0']:.0f}% / "
        f"{100 * c['ici_fraction_below']['2.0']:.0f}% (bursts). Shortest interval per fish {_qtext(c['min_ici_per_fish_s'])} s."
    )
    co = c["coincidence_0.25s"]["pooled"]
    w(
        f"- Chirps with a chirp of the other fish within ±0.25 s: {100 * (co['observed'] or 0):.2f}% observed vs "
        f"{100 * (co['null'] or 0):.2f}% under a null that shifts the other fish's train by 5–30 s."
        + (
            " Near-simultaneous chirps are under-represented in v1 detections (two overlapping chirps are probably "
            "merged or lost by the v1 detector or assignment), so the simulator should place them deliberately rather "
            "than copy this rate."
            if (co["observed"] or 0) < (co["null"] or 0)
            else ""
        )
    )
    lr = c.get("latency_ratio", {})
    w(
        "- Next-chirp latency to the other fish, observed / null: "
        + ", ".join(f"{k}: {_fmt(v, 2)}" for k, v in lr.items())
        + f". Over 0–1 s: {_fmt(c.get('latency_excess_0_1s'), 2)}; 1–3 s: {_fmt(c.get('latency_excess_1_3s'), 2)}."
    )
    w("")
    w("## What could not be measured")
    w("")
    w(
        "- Absolute fish positions, speeds and headings; electrode positions (channel order and, given the tank-size "
        "conflict, the spacing)."
    )
    w(
        "- Gain mismatch between electrodes (confounded with position); the depth offset separately from the regulariser."
    )
    w("- Chirps missed by v1, true emitters, the true rate of overlapping chirps.")
    w("- Mains frequency jitter (2.4-Hz resolution of the hum spectra).")
    w("- Anything about the 17 recordings outside the 33 (not measured).")
    w("")
    w("## Provenance")
    w("")
    for k, v in provenance.items():
        w(f"- {k}: {v}")
    w("")
    return "\n".join(lines)
