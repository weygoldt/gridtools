"""``gridtools sim``: simulate scenes and plot them."""

import json
import logging
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated, Any

import typer

from gridtools.sim.config import SceneConfig, default_config_path

__all__ = ["app", "parse_seeds", "resolve_config"]

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Simulate grid recordings of chirping fish with exact ground truth.",
    no_args_is_help=True,
)


def parse_seeds(text: str) -> list[int]:
    """Parse ``"0:10"`` (range, stop excluded), ``"0:10:2"``, ``"3"`` or ``"1,4,7"``."""
    text = text.strip()
    try:
        if ":" in text:
            parts = [int(p) for p in text.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError
            seeds = list(range(*parts))
        else:
            seeds = [int(p) for p in text.split(",") if p.strip()]
    except ValueError:
        msg = f"cannot parse seeds {text!r}; use e.g. 0:10, 0:10:2, 3 or 1,4,7"
        raise typer.BadParameter(msg) from None
    if not seeds or any(s < 0 for s in seeds):
        msg = f"seeds must be a non-empty set of non-negative integers, got {text!r}"
        raise typer.BadParameter(msg)
    return seeds


def resolve_config(config: str) -> SceneConfig:
    """Load a TOML file, or a shipped config by name (``default``)."""
    path = Path(config)
    if path.suffix != ".toml" and not path.exists():
        shipped = default_config_path().parent / f"{config}.toml"
        if shipped.exists():
            path = shipped
    if not path.exists():
        msg = f"no config file {config!r} (and no shipped config of that name)"
        raise typer.BadParameter(msg)
    return SceneConfig.from_toml(path)


def _run_one(config_json: str, seed: int, out_dir: str, plots: bool) -> dict[str, Any]:
    from gridtools.sim.scene import simulate_scene, write_scene

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    config = SceneConfig.from_json(config_json)
    scene = simulate_scene(config, seed)
    path = write_scene(scene, Path(out_dir) / f"{seed:06d}", plots=plots)
    # peak resident memory of this process so far (KiB on Linux)
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "seed": seed,
        "path": str(path),
        "n_fish": len(scene.model.fish),
        "n_chirps": len(scene.chirps),
        "n_electrodes": scene.model.n_electrodes,
        "background": "synthetic"
        if scene.plan.background is None
        else f"{scene.plan.background.segment.segment_id} "
        f"({len(scene.real_fish)} real fish, SNR {scene.plan.background.snr_db:.0f} dB)",
        "peak_rss_mb": peak_rss_mb,
        **scene.timings_s,
    }


@app.command()
def run(
    config: Annotated[
        str, typer.Argument(help="Scene config TOML, or a shipped config name.")
    ],
    seeds: Annotated[
        str, typer.Option(help="Seeds: 0:10 (stop excluded), 0:10:2, 3 or 1,4,7.")
    ] = "0:1",
    out: Annotated[
        Path | None,
        typer.Option(help="Output root; default /mnt/data1/chirp_v2/scenes/<name>."),
    ] = None,
    workers: Annotated[int, typer.Option(min=1, help="Parallel processes.")] = 1,
    no_plots: Annotated[
        bool, typer.Option("--no-plots", help="Skip the annotated spectrograms.")
    ] = False,
) -> None:
    """Simulate scenes, one directory per seed (``OUT/<seed:06d>``)."""
    from gridtools.sim.scene import DEFAULT_ROOT

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    scene_config = resolve_config(config)
    seed_list = parse_seeds(seeds)
    root = out if out is not None else DEFAULT_ROOT / scene_config.name
    root.mkdir(parents=True, exist_ok=True)
    config_json = scene_config.to_json()
    (root / "config.json").write_text(json.dumps(scene_config.to_dict(), indent=1))
    plots = scene_config.plot.enabled and not no_plots
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    if workers == 1:
        for seed in seed_list:
            results.append(_run_one(config_json, seed, str(root), plots))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_one, config_json, seed, str(root), plots)
                for seed in seed_list
            ]
            results = [f.result() for f in futures]
    for r in results:
        logger.info(
            "seed %06d: %d fish, %d electrodes, %d chirps, %s background; "
            "simulate %.1f s, write %.1f s, plot %.1f s; peak RSS %.0f MB",
            r["seed"],
            r["n_fish"],
            r["n_electrodes"],
            r["n_chirps"],
            r["background"],
            r.get("simulate_total", float("nan")),
            r.get("write", float("nan")),
            r.get("plot", 0.0),
            r["peak_rss_mb"],
        )
    logger.info(
        "%d scenes in %s (%.1f s wall)", len(results), root, time.perf_counter() - t0
    )


@app.command()
def compare(
    scenes: Annotated[
        Path, typer.Argument(help="Directory of scene directories (one per seed).")
    ],
    calibration: Annotated[
        Path, typer.Option(help="calibration.json of gridtools.calibrate.")
    ] = Path("/mnt/data1/chirp_v2/calibration/v1/calibration.json"),
    out: Annotated[
        Path | None,
        typer.Option(help="Output stem; default SCENES/comparison(.md, .csv)."),
    ] = None,
) -> None:
    """Measure scenes with the calibration's definitions and compare to real data."""
    from gridtools.calibrate.scenes import compare_scenes

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    dirs = sorted(p for p in scenes.iterdir() if (p / "scene.json").exists())
    if not dirs:
        msg = f"no scene directories in {scenes}"
        raise typer.BadParameter(msg)
    table = compare_scenes(list(dirs), calibration)
    stem = out if out is not None else scenes / "comparison"
    table.to_csv(stem.with_suffix(".csv"), index=False)
    lines = [
        f"Simulated ({len(dirs)} scenes in {scenes}) vs real ({calibration}); "
        "median (5-95 %).",
        "",
        "| quantity | simulated | real |",
        "|---|---|---|",
        *(f"| {q} | {s} | {r} |" for q, s, r in table.itertuples(index=False)),
    ]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    typer.echo("\n".join(lines))


@app.command()
def plot(
    scene_dir: Annotated[Path, typer.Argument(help="Scene directory.")],
    out: Annotated[
        Path | None, typer.Option(help="Directory for the PNGs (default: scene dir).")
    ] = None,
) -> None:
    """Write the annotated spectrograms of a scene directory."""
    from gridtools.sim.plot import plot_scene_dir

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    for path in plot_scene_dir(scene_dir, out):
        typer.echo(str(path))
