"""Command line: build a background library and write its report.

.. code-block:: sh

    python -m gridtools.background build LIB/v1 \\
        --datasets competition2019 competition2022 \\
        --root-2019 DATA/2019_tube_competition --root-2022 DATA/2022_tube_competition \\
        --workers 6 --seed 0
    python -m gridtools.background review LIB/v1
    python -m gridtools.background report LIB/v1

Every dataset of ``--datasets`` needs its root (``--root-2019`` etc.).
``review``, ``report`` and ``diagnose`` exit with status 1 and a message if
the directory is not a library or has no segments.

A build can take its options from a TOML file (``BackgroundConfig.from_toml``:
per-dataset time windows, external chirp screens, the recording gate,
per-dataset selection); its keys replace the command-line defaults:

.. code-block:: sh

    python -m gridtools.background carry-over LIB/v1 LIB/v2 \
        --datasets field_flona2025 field_iriri2026
    python -m gridtools.background build LIB/v2 --config configs/background/v2.toml \
        --datasets competition2019 competition2022 --root-2019 ... --root-2022 ...
    python -m gridtools.background diagnose LIB/v2 \
        --screen v1 --snippets 2020-03-18-10_34 2020-03-10-09_56
    python -m gridtools.background report LIB/v2 \
        --splits train=configs/scenes/train_v3_train.toml ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gridtools.background.build import BackgroundConfig, build_background_library
from gridtools.background.field import (
    FieldFolder,
    FieldScanConfig,
    flona2023_folders,
    flona2025_folders,
    iriri2026_folders,
)
from gridtools.background.library import read_manifest
from gridtools.background.select import SelectionConfig
from gridtools.background.sources import (
    SourceRecording,
    competition_2019_sources,
    competition_2022_sources,
)

DATASETS = (
    "competition2019",
    "competition2022",
    "field_iriri2026",
    "field_flona2025",
    "field_flona2023",
)

#: The root option of every dataset (no defaults: the data are the author's).
ROOT_OPTIONS = {
    "competition2019": "--root-2019",
    "competition2022": "--root-2022",
    "field_iriri2026": "--root-iriri2026",
    "field_flona2025": "--root-flona2025",
    "field_flona2023": "--root-flona2023",
}


def _fail(message: str) -> None:
    """Print ``error: message`` and exit with status 1."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _check_library(root: Path, command: str) -> None:
    """Exit with a message unless ``root`` is a library with segments."""
    if not (root / "manifest.parquet").exists():
        _fail(
            f"{root} is not a background library (no manifest.parquet); "
            "build one with `python -m gridtools.background build`"
        )
    if read_manifest(root).empty:
        _fail(f"background library {root} has no segments: nothing to {command}")


def _sources(args: argparse.Namespace) -> list[SourceRecording | FieldFolder]:
    sources: list[SourceRecording | FieldFolder] = []
    if "competition2019" in args.datasets:
        sources += competition_2019_sources(args.root_2019, args.recordings_2019)
    if "competition2022" in args.datasets:
        sources += competition_2022_sources(args.root_2022, args.recordings_2022)
    if "field_iriri2026" in args.datasets:
        sources += iriri2026_folders(args.root_iriri2026)
    if "field_flona2025" in args.datasets:
        sources += flona2025_folders(args.root_flona2025)
    if "field_flona2023" in args.datasets:
        sources += flona2023_folders(args.root_flona2023)
    return sources


def main(argv: list[str] | None = None) -> None:
    """Entry point of ``python -m gridtools.background``."""
    parser = argparse.ArgumentParser(prog="python -m gridtools.background")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="select, measure and write segments")
    build.add_argument("out_dir", type=Path)
    build.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
    )
    for dataset, option in ROOT_OPTIONS.items():
        build.add_argument(
            option, type=Path, default=None, help=f"root of {dataset} (if built)"
        )
    build.add_argument("--field-probes", type=int, default=8)
    build.add_argument("--field-segments", type=int, default=2)
    build.add_argument("--recordings-2019", nargs="+", default=None)
    build.add_argument("--recordings-2022", nargs="+", default=None)
    build.add_argument("--segment-s", type=float, default=60.0)
    build.add_argument("--segments-per-recording", type=int, default=8)
    build.add_argument("--fish-free-per-recording", type=int, default=2)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--no-report", action="store_true")
    build.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML file with BackgroundConfig keys (replace the defaults above)",
    )
    carry = sub.add_parser(
        "carry-over", help="take whole datasets from another library, unchanged"
    )
    carry.add_argument("source_dir", type=Path)
    carry.add_argument("out_dir", type=Path)
    carry.add_argument("--datasets", nargs="+", required=True)
    diagnose = sub.add_parser(
        "diagnose", help="screen rates per recording, detection snippets"
    )
    diagnose.add_argument("out_dir", type=Path)
    diagnose.add_argument("--screen", default="v1", help="screen of library.json")
    diagnose.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.5])
    diagnose.add_argument(
        "--snippets", nargs="*", default=[], help="recordings to draw snippets of"
    )
    diagnose.add_argument("--n-snippets", type=int, default=16)
    diagnose.add_argument("--seed", type=int, default=0)
    review = sub.add_parser("review", help="plot every segment, merge review.csv")
    review.add_argument("out_dir", type=Path)
    review.add_argument("--workers", type=int, default=6)
    review.add_argument("--overwrite", action="store_true")
    report = sub.add_parser("report", help="write report.md and figures")
    report.add_argument("out_dir", type=Path)
    report.add_argument("--seed", type=int, default=0)
    report.add_argument(
        "--splits",
        nargs="*",
        default=[],
        metavar="NAME=SCENE_TOML",
        help="scene configs whose [background] include_groups define splits",
    )
    args = parser.parse_args(argv)
    if args.command == "build":
        lacking = [
            ROOT_OPTIONS[d]
            for d in args.datasets
            if getattr(args, ROOT_OPTIONS[d].removeprefix("--").replace("-", "_"))
            is None
        ]
        if lacking:
            build.error(f"give the root of every dataset built: {' '.join(lacking)}")
    elif args.command in ("review", "report", "diagnose"):
        _check_library(args.out_dir, args.command)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    if args.command == "build":
        config = BackgroundConfig(
            selection=SelectionConfig(
                segment_s=args.segment_s,
                segments_per_recording=args.segments_per_recording,
            ),
            fish_free_per_recording=args.fish_free_per_recording,
            field=FieldScanConfig(
                probes_per_folder=args.field_probes,
                segments_per_folder=args.field_segments,
            ),
            seed=args.seed,
            workers=min(args.workers, 6),
        )
        if args.config is not None:
            config = _from_file(config, args.config)
        manifest = build_background_library(_sources(args), args.out_dir, config)
        if manifest.empty:
            _fail(f"the build selected no segments: {args.out_dir} is empty")
        if not args.no_report:
            from gridtools.background.report import write_report
            from gridtools.background.review import write_review

            write_review(args.out_dir, workers=config.workers)
            write_report(args.out_dir, seed=config.seed)
    elif args.command == "carry-over":
        from gridtools.background.build import carry_over

        carry_over(args.source_dir, args.out_dir, args.datasets)
    elif args.command == "diagnose":
        from gridtools.background.diagnostics import write_diagnostics

        write_diagnostics(
            args.out_dir,
            screen=args.screen,
            thresholds=tuple(args.thresholds),
            snippet_recordings=args.snippets,
            n_snippets=args.n_snippets,
            seed=args.seed,
        )
    elif args.command == "review":
        from gridtools.background.review import write_review

        write_review(args.out_dir, workers=args.workers, overwrite=args.overwrite)
    else:
        from gridtools.background.report import write_report

        write_report(args.out_dir, seed=args.seed, splits=_splits(args.splits))


def _from_file(defaults: BackgroundConfig, path: Path) -> BackgroundConfig:
    """``defaults`` with the keys of a TOML file replaced."""
    import tomllib

    data = defaults.to_dict()
    for key, value in tomllib.loads(path.read_text()).items():
        if key in ("selection", "field") and isinstance(value, dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return BackgroundConfig.from_dict(data)


def _splits(specs: list[str]) -> dict[str, list[str]]:
    """``NAME=SCENE_TOML`` -> the scene config's background include_groups."""
    import tomllib

    out: dict[str, list[str]] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep:
            msg = f"--splits expects NAME=SCENE_TOML, got {spec!r}"
            raise SystemExit(msg)
        config = tomllib.loads(Path(path).read_text())
        groups = config.get("background", {}).get("include_groups")
        if not groups:
            msg = f"{path}: no [background] include_groups"
            raise SystemExit(msg)
        out[name] = [str(g) for g in groups]
    return out


if __name__ == "__main__":
    main()
