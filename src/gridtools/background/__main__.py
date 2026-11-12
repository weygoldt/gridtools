"""Command line: build a background library and write its report.

.. code-block:: sh

    python -m gridtools.background build /mnt/data1/chirp_v2/background/v1 \\
        --datasets competition2019 competition2022 field_iriri2026 \\
        --workers 6 --seed 0
    python -m gridtools.background review /mnt/data1/chirp_v2/background/v1
    python -m gridtools.background report /mnt/data1/chirp_v2/background/v1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gridtools.background.build import BackgroundConfig, build_background_library
from gridtools.background.field import (
    FLONA2023_ROOT,
    FLONA2025_ROOT,
    IRIRI2026_ROOT,
    FieldFolder,
    FieldScanConfig,
    flona2023_folders,
    flona2025_folders,
    iriri2026_folders,
)
from gridtools.background.select import SelectionConfig
from gridtools.background.sources import (
    COMPETITION_2019_ROOT,
    COMPETITION_2022_ROOT,
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
    build.add_argument("--root-2019", type=Path, default=COMPETITION_2019_ROOT)
    build.add_argument("--root-2022", type=Path, default=COMPETITION_2022_ROOT)
    build.add_argument("--root-iriri2026", type=Path, default=IRIRI2026_ROOT)
    build.add_argument("--root-flona2025", type=Path, default=FLONA2025_ROOT)
    build.add_argument("--root-flona2023", type=Path, default=FLONA2023_ROOT)
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
    review = sub.add_parser("review", help="plot every segment, merge review.csv")
    review.add_argument("out_dir", type=Path)
    review.add_argument("--workers", type=int, default=6)
    review.add_argument("--overwrite", action="store_true")
    report = sub.add_parser("report", help="write report.md and figures")
    report.add_argument("out_dir", type=Path)
    report.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
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
        build_background_library(_sources(args), args.out_dir, config)
        if not args.no_report:
            from gridtools.background.report import write_report
            from gridtools.background.review import write_review

            write_review(args.out_dir, workers=config.workers)
            write_report(args.out_dir, seed=args.seed)
    elif args.command == "review":
        from gridtools.background.review import write_review

        write_review(args.out_dir, workers=args.workers, overwrite=args.overwrite)
    else:
        from gridtools.background.report import write_report

        write_report(args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
