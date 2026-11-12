"""Shared pytest configuration.

- ``fixture_dir``: the 20 s real-recording excerpt in ``tests/data``.
- Tests marked ``data`` need the tube competition recordings and are skipped
  when ``DATA_ROOT`` is absent.
- Tests marked ``slow`` are opt-in: run them with ``--run-slow`` or select
  them with ``-m slow``.
"""

from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "data" / "2020-03-16-10_00_subset"
DATA_ROOT = Path("/mnt/data2/2019_tube_competition")


@pytest.fixture
def fixture_dir() -> Path:
    """20 s excerpt of trial 2020-03-16: 15-channel wav plus tracks of two fish."""
    return FIXTURE


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked slow (also run when selected with -m slow)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    markexpr = str(config.getoption("markexpr", default="") or "")
    run_slow = config.getoption("--run-slow") or (
        "slow" in markexpr and "not slow" not in markexpr
    )
    skip_slow = pytest.mark.skip(reason="slow; run with --run-slow or -m slow")
    skip_data = pytest.mark.skip(reason=f"{DATA_ROOT} not available")
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "data" in item.keywords and not DATA_ROOT.exists():
            item.add_marker(skip_data)
