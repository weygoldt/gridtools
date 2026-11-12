from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "data" / "2020-03-16-10_00_subset"
DATA_ROOT = Path("/mnt/data2/2019_tube_competition")


@pytest.fixture
def fixture_dir() -> Path:
    """20 s excerpt of trial 2020-03-16: 15-channel wav plus tracks of two fish."""
    return FIXTURE


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if DATA_ROOT.exists():
        return
    skip = pytest.mark.skip(reason=f"{DATA_ROOT} not available")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
