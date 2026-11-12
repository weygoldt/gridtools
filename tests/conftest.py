"""Shared pytest configuration: slow tests are opt-in."""

import pytest


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
    selected = "slow" in markexpr and "not slow" not in markexpr
    if config.getoption("--run-slow") or selected:
        return
    skip = pytest.mark.skip(reason="slow; run with --run-slow or -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
