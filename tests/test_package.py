import gridtools


def test_version_is_v2() -> None:
    assert gridtools.__version__.startswith("2.")
