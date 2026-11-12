"""Every random draw comes from the Generator passed in, nothing else."""

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import gridtools.sim
from gridtools.sim.eod import (
    baseline_frequency,
    jitter_harmonics,
    shared_frequency_drift,
)
from gridtools.sim.movement import Arena, simulate_path
from gridtools.sim.noise import (
    ElectrodeFaultParams,
    colored_noise,
    electrode_faults,
    gain_mismatch,
    powerline_hum,
)

ARENA = Arena(0.0, 1.0, 0.0, 1.0)


def _arrays(result: object) -> list[np.ndarray]:
    """Collect the arrays (or tuples) that make up a function's output."""
    if isinstance(result, np.ndarray):
        return [result]
    fields = (
        "amplitudes",
        "phases_rad",
        "xy_m",
        "heading_rad",
        "velocity_m_s",
        "moving",
        "source",
        "coupling",
        "dead",
        "noisy",
    )
    return [np.asarray(getattr(result, f)) for f in fields if hasattr(result, f)]


GENERATORS: dict[str, Callable[[np.random.Generator], object]] = {
    "jitter_harmonics": lambda rng: jitter_harmonics(rng),
    "baseline_frequency": lambda rng: baseline_frequency(rng, 20_000, 2_000.0, 700.0),
    "shared_frequency_drift": lambda rng: shared_frequency_drift(rng, 20_000, 2_000.0),
    "simulate_path": lambda rng: simulate_path(rng, 60.0, ARENA),
    "colored_noise": lambda rng: colored_noise(rng, 4096, 3, 20_000.0),
    "powerline_hum": lambda rng: powerline_hum(rng, 4096, 3, 20_000.0),
    "gain_mismatch": lambda rng: gain_mismatch(rng, 16),
    "electrode_faults": lambda rng: electrode_faults(
        rng, 64, ElectrodeFaultParams(dead_prob=0.3, noisy_prob=0.3)
    ),
}


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_same_seed_same_output_different_seed_different(name: str) -> None:
    make = GENERATORS[name]
    first = _arrays(make(np.random.default_rng(42)))
    again = _arrays(make(np.random.default_rng(42)))
    other = _arrays(make(np.random.default_rng(43)))
    assert first
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, c) for a, c in zip(first, other, strict=True))


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_global_numpy_rng_is_neither_used_nor_touched(name: str) -> None:
    make = GENERATORS[name]
    np.random.seed(1)  # noqa: NPY002 - deliberately seeding the legacy global RNG
    state_before = np.random.get_state(legacy=False)  # noqa: NPY002
    first = _arrays(make(np.random.default_rng(7)))
    state_after = np.random.get_state(legacy=False)  # noqa: NPY002
    assert repr(state_before) == repr(state_after)
    np.random.seed(999)  # noqa: NPY002
    second = _arrays(make(np.random.default_rng(7)))
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_sources_never_use_global_or_internal_seeding() -> None:
    package_dir = Path(gridtools.sim.__file__).parent
    forbidden = re.compile(
        r"np\.random\.(?!Generator\b)|numpy\.random\.(?!Generator\b)"
        r"|default_rng|SeedSequence|^\s*import random|^\s*from random "
    )
    # seeding.py is the one module that turns a scene seed into generators
    offenders = [
        f"{path.name}:{lineno}: {line.strip()}"
        for path in sorted(package_dir.glob("*.py"))
        if path.name != "seeding.py"
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if forbidden.search(line)
    ]
    assert offenders == []


def test_public_api_is_exported_without_shadowing_submodules() -> None:
    import gridtools.sim.mix as mix_module

    for name in gridtools.sim.__all__:
        assert hasattr(gridtools.sim, name), name
    assert callable(mix_module.mix)
    assert mix_module.__name__ == "gridtools.sim.mix"
