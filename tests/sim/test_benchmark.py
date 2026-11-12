"""End-to-end cost of the building blocks at production scale (opt-in).

Run with ``uv run pytest -m slow -o log_cli=true --log-cli-level=INFO``.
Set ``GRIDTOOLS_BENCH_WORKERS`` to change the thread count (default 1).
"""

import logging
import os
import resource
import time
import tracemalloc

import numpy as np
import pytest

from gridtools.sim.eod import (
    baseline_frequency,
    jitter_harmonics,
    shared_frequency_drift,
    synthesize_eod,
)
from gridtools.sim.field import gain_track
from gridtools.sim.mix import mix
from gridtools.sim.movement import Arena, simulate_path
from gridtools.sim.noise import (
    ClipParams,
    colored_noise,
    electrode_faults,
    gain_mismatch,
    powerline_hum,
)

logger = logging.getLogger(__name__)

DURATION_S = 600.0
FS_HZ = 20_000.0
N_FISH = 5
WORKERS = int(os.environ.get("GRIDTOOLS_BENCH_WORKERS", "1"))


@pytest.mark.slow
def test_benchmark_600s_15_electrodes_5_fish() -> None:
    electrodes = np.array(
        [[x, y] for y in np.arange(5) * 0.5 for x in np.arange(3) * 0.5]
    )
    arena = Arena(-0.25, 1.25, -0.25, 2.25)
    n = int(DURATION_S * FS_HZ)
    n_electrodes = electrodes.shape[0]
    seeds = np.random.SeedSequence(20261112).spawn(5 + N_FISH)
    rngs = [np.random.Generator(np.random.PCG64(s)) for s in seeds]

    tracemalloc.start()
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        timings[name] = timings.get(name, 0.0) + now - since
        return now

    t = time.perf_counter()
    shared = shared_frequency_drift(rngs[0], n, FS_HZ)
    faults = electrode_faults(rngs[1], n_electrodes)
    channel = gain_mismatch(rngs[2], n_electrodes) * faults.signal_gain
    t = lap("drift+faults", t)
    noise = colored_noise(
        rngs[3], n, n_electrodes, FS_HZ, sd_scale=faults.noise_scale, workers=WORKERS
    )
    t = lap("colored_noise", t)
    hum = powerline_hum(rngs[4], n, n_electrodes, FS_HZ)
    t = lap("hum", t)

    eods, tracks = [], []
    baselines_hz = np.linspace(600.0, 900.0, N_FISH)
    for i in range(N_FISH):
        rng = rngs[5 + i]
        path = simulate_path(rng, DURATION_S, arena)
        tracks.append(gain_track(path.times_s, path.xy_m, path.heading_rad, electrodes))
        t = lap("movement+field", t)
        freq = baseline_frequency(
            rng, n, FS_HZ, float(baselines_hz[i]), shared_drift_rel=shared
        )
        t = lap("baseline_frequency", t)
        harmonics = jitter_harmonics(rng)
        eods.append(synthesize_eod(freq, 1.0, FS_HZ, harmonics, workers=WORKERS))
        del freq
        t = lap("synthesize_eod", t)
    del shared

    recording = mix(
        eods,
        tracks,
        samplerate_hz=FS_HZ,
        additive=[noise, hum],
        channel_gain=channel,
        clip=ClipParams(),
        workers=WORKERS,
        out=noise,
    )
    lap("mix", t)
    elapsed_s = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    max_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    logger.info(
        "600 s x 20 kHz x 15 electrodes x 5 fish, %d workers: %.1f s wall, "
        "peak traced %.2f GB, max RSS %.2f GB, load average %.1f",
        WORKERS,
        elapsed_s,
        peak_bytes / 1024**3,
        max_rss_gb,
        os.getloadavg()[0],
    )
    for name, seconds in timings.items():
        logger.info("  %-20s %6.2f s", name, seconds)

    assert recording.shape == (n, n_electrodes)
    assert recording.dtype == np.float32
    assert np.all(np.isfinite(recording[:: int(FS_HZ)]))
    # Loose guard against pathological regressions; the target on an idle
    # 32-core machine is well under a minute (see the logged numbers).
    assert elapsed_s < 300.0
    assert peak_bytes < 4 * 1024**3
