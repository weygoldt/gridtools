"""Random streams of a scene: the only place where seeds become generators.

Every building block of :mod:`gridtools.sim` takes a ``numpy.random.Generator``
and never seeds one itself. Scene assembly derives all of them here from one
``SeedSequence(seed)`` per scene: each component (and each fish) draws from
its own child sequence with a fixed spawn key, so changing one component's
config does not reshuffle the others.
"""

import numpy as np

__all__ = ["FISH_STREAMS", "SEED_SCHEME", "STREAMS", "fish_stream", "stream"]

#: How generators are derived (recorded in ``scene.json``).
SEED_SCHEME = "numpy.random.default_rng(numpy.random.SeedSequence(seed, spawn_key=key))"

#: Spawn keys of the scene-level streams.
STREAMS: dict[str, tuple[int, ...]] = {
    "grid": (0,),
    "population": (1,),
    "shared_drift": (2,),
    "responses": (3,),
    "noise": (4,),
    "hum": (5,),
    "channels": (6,),
    "faults": (7,),
    "clicks": (8,),
    "levels": (9,),
}

#: Last element of the per-fish spawn keys ``(100, fish_id, k)``.
FISH_STREAMS: dict[str, int] = {
    "drift": 0,
    "harmonics": 1,
    "body": 2,
    "movement": 3,
    "timing": 4,
    "templates": 5,
    "dips": 6,
}


def stream(seed: int, name: str) -> np.random.Generator:
    """Generator of a scene-level component (key from :data:`STREAMS`)."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=STREAMS[name]))


def fish_stream(seed: int, fish_id: int, name: str) -> np.random.Generator:
    """Generator of one fish's component (key ``(100, fish_id, FISH_STREAMS[name])``)."""
    key = (100, fish_id, FISH_STREAMS[name])
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))
