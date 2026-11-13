# Development

gridtools is developed together with
[chirpdetector](https://github.com/weygoldt/chirpdetector) in the
chirp_competition_upgrade workspace (private), which holds both as git
submodules and pins one environment for both (`make check` runs all checks
of both packages).

In a standalone checkout:

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
```

Test markers: `slow` (long-running) and `data` (needs the competition
recordings; skipped without them). The tests check physics and geometry:
the instantaneous frequency of a synthesized EOD equals f(t), re-rendering
is exact, the dipole falloff, bit-exact seed reproducibility, chirps at
scene edges.

## Documentation

The documentation is plain Markdown in `docs/` (index:
[docs/README.md](README.md)), read on GitHub. `tests/test_docs.py` checks
that every relative link in `README.md` and `docs/*.md` points to an
existing file and, for a `#fragment`, to an existing heading.

## Conventions

- Every random draw comes from an explicit `numpy.random.Generator`; scene
  assembly derives them all from one `SeedSequence` per scene, one child per
  component.
- Units in field names (`_s`, `_hz`, `_m`, `_db`); configs are frozen
  dataclasses, validated on construction; unknown keys are errors.
- No pydantic in the data path.
