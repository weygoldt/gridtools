<p align="center">
  <img src="assets/logo.svg" width="120" alt="gridtools logo">
</p>

# gridtools

Tools for multi-electrode grid recordings of wave-type weakly electric fish:
reading recordings, electrode geometry and frequency tracks, and simulating
grid recordings with moving, chirping fish for training and testing
detectors.

> **Status:** version 2 is being rebuilt on the `v2` branch. The code used for
> the chirp competition paper (2024) is available as tag
> `paper-chirp-competition-v1`.

## Installation

```sh
uv add "gridtools @ git+https://github.com/weygoldt/gridtools@v2"
```

The name `gridtools` on PyPI belongs to an unrelated package, so install from
GitHub.

## Development

```sh
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

## License

MIT
