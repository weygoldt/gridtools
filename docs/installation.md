# Installation

gridtools needs Python 3.13 or later.

```sh
uv pip install "gridtools @ git+https://github.com/weygoldt/gridtools"
# or
pip install "gridtools @ git+https://github.com/weygoldt/gridtools"
# into a uv project
uv add "gridtools @ git+https://github.com/weygoldt/gridtools"
```

The PyPI package named `gridtools` is unrelated; always install from GitHub.

| extra | adds | needed for |
|---|---|---|
| `plot` | matplotlib | the annotated scene spectrograms (`gridtools sim run` without `--no-plots`, `gridtools sim plot`), the reports of the template and background libraries, the calibration figures |

The background library's field-recording readers decode WavPack files with
`ffmpeg`, which must be on the `PATH` for those datasets.

## From source

```sh
git clone https://github.com/weygoldt/gridtools
cd gridtools
uv sync
uv run gridtools --help
```

See [Development](development.md) for the checks and the tests.
