"""Scene configuration: defaults, TOML/JSON round trips, validation."""

import json
from pathlib import Path

import pytest

from gridtools.sim.config import (
    ChirpConfig,
    NoiseConfig,
    PopulationConfig,
    SceneConfig,
    default_config_path,
)


def test_shipped_default_toml_equals_the_dataclass_defaults() -> None:
    assert SceneConfig.from_toml(default_config_path()) == SceneConfig()


def test_json_round_trip_is_exact() -> None:
    config = SceneConfig(
        duration_s=7.5,
        noise=NoiseConfig(clip_limit=3.0, snr_db=(0.1, 0.30000000000000004)),
    )
    text = config.to_json()
    assert SceneConfig.from_json(text) == config
    assert json.loads(text)["noise"]["clip_limit"] == 3.0
    assert SceneConfig.from_json(SceneConfig().to_json()).noise.clip_limit is None


def test_toml_name_defaults_to_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "my_scenes.toml"
    path.write_text("duration_s = 10\n[population]\nn_fish = [2, 2]\n")
    config = SceneConfig.from_toml(path)
    assert config.name == "my_scenes"
    assert config.duration_s == 10.0
    assert isinstance(config.duration_s, float)
    assert config.population.n_fish == (2, 2)
    assert config.chirps == ChirpConfig()


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ({"duraton_s": 3}, ValueError),  # typo
        ({"population": {"n_fihs": [1, 2]}}, ValueError),
        ({"population": {"drift": {"sd": 1.0}}}, ValueError),
        ({"duration_s": -1.0}, ValueError),
        ({"duration_s": "long"}, TypeError),
        ({"population": {"n_fish": [3, 2]}}, ValueError),
        ({"population": {"n_fish": [1.5, 2]}}, TypeError),
        ({"population": {"close_pair_prob": 1.5}}, ValueError),
        ({"chirps": {"amplitude_mode": "guess"}}, ValueError),
        ({"chirps": {"families": ["huge"]}}, ValueError),
        ({"chirps": {"dip": [0.0, 1.2]}}, ValueError),
        ({"chirps": {"equalize": 1}}, TypeError),
        ({"grid": {"layouts": [[3, 0]]}}, ValueError),
        ({"noise": {"snr_db": [10.0]}}, ValueError),
        ({"samplerate_hz": 2000.0}, ValueError),  # too low for 1100 Hz fish
        ({"grid": []}, TypeError),
    ],
)
def test_invalid_configs_are_rejected(data: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        SceneConfig.from_dict(data)


def test_sections_validate_on_construction() -> None:
    with pytest.raises(ValueError, match="baseline_hz"):
        PopulationConfig(baseline_hz=(1000.0, 900.0))
    with pytest.raises(ValueError, match="clip_limit"):
        NoiseConfig(clip_limit=0.0)


def test_n_samples() -> None:
    assert SceneConfig(duration_s=1.5, samplerate_hz=20_000.0).n_samples == 30_000
