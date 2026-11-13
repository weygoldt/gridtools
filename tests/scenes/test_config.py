"""Scene configuration: defaults, TOML/JSON round trips, validation."""

import json
from pathlib import Path

import pytest

from gridtools.sim.config import (
    TEMPLATES_ENV,
    ChirpConfig,
    HybridConfig,
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
        noise=NoiseConfig(clip_limit=3.0, harder_db=(0.1, 0.30000000000000004)),
    )
    text = config.to_json()
    assert SceneConfig.from_json(text) == config
    assert json.loads(text)["noise"]["clip_limit"] == 3.0
    no_clip = SceneConfig(noise=NoiseConfig(clip_limit=None))
    assert SceneConfig.from_json(no_clip.to_json()).noise.clip_limit is None
    assert SceneConfig().noise.clip_limit == 0.02048


def test_toml_name_defaults_to_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "my_scenes.toml"
    path.write_text(
        "duration_s = 10\n[population]\nn_fish = [2, 2]\n"
        "[movement.light]\nmean_rest_s = 300\n"
    )
    config = SceneConfig.from_toml(path)
    assert config.name == "my_scenes"
    assert config.duration_s == 10.0
    assert isinstance(config.duration_s, float)
    assert config.population.n_fish == (2, 2)
    assert config.chirps == ChirpConfig()
    assert config.movement.light.mean_rest_s == 300.0
    # a partial table falls back to MovementParams defaults, not the light regime's
    assert config.movement.light.rest_turn_factor == 0.2


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
        ({"noise": {"harder_db": [10.0]}}, ValueError),
        ({"noise": {"floor_modes": [[-120.0, 1.0]]}}, ValueError),
        ({"movement": {"per": "trial"}}, ValueError),
        ({"field": {"ridge_exponents": [2.0, 1.5]}}, ValueError),
        ({"footprint": {"background": "noise"}}, ValueError),
        ({"grid": {"reference_prob": 2.0}}, ValueError),
        ({"samplerate_hz": 2000.0}, ValueError),  # too low for 1100 Hz fish
        ({"grid": []}, TypeError),
    ],
)
def test_invalid_configs_are_rejected(data: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        SceneConfig.from_dict(data)


def test_sections_validate_on_construction() -> None:
    with pytest.raises(ValueError, match="baseline_uniform_hz"):
        PopulationConfig(baseline_uniform_hz=(1000.0, 900.0))
    with pytest.raises(ValueError, match="clip_limit"):
        NoiseConfig(clip_limit=0.0)


def test_n_samples() -> None:
    assert SceneConfig(duration_s=1.5, samplerate_hz=20_000.0).n_samples == 30_000


# -- portable defaults: nothing points to the development machine ----------


def test_shipped_default_has_no_machine_paths() -> None:
    text = default_config_path().read_text()
    for prefix in ("/mnt/", "/home/", "/run/media/", "chirp_v2"):
        assert prefix not in text


def test_default_scene_is_synthetic_without_a_background_library() -> None:
    config = SceneConfig()
    assert config.background.library is None
    assert config.background.prob == 0.0
    with pytest.raises(ValueError, match=r"background\.library"):
        HybridConfig(prob=0.6)
    assert HybridConfig(library="/libs/background", prob=0.6).prob == 0.6


def test_template_library_defaults_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TEMPLATES_ENV, raising=False)
    assert ChirpConfig().library is None
    assert SceneConfig.from_toml(default_config_path()).chirps.library is None
    monkeypatch.setenv(TEMPLATES_ENV, "/libs/templates")
    assert ChirpConfig().library == "/libs/templates"
    assert SceneConfig.from_toml(default_config_path()).chirps.library == (
        "/libs/templates"
    )
    # an explicit key wins, and the resolved config records the path
    explicit = SceneConfig.from_dict({"chirps": {"library": "/other"}})
    assert explicit.chirps.library == "/other"
    assert SceneConfig.from_json(SceneConfig().to_json()).chirps.library == (
        "/libs/templates"
    )


def test_missing_template_library_says_what_to_provide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gridtools.sim.chirps import TemplatePool

    monkeypatch.delenv(TEMPLATES_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match=TEMPLATES_ENV) as err:
        TemplatePool.open(ChirpConfig())
    assert "[chirps] library" in str(err.value)
