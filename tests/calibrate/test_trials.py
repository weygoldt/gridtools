"""Trial discovery, metadata files and tolerant raw access."""

from pathlib import Path

import numpy as np
import pytest

from gridtools.calibrate.trials import (
    ADC_STEP_V,
    ANALYSED_TRIAL_DATES,
    open_raw,
    read_temperatures,
    read_trial_summary,
    resolve_trial_dir,
    sign_v_nfft,
)

from .conftest import CONFIG


def test_analysed_trials() -> None:
    assert len(ANALYSED_TRIAL_DATES) == 33
    assert len(set(ANALYSED_TRIAL_DATES)) == 33
    assert pytest.approx(6.25e-7) == ADC_STEP_V


def test_sign_v_nfft_from_first_frame() -> None:
    assert sign_v_nfft([0.8192, 1.1469], 20000.0) == 32768
    assert sign_v_nfft([0.4096, 0.5325], 20000.0) == 16384
    with pytest.raises(ValueError, match="power-of-two"):
        sign_v_nfft([0.5, 0.6], 20000.0)


def test_resolve_trial_dir(tmp_path: Path) -> None:
    (tmp_path / "2019-11-25-09_59").mkdir()
    (tmp_path / "2019-11-26-10_00").mkdir()
    assert resolve_trial_dir(tmp_path, "2019-11-25").name == "2019-11-25-09_59"
    with pytest.raises(FileNotFoundError):
        resolve_trial_dir(tmp_path, "2019-11-27")


def test_read_trial_summary(tmp_path: Path) -> None:
    (tmp_path / "trial_summary.csv").write_text(
        ",recording,group,win_fish,lose_fish,win_ID,lose_ID,sex_win,sex_lose,size_win,size_lose\n"
        "0,2019-11-25-09_59,3,1,2,10.0,6.0,f,m,13.2,12.0\n",
        encoding="utf-8",
    )
    meta = read_trial_summary(tmp_path)["2019-11-25-09_59"]
    assert (meta.winner_id, meta.loser_id) == (10, 6)
    assert meta.role(6) == "loser"
    assert meta.sex(10) == "f"
    assert meta.role(3) == "other"


def test_read_temperatures(tmp_path: Path) -> None:
    (tmp_path / "temperatures.csv").write_text(
        "time/s; T/C    \n     0;  25.500\n   300;  25.375\n", encoding="utf-8"
    )
    result = read_temperatures(tmp_path)
    assert result is not None
    np.testing.assert_allclose(result[0], [0.0, 300.0])
    np.testing.assert_allclose(result[1], [25.5, 25.375])
    assert read_temperatures(tmp_path / "missing") is None


def test_open_raw_drops_trailing_partial_frame(tmp_path: Path) -> None:
    (tmp_path / "fishgrid.cfg").write_text(CONFIG, encoding="utf-8")
    data = np.arange(9 * 100, dtype=np.float32).reshape(100, 9)
    raw = data.tobytes() + b"\x00" * 12
    (tmp_path / "traces-grid1.raw").write_bytes(raw)
    rec = open_raw(tmp_path)
    assert rec.n_channels == 9
    assert rec.n_samples == 100
    np.testing.assert_array_equal(rec.read(0, 100), data)
