from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridtools.io import AUDIAN_COLUMNS, read_labels, read_v1_detections, write_labels

AUDIAN = """category,kind,channel,t_start_s,t_end_s,f_low_hz,f_high_hz,note
labeled_region,span,,1142.500978,1416.790750,683.939,960.186,
chirp,span,,1148.079259,1148.203771,749.194,837.838,
chirp,span,3,1148.707894,1148.829369,751.304,832.913,loud
"""


def test_read_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(AUDIAN)
    df = read_labels(path)
    assert list(df["category"]) == ["labeled_region", "chirp", "chirp"]
    assert df["channel"].isna().tolist() == [True, True, False]
    assert df["note"].tolist() == ["", "", "loud"]
    assert df["t_start_s"].dtype == np.float64


def test_write_read_roundtrip_keeps_extra_columns(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(AUDIAN)
    df = read_labels(path)
    df["emitter_id"] = pd.array([pd.NA, 2, 3], dtype="Int64")
    out = tmp_path / "out.csv"
    write_labels(df, out)
    back = read_labels(out)
    assert list(back.columns) == [*AUDIAN_COLUMNS, "emitter_id"]
    pd.testing.assert_frame_equal(
        back.drop(columns="emitter_id"), df.drop(columns="emitter_id")
    )
    assert back["emitter_id"].tolist()[1:] == [2, 3]
    assert not list(tmp_path.glob(".out.csv.*"))


def test_write_fills_missing_audian_columns(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "t_start_s": [1.0],
            "t_end_s": [1.1],
            "f_low_hz": [700.0],
            "f_high_hz": [800.0],
        }
    )
    write_labels(df, tmp_path / "x.csv")
    assert list(read_labels(tmp_path / "x.csv").columns) == list(AUDIAN_COLUMNS)


def test_malformed_boxes_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(AUDIAN.replace("1148.203771", "1140.0"))
    with pytest.raises(ValueError, match="t_end_s < t_start_s"):
        read_labels(path)


V1 = """recording,batch,window,spec,box_ident,raw_indices,freq_range,x1,y1,x2,y2,t1,f1,t2,f2,score,emitter_eodf,track_id
r,3.0,4.0,4.0,8,"(1, 2)","(500, 1500)",1,1,2,2,610.5,739.4,610.6,853.8,0.8,771.0,9.0
r,3.0,4.0,4.0,6,"(1, 2)","(500, 1500)",1,1,2,2,610.2,720.9,610.3,928.1,0.9,765.8,
"""


def test_read_v1_detections(tmp_path: Path) -> None:
    (tmp_path / "chirpdetector_bboxes.csv").write_text(V1)
    df = read_v1_detections(tmp_path)
    assert df["t_start_s"].tolist() == [610.2, 610.5]
    assert df["track_id"].isna().tolist() == [True, False]
    assert df["track_id"].dtype == "Int64"
    assert df.loc[1, "emitter_hz"] == pytest.approx(771.0)
