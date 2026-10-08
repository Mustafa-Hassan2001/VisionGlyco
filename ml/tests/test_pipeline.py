import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(ML_DIR / "tools"))

import make_synthetic_dataset  # noqa: E402
import predict  # noqa: E402
import train  # noqa: E402
from visionglyco.data import load_labels, split_by_patient  # noqa: E402
from visionglyco.metrics import classification_metrics, regression_metrics  # noqa: E402
from visionglyco.preprocess import load_eye_image  # noqa: E402


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    make_synthetic_dataset.main(["--out", str(out), "--patients", "40", "--size", "96"])
    return out


def test_preprocess_center_crop(tmp_path):
    # Left half red, right half blue, 200x100. Centre crop keeps 80px around the middle.
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    arr[:, :100] = (255, 0, 0)
    arr[:, 100:] = (0, 0, 255)
    Image.fromarray(arr).save(tmp_path / "x.png")
    out = load_eye_image(tmp_path / "x.png", image_size=8, crop_fraction=0.8)
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8
    assert out[0, 0, 0] == 255 and out[0, -1, 2] == 255


def test_split_has_no_patient_overlap(synthetic):
    df = load_labels(synthetic / "labels.csv", synthetic / "images", "hba1c")
    splits = split_by_patient(df, "hba1c", "classify", 6.5, 0.15, 0.15, seed=1)
    sets = [set(s.patients) for s in splits.values()]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(s) for s in splits.values()) == len(df)


def test_missing_column_is_rejected(tmp_path):
    pd.DataFrame({"image": ["a.jpg"], "patient_id": ["1"]}).to_csv(tmp_path / "l.csv", index=False)
    with pytest.raises(ValueError, match="hba1c"):
        load_labels(tmp_path / "l.csv", tmp_path, "hba1c")


def test_classification_metrics():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9], threshold=0.5, n_bootstrap=50)
    assert m["accuracy"] == 0.75
    assert m["sensitivity"] == 1.0 and m["specificity"] == 0.5
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}


def test_regression_metrics_iso_zone():
    m = regression_metrics([80, 200], [94, 240])  # 14 mg/dL ok below 100; 20% too far above
    assert m["within_iso15197_percent"] == 50.0
    assert not m["meets_iso15197"]


@pytest.mark.slow
def test_train_export_predict_end_to_end(synthetic, tmp_path):
    app = tmp_path / "app"
    (app / "src" / "ml").mkdir(parents=True)
    meta = train.main([
        "--labels", str(synthetic / "labels.csv"), "--images-dir", str(synthetic / "images"),
        "--output-dir", str(tmp_path / "out"), "--weights", "none", "--image-size", "64",
        "--batch-size", "16", "--epochs-head", "2", "--epochs-finetune", "1",
        "--install-to-app", str(app), "--synthetic-data",
    ])
    assert meta["synthetic_data"] is True and meta["clinically_validated"] is False
    assert meta["tflite_max_abs_diff"] < 0.02
    assert (app / "assets" / "model" / "visionglyco.tflite").stat().st_size > 0
    assert "require('../../assets/model/visionglyco.tflite')" in (app / "src" / "ml" / "bundledModel.ts").read_text()
    saved = json.loads((tmp_path / "out" / "model_metadata.json").read_text())
    assert saved["input"]["size"] == 64

    image = next((synthetic / "images").iterdir())
    results = predict.main(["--model-dir", str(tmp_path / "out"), str(image)])
    assert 0.0 <= results[0]["probability"] <= 1.0
