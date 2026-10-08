"""Package the public-iris-dataset model for the VisionGlyco app.

    python experiments/iridology/export_app_model.py \\
        --data-zip /tmp/iridology/Data.zip --install-to-app ../mobile

Builds one TFLite model (EfficientNet-B0 features + the logistic regression from
results/model.json folded into a sigmoid Dense layer) that takes an unwrapped iris
strip. The app produces that strip from an eye photo with mobile/src/ml/polar.ts,
which mirrors visionglyco/iris_polar.py.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from run_experiment import load_dataset  # noqa: E402
from visionglyco.export import (  # noqa: E402
    METADATA_FILENAME,
    MODEL_FILENAME,
    check_parity,
    export_tflite,
    install_to_app,
    tflite_predict,
    write_metadata,
)
from visionglyco.iris_polar import (  # noqa: E402
    PUPIL_DEFAULT,
    PUPIL_MAX,
    PUPIL_MIN,
    estimate_pupil_radius,
    unwrap_iris,
    wrap_strip,
)

SOURCE_SIZE = 320      # the iris crop is resized to this square before unwrapping
CROP_FRACTION = 0.8    # guide circle diameter / shorter side of the camera photo
CITATION = ("P. Moradi, N. Nazer, A. Khasahmadi, H. Mohammadzadeh, H. Khojasteh Jafari. "
            "Discovering Informative Regions in Iris Images to Predict Diabetes. ICBME 2018.")


def build_keras_model(params, height, width):
    import keras
    from keras import layers

    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                             input_shape=(height, width, 3), pooling="avg")
    coef = np.array(params["coef"])
    mean = np.array(params["scaler_mean"])
    scale = np.array(params["scaler_scale"])
    # logit = sum(coef * (f - mean) / scale) + intercept, folded into one Dense layer.
    kernel = (coef / scale)[:, None].astype(np.float32)
    bias = np.array([params["intercept"] - np.sum(coef * mean / scale)], dtype=np.float32)

    inputs = keras.Input((height, width, 3), name="iris_strip")
    feats = base(inputs, training=False)
    head = layers.Dense(1, activation="sigmoid", name="probability")
    outputs = head(feats)
    head.set_weights([kernel, bias])
    return keras.Model(inputs, outputs, name="visionglyco_iris"), base


def write_js_fixture(path):
    """Small input/output pair proving polar.ts matches iris_polar.py."""
    rng = np.random.default_rng(7)
    size, height, width = 32, 9, 24
    ys, xs = np.mgrid[0:size, 0:size]
    rel = np.hypot(xs - (size - 1) / 2, ys - (size - 1) / 2) / (size / 2)
    img = rng.integers(90, 200, (size, size, 3)).astype(np.uint8)
    img[rel < 0.3] = 20
    rgba = np.concatenate([img, np.full((size, size, 1), 255, np.uint8)], axis=2)
    pupil = estimate_pupil_radius(rgba)
    strip = unwrap_iris(rgba, pupil, height, width)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "size": size, "height": height, "width": width,
        "rgba": rgba.reshape(-1).tolist(), "pupil": pupil,
        "strip": [round(float(v), 4) for v in strip.reshape(-1)],
    }) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-zip", required=True)
    p.add_argument("--results-dir", default=str(HERE / "results"))
    p.add_argument("--out", default="outputs/iridology_app")
    p.add_argument("--install-to-app", metavar="MOBILE_DIR")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads((Path(args.results_dir) / "model.json").read_text())
    results = json.loads((Path(args.results_dir) / "results.json").read_text())
    height, width = params["input"]["height"], params["input"]["width"]

    images, labels = load_dataset(args.data_zip)
    model, base = build_keras_model(params, height, width)

    # 1. Folded Dense layer == scaler + logistic regression.
    sample = np.concatenate([images[:8], images[-8:]])
    feats = base.predict(sample, verbose=0)
    z = ((feats - params["scaler_mean"]) / params["scaler_scale"]) @ np.array(params["coef"]) + params["intercept"]
    sklearn_prob = 1 / (1 + np.exp(-z))
    keras_prob = model.predict(sample, verbose=0).reshape(-1)
    fold_diff = float(np.max(np.abs(sklearn_prob - keras_prob)))
    assert fold_diff < 1e-4, f"folded head differs by {fold_diff}"

    # 2. TFLite == Keras.
    tflite_path = out / MODEL_FILENAME
    export_tflite(model, tflite_path)
    tflite_diff = check_parity(model, tflite_path, sample, tolerance=0.02)

    # 3. App path on all people: strip -> round eye image (random pupil size) ->
    #    pupil estimate + unwrap -> model, compared with the model on the strip itself.
    rng = np.random.default_rng(0)
    true_pupils = rng.uniform(0.25, 0.5, len(images))
    direct_prob = tflite_predict(tflite_path, images)
    app_prob, pupil_err = [], []
    for strip, true_pupil in zip(images.astype(np.uint8), true_pupils):
        eye = wrap_strip(strip, SOURCE_SIZE, true_pupil)
        pupil = estimate_pupil_radius(eye)
        pupil_err.append(abs(pupil - true_pupil))
        app_prob.append(tflite_predict(tflite_path, unwrap_iris(eye, pupil, height, width)[None])[0])
    app_prob = np.array(app_prob)
    agree = float(np.mean((app_prob >= 0.5) == (direct_prob >= 0.5)))
    print(f"Folded head max diff {fold_diff:.2e}; TFLite max diff {tflite_diff:.4f}")
    print(f"App path on {len(images)} re-wrapped eyes: pupil error {np.mean(pupil_err):.3f}, "
          f"same decision as on the original strip for {agree:.1%}")

    deep = results["results"]["efficientnet_b0"]
    n = results["dataset"]["people"]
    metadata = {
        "schema_version": 1,
        "model_file": MODEL_FILENAME,
        "task": "classify",
        "backbone": "efficientnet_b0",
        "input": {
            "size": SOURCE_SIZE, "crop_fraction": CROP_FRACTION,
            "channels": "RGB", "dtype": "float32", "range": [0, 255],
            "polar": {"height": height, "width": width,
                      "pupil_min": PUPIL_MIN, "pupil_max": PUPIL_MAX, "pupil_default": PUPIL_DEFAULT},
        },
        "output": {
            "type": "probability", "threshold": 0.5,
            "positive_label": "Diabetic iris pattern",
            "negative_label": "No diabetic pattern",
            "target_column": "diabetes", "target_threshold": 1,
        },
        "test_metrics": {
            "n": n, "n_positive": results["dataset"]["diabetic"], "threshold": 0.5,
            "accuracy": deep["accuracy"]["mean"],
            "accuracy_ci95": None,
            "balanced_accuracy": deep["balanced_accuracy"]["mean"],
            "auc": deep["auc"]["mean"],
            "sensitivity": deep["sensitivity"]["mean"],
            "specificity": deep["specificity"]["mean"],
            "majority_baseline_accuracy": results["majority_baseline_accuracy"],
        },
        "evaluation": f"{results['evaluation']}: each of the {n} people was tested by a model that never saw them.",
        "target_accuracy": 0.85,
        "meets_target": deep["accuracy"]["mean"] >= 0.85,
        "dataset": {"train": {"images": n, "patients": n},
                    "val": {"images": 0, "patients": 0},
                    "test": {"images": n, "patients": n}},
        "source": {"name": "Farabi Hospital iris dataset (public research data)",
                   "citation": CITATION, "url": results["dataset"]["source"]},
        "synthetic_data": False,
        "clinically_validated": False,
        "tflite_max_abs_diff": tflite_diff,
        "trained_at": results["run_at"],
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_metadata(out / METADATA_FILENAME, metadata)

    if args.install_to_app:
        assets = install_to_app(out, args.install_to_app)
        write_js_fixture(Path(args.install_to_app) / "src/ml/__tests__/fixtures/polar_fixture.json")
        print(f"Installed model into {assets}")


if __name__ == "__main__":
    main()
